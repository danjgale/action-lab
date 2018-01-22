"""Normalization workflow using FSL's FLIRT and FNIRT"""

import os
import sys

if sys.platform == 'linux':
    import matplotlib
    matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import nipype
from nipype import logging
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces import fsl, spm
from nipype.interfaces.utility import IdentityInterface, Function
from nilearn.plotting import plot_anat


def registration_report(fn, in_file, target=None, nslices=8,
                        title=None):

    if target is None:
        module_path = os.path.dirname(__file__)
        target = os.path.join(module_path, '../../resources/MNI152_T1_2mm_brain.nii')

    fig, ax = plt.subplots(3, 1, figsize=(20, 25))
    ax[0].set_title(title, fontsize=30)

    for i, j in enumerate(['x', 'y', 'z']):
        plot_anat(
            in_file, draw_cross=False, cut_coords=nslices,
            display_mode=j, axes=ax[i]
        ).add_edges(target)

    # save off subplot figure into png
    fig.savefig(fn)

def _get_linear_transform(node_name, dof=12, bins=None):
    """ FLIRT node for getting transformation matrix for coregistration"""
    transform = fsl.FLIRT()
    transform.inputs.dof = dof
    transform.inputs.interp = 'trilinear'
    transform.inputs.terminal_output = 'file'
    if bins is not None:
        transform.inputs.bins = bins
    transform.inputs.cost_func = 'mutualinfo'
    return Node(transform, name=node_name)


def _apply_linear_transform(node_name):
    """FLIRT node for apply an existing transform"""
    transform = fsl.FLIRT()
    transform.inputs.apply_xfm = True
    transform.inputs.terminal_output = 'file'
    return MapNode(transform, name=node_name, iterfield='in_file')


def _concat_transforms(node_name):
    convert = fsl.ConvertXFM()
    convert.inputs.concat_xfm = True
    return Node(convert, name=node_name)

def _get_file(directory, endswith='nii.gz'):

    file_ = [os.path.join(directory, i)
                for i in os.listdir(directory) if i.endswith(endswith)]

    if len(file_) > 1:
        raise ValueError('Too many matching files found in directory. Check directory')
    elif len(file_) < 1:
        raise ValueError('No matching file found in directory. Check directory')
    else:
        return file_[0]

def MNI152_T1_2mm_config():
    # no mask added for now
    args = {
        'subsampling_scheme': [4, 4, 2, 2, 1, 1],
        'max_nonlin_iter': [5, 5, 5, 5, 5, 10],
        'in_fwhm': [8, 6, 5, 4.5, 3, 2],
        'ref_fwhm': [8, 6, 5, 4, 2, 0]
        'intensity_mapping_model': 'global_non_linear_with_bias',
        'apply_intensity_mapping': [1, 1, 1, 1, 1, 0],
        'regularization_lambda': 'bending_energy',
        'biasfield_resolution': [50, 50, 50],
        'bias_regularization_lambda': 10000,
        'derive_from_ref': False,
        'fieldcoeff_file': True # not in config but needed in pipeline
    }
    return args


class Normalizer:

    def __init__(self, sub_id, data_dir, output_dir,
                 t1, t2, t2_ref, standard=None):

        self.sub_id = sub_id
        self.data_dir = data_dir

        self.output_dir = output_dir
        self.__working_dir = os.path.join(self.output_dir, 'working')
        self.__datasink_dir = os.path.join(self.output_dir, 'output')


        # typically ends with *CNS_SAG_MPRAGE_*.nii.gz
        self.t1 = t1
        self.t2_ref = t2_ref

        if isinstance(t2, str):
            self.t2 = [os.path.join(os.path.join(t2), i)
                       for i in os.listdir(os.path.join(self.data_dir, self.sub_id, t2))
                       if 'nii' in i]
        else:
            # assume as list (to be specified in docs)
            self.t2 = t2

        if standard is None:
            # default to MNI
            module_path = os.path.dirname(__file__)
            self.standard = os.path.abspath(
                os.path.join(module_path, '../../resources/MNI152_T1_2mm_brain.nii')
            )
        else:
            self.standard = standard

        self.__is_nonlinear = None


    def build_nonlinear(self, parameterize_output=False,
                        fnirt_kwargs={'fieldcoeff_file': True}):

        self.__is_nonlinear = True
        self.parameterize_output = parameterize_output
        self.fnirt_fwhm = fnirt_fwhm
        self.fnirt_subsampling_scheme = fnirt_subsampling_scheme
        self.fnirt_warp_resolution = fnirt_warp_resolution

        nipype.config.set('execution', 'remove_unnecessary_outputs', 'true')

        # setup subject's data folder
        self.__sub_output_dir = os.path.join(
            self.__datasink_dir,
            self.sub_id
        )
        if not os.path.exists(self.__sub_output_dir):
            os.makedirs(self.__sub_output_dir)

        self.datasink = Node(
            DataSink(
                base_directory=self.__datasink_dir,
                container=self.__sub_output_dir,
                substitutions=[('_subject_id_', ''), ('sub_id_', '')],
                parameterization=self.parameterize_output
            ),
            name='datasink'
        )

        # ---------------------------------------------------------------------
        # ANATOMICAL NORMALIZATION SUBFLOW
        #
        # Make subflow so that FNIRT only runs once on the anatomical scan,
        # rather than running once per iteration due to multiple functional
        # runs
        # ---------------------------------------------------------------------

        self.__normalize_anat_workflow = Workflow(name='norm_anat')
        self.__normalize_anat_workflow.base_dir = self.__working_dir

        self.anat_files = Node(
            SelectFiles(
                {'t1': os.path.join(self.data_dir, self.sub_id, self.t1),
                 'standard': self.standard
                }
            ),
            name='anat_files'
        )

        # normalization nodes
        self.anat_transform = _get_linear_transform('anat_transform')
        self.nonlinear_transform = Node(
            fsl.FNIRT(**fnirt_kwargs), name='nonlinear_transform'
        )
        self.normalize_anat = Node(fsl.ApplyWarp(), name='normalize_anat')


        # intra-normalization workflow
        self.__normalize_anat_workflow.connect([
            # transform anat to mni
            (self.anat_transform, self.nonlinear_transform, [
                ('out_matrix_file', 'affine_file')
            ]),
            (self.nonlinear_transform, self.normalize_anat, [
                ('fieldcoeff_file', 'field_file')
            ])
        ])

        # data flow
        self.__normalize_anat_workflow.connect([
            (self.anat_files, self.anat_transform, [
                ('t1', 'in_file'),
                ('standard', 'reference')
            ]),
            (self.anat_files, self.nonlinear_transform, [
                ('t1', 'in_file'),
                ('standard', 'ref_file')
            ]),
            (self.anat_files, self.normalize_anat, [
                ('t1', 'in_file'),
                ('standard', 'ref_file')
            ]),
            # output
            (self.nonlinear_transform, self.datasink, [
                ('fieldcoeff_file', 'normalized.field'),
                ('warped_file', 'normalized.fnirt_warped_file')
            ]),
            (self.normalize_anat, self.datasink, [
                ('out_file', 'normalized.anat')
            ])
        ])


        # ---------------------------------------------------------------------
        # FUNCTIONAL NORMALIZATION SUBFLOW
        #
        # Coregister T2 scans and normalize (field coeff to be supplied by
        # joining norm_anat workflow)
        # ---------------------------------------------------------------------

        self.__normalize_func_workflow = Workflow(name='norm_func')
        self.__normalize_func_workflow.base_dir = self.__working_dir


        self.infosource = Node(
            IdentityInterface(
                fields=['t2_files']
            ),
            name='infosource'
        )
        self.infosource.iterables = [('t2_files', self.t2)]

        self.select_files = Node(
            SelectFiles(
                {'t1': os.path.join(self.data_dir, self.sub_id, self.t1),
                 't2_ref': os.path.join(self.data_dir, self.sub_id, self.t2_ref),
                 't2_files': os.path.join(self.data_dir, self.sub_id, '{t2_files}'),
                 'standard': self.standard}
            ),
            name='select_files'
        )

        # nodes only necessary for coregistration
        self.coregister_transform = _get_linear_transform('coregister_transform')
        self.coregister = _apply_linear_transform('coregister')

        self.normalize_func = MapNode(fsl.ApplyWarp(), name='normalize_func',
                                      iterfield=['in_file'])

        self.normalize_motion_ref = Node(fsl.ApplyWarp(), name='normalize_motion_ref')

        # intra-norm/coreg nodes
        self.__normalize_func_workflow.connect([

            # transform funct to anat
            (self.coregister_transform, self.coregister, [
                ('out_matrix_file', 'in_matrix_file')
            ]),
            # transform funct to mni
            (self.coregister_transform, self.normalize_func, [
                ('out_matrix_file', 'premat')
            ]),
            (self.coregister_transform, self.normalize_motion_ref, [
                ('out_matrix_file', 'premat'),
            ])
        ])

        # data flow
        self.__normalize_func_workflow.connect([
            (self.infosource, self.select_files, [('t2_files', 't2_files')]),
            (self.select_files, self.coregister_transform, [
                ('t1', 'reference'),
                ('t2_ref', 'in_file')
            ]),
            (self.select_files, self.coregister, [
                ('t1', 'reference'),
                ('t2_ref', 'in_file')
            ]),
            (self.select_files, self.normalize_func, [
                ('t2_files', 'in_file'),
                ('standard', 'ref_file')
            ]),
            (self.select_files, self.normalize_motion_ref, [
                ('t2_ref', 'in_file'),
                ('standard', 'ref_file')
            ]),
            # output
            (self.coregister, self.datasink, [
                ('out_file', 'registered')
            ]),
            (self.coregister_transform, self.datasink, [
                ('out_matrix_file', 'registered.@mat')
            ]),
            (self.normalize_func, self.datasink, [
                ('out_file', 'normalized.@func')
            ]),
            (self.normalize_motion_ref, self.datasink, [
                ('out_file', 'normalized.motion_ref')
            ])
        ])

        # ---------
        # META-FLOW
        # ---------

        self.workflow = Workflow('nonlinear_normalize')
        self.workflow.base_dir = self.__working_dir

        self.workflow.connect([
            (self.__normalize_anat_workflow, self.__normalize_func_workflow, [
                ('nonlinear_transform.fieldcoeff_file', 'normalize_func.field_file'),
                ('nonlinear_transform.fieldcoeff_file', 'normalize_motion_ref.field_file')
            ])
        ])

        return self


    def build_linear(self, parameterize_output=False):

        self.__is_nonlinear = False
        self.parameterize_output = parameterize_output

        nipype.config.set('execution', 'remove_unnecessary_outputs', 'true')
        self.workflow = Workflow(name='linear_normalize')
        self.workflow.base_dir = self.__working_dir

        # ----------
        # Data Input
        # ----------

        self.infosource = Node(
            IdentityInterface(
                fields=['t2_files']
            ),
            name='infosource'
        )
        self.infosource.iterables = [('t2_files', self.t2)]

        self.select_files = Node(
            SelectFiles(
                {'t1': os.path.join(self.data_dir, self.sub_id, self.t1),
                 't2_ref': os.path.join(self.data_dir, self.sub_id, self.t2_ref),
                 't2_files': os.path.join(self.data_dir, self.sub_id, '{t2_files}'),
                 'standard': self.standard}
            ),
            name='select_files'
        )

        # -----------
        # Data Output
        # -----------

        # setup subject's data folders
        self.__sub_output_dir = os.path.join(
            self.__datasink_dir,
            self.sub_id
        )

        if not os.path.exists(self.__sub_output_dir):
            os.makedirs(self.__sub_output_dir)

        self.datasink = Node(
            DataSink(
                base_directory=self.__datasink_dir,
                container=self.__sub_output_dir,
                substitutions=[('_subject_id_', ''), ('sub_id_', '')],
                parameterization=self.parameterize_output
            ),
            name='datasink'
        )

        # -------------------
        # Normalization nodes
        # -------------------

        # nodes only necessary for coregistration
        self.coregister_transform = _get_linear_transform('coregister_transform')
        self.coregister = _apply_linear_transform('coregister')

        # extra nodes to complete normalization
        self.anat_transform = _get_linear_transform('anat_transform')
        self.normalize_anat = _apply_linear_transform('normalize_anat')
        self.concat = _concat_transforms('concat')
        self.normalize_func = _apply_linear_transform('normalize_func')

        # -------------------------------
        # Intra-normalization connections
        # -------------------------------

        self.workflow.connect([
            (self.coregister_transform, self.coregister, [
                ('out_matrix_file', 'in_matrix_file')
            ]),
            (self.coregister_transform, self.concat,
             [('out_matrix_file', 'in_file')]),
            (self.anat_transform, self.normalize_anat, [
                ('out_matrix_file', 'in_matrix_file')
            ]),
            (self.anat_transform, self.concat, [
             ('out_matrix_file', 'in_file2')]),
            (self.concat, self.normalize_func,
             [('out_file', 'in_matrix_file')])
        ])

        # ---------
        # Data flow
        # ---------

        self.workflow.connect([
            (self.infosource, self.select_files, [('t2_files', 't2_files')]),
            (self.select_files, self.coregister_transform, [
                ('t1', 'reference'),
                ('t2_ref', 'in_file')
            ]),
            (self.select_files, self.coregister, [
               ('t1', 'reference'),
                ('t2_ref', 'in_file')
            ]),

            (self.select_files, self.anat_transform, [
                ('t1', 'in_file'),
                ('standard', 'reference')
            ]),
            (self.select_files, self.normalize_anat, [
                ('t1', 'in_file'),
                ('standard', 'reference')
            ]),
            (self.select_files, self.normalize_func, [
                ('t2_files', 'in_file'),
                ('standard', 'reference')
            ]),

            # output
            (self.coregister, self.datasink, [
                ('out_file', 'registered')
            ]),
            (self.coregister_transform, self.datasink, [
                ('out_matrix_file', 'registered.mat')
            ]),
            (self.anat_transform, self.datasink, [
                ('out_matrix_file', 'normalized.mat')
            ]),
            (self.concat, self.datasink, [
                ('out_file', 'normalized.mat.@norm')
            ]),
            (self.normalize_anat, self.datasink, [
                ('out_file', 'normalized.anat')
            ]),
            (self.normalize_func, self.datasink, [
                ('out_file', 'normalized.@func')
            ])
        ])

        return self

    def run(self, parallel=True, print_header=True, n_procs=8):

        if print_header:
            print('=' * 30 + 'SUBJECT {}'.format(self.sub_id) + '=' * 30)

        if parallel:
            self.workflow.run('MultiProc', plugin_args={'n_procs': n_procs})
        else:
            self.workflow.run()

        return self

    def make_reports(self):

        self.__report_dir = os.path.join(self.__sub_output_dir, 'reports')

        if not os.path.exists(self.__report_dir):
            os.makedirs(self.__report_dir)

        raw_t1 = _get_file(os.path.join(self.data_dir, self.sub_id, 'anatomical'))
        coreg_t2 = _get_file(os.path.join(self.__sub_output_dir, 'registered'))
        normed_t1 = _get_file(os.path.join(self.__sub_output_dir, 'normalized/anat'))
        normed_t2 = _get_file(os.path.join(self.__sub_output_dir, 'normalized/motion_ref'))

        registration_report(os.path.join(self.__report_dir, 't1_to_mni.png'),
                            normed_t1, title='T1w to MNI Normalization')
        registration_report(os.path.join(self.__report_dir, 't2_to_t1.png'),
                            coreg_t2, raw_t1, title='Coregistration')
        registration_report(os.path.join(self.__report_dir, 't2_to_mni.png'),
                            normed_t2, title='T2w to MNI Normalization')


def apply_registration(self, files, output_dir, coreg_matrix, t1_affine,
                       field_file=None):
    """Take transformations matrices from previous coregistrations and/or
    normalizations and apply to set of files. Mainly useful for
    transforming model stat-maps
    """

    # if field_file none, do linear normalization, if not none, do
    # nonlinear transformation
    pass