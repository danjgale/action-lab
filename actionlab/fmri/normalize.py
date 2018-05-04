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

from .base import BaseProcessor

def registration_report(fn, in_file, target, nslices=8,
                        title=None):

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



class Normalizer(BaseProcessor):

    def __init__(self, sub_id, t1, t2, t2_ref, output_path,
                 standard='MNI152_T1_2mm_brain.nii.gz', zipped=True,
                 input_file_endswith=None, sort_input_files=True):

        # typically ends with *CNS_SAG_MPRAGE_*.nii.gz
        self.t1 = t1
        self.t2_ref = t2_ref
        self.t2 = t2

        #  *** Note that t2 is input data ***
        BaseProcessor.__init__(self, sub_id, self.t2, output_path, zipped,
                               input_file_endswith,
                               sort_input_files=sort_input_files)

        self.standard = fsl.Info.standard_image(standard)

        self.__is_nonlinear = None


    def build_nonlinear(self, parameterize_output=False, t2_t1_dof=6,
                        t2_t1_bins=None, t1_mni_dof=12, t1_mni_bins=None,
                        fnirt_config_file=None, workflow_name='nonlinear_normalize'):

        self.__is_nonlinear = True
        self.parameterize_output = parameterize_output
        self.t2_t1_dof = t2_t1_dof
        self.t2_t1_bins = t2_t1_bins
        self.t1_mni_dof = t1_mni_dof
        self.t1_mni_bins = t1_mni_bins

        # set up configuration file for FNIRT
        if fnirt_config_file == 'default':
            self.config_file = os.path.join(os.environ["FSLDIR"], "etc/flirtsch/T1_2_MNI152_2mm.cnf")
        elif fnirt_config_file is not None:
            self.config_file = fnirt_config_file
        else:
            self.config_file = None

        nipype.config.set('execution', 'remove_unnecessary_outputs', 'true')

        # ---------------------------------------------------------------------
        # ANATOMICAL NORMALIZATION SUBFLOW
        #
        # Make subflow so that FNIRT only runs once on the anatomical scan,
        # rather than running once per iteration due to multiple functional
        # runs
        # ---------------------------------------------------------------------

        self.__normalize_anat_workflow = Workflow(name='norm_anat')
        self.__normalize_anat_workflow.base_dir = self._working_dir

        self.anat_infosource = Node(
            IdentityInterface(
                fields=['t1', 'standard']
            ),
            name='infosource'
        )
        self.anat_infosource.inputs.t1 = self.t1
        self.anat_infosource.inputs.standard = self.standard

        self.anat_transform = _get_linear_transform(
            'anat_transform', self.t1_mni_dof, self.t1_mni_bins
        )

        # must handle no config file because nipype doesn't handle None arguments
        if self.config_file is None:
            self.nonlinear_transform = Node(
                fsl.FNIRT(ref_file=self.standard,
                        fieldcoeff_file=True
                ),
                name='nonlinear_transform'
            )
        else:
            self.nonlinear_transform = Node(
                fsl.FNIRT(ref_file=self.standard,
                        config_file=self.config_file,
                        fieldcoeff_file=True
                ),
                name='nonlinear_transform'
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
            (self.anat_infosource, self.anat_transform, [
                ('t1', 'in_file'),
                ('standard', 'reference')
            ]),
            (self.anat_infosource, self.nonlinear_transform, [
                ('t1', 'in_file')
            ]),
            (self.anat_infosource, self.normalize_anat, [
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
        self.__normalize_func_workflow.base_dir = self._working_dir


        self.infosource = Node(
            IdentityInterface(
                fields=['t1', 't2_ref', 't2_files', 'standard']
            ),
            name='infosource'
        )
        self.infosource.iterables = [('t2_files', self._input_files)]
        self.infosource.inputs.t1 = self.t1
        self.infosource.inputs.t2_ref = self.t2_ref
        self.infosource.inputs.standard = self.standard

        # nodes only necessary for coregistration
        self.coregister_transform = _get_linear_transform(
            'coregister_transform', self.t2_t1_dof, self.t2_t1_bins
        )
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
            (self.infosource, self.coregister_transform, [
                ('t1', 'reference'),
                ('t2_ref', 'in_file')
            ]),
            (self.infosource, self.coregister, [
                ('t1', 'reference'),
                ('t2_ref', 'in_file')
            ]),
            (self.infosource, self.normalize_func, [
                ('t2_files', 'in_file'),
                ('standard', 'ref_file')
            ]),
            (self.infosource, self.normalize_motion_ref, [
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

        self.workflow = Workflow(workflow_name)
        self.workflow.base_dir = self._working_dir

        self.workflow.connect([
            (self.__normalize_anat_workflow, self.__normalize_func_workflow, [
                ('nonlinear_transform.fieldcoeff_file', 'normalize_func.field_file'),
                ('nonlinear_transform.fieldcoeff_file', 'normalize_motion_ref.field_file')
            ])
        ])

        return self


    def build_linear(self, t2_t1_dof=6, t2_t1_bins=None, t1_mni_dof=12,
                     t1_mni_bins=None, workflow_name='linear_normalize'):

        #raise Exception('Not available for now')

        self.__is_nonlinear = False
        self.t2_t1_dof = t2_t1_dof
        self.t2_t1_bins = t2_t1_bins
        self.t1_mni_dof = t1_mni_dof
        self.t1_mni_bins = t1_mni_bins
        self.workflow_name = workflow_name

        nipype.config.set('execution', 'remove_unnecessary_outputs', 'true')
        self.workflow = Workflow(name=workflow_name)
        self.workflow.base_dir = self._working_dir

        # ----------
        # Data Input
        # ----------

        self.infosource = Node(
            IdentityInterface(
                fields=['t2_files']
            ),
            name='infosource'
        )
        self.infosource.iterables = [('t2_files', self._input_files)]
        self.infosource.inputs.t1 = self.t1
        self.infosource.inputs.t2_ref = self.t2_ref
        self.infosource.inputs.standard = self.standard

        # -------------------
        # Normalization nodes
        # -------------------

        # nodes only necessary for coregistration
        self.coregister_transform = _get_linear_transform(
            'coregister_transform', self.t2_t1_dof, self.t2_t1_bins
        )
        self.coregister = _apply_linear_transform('coregister')

        # extra nodes to complete normalization
        self.anat_transform = _get_linear_transform(
            'anat_transform', self.t1_mni_dof, self.t1_mni_bins
        )
        self.normalize_anat = _apply_linear_transform('normalize_anat')
        self.concat = _concat_transforms('concat')
        self.normalize_func = _apply_linear_transform('normalize_func')
        self.normalize_motion_ref = _apply_linear_transform('normalize_motion_ref')

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
             [('out_file', 'in_matrix_file')]),
            (self.concat, self.normalize_motion_ref, [
                ('out_file', 'in_matrix_file')
            ])
        ])

        # ---------
        # Data flow
        # ---------

        self.workflow.connect([
            (self.infosource, self.coregister_transform, [
                ('t1', 'reference'),
                ('t2_ref', 'in_file')
            ]),
            (self.infosource, self.coregister, [
               ('t1', 'reference'),
                ('t2_ref', 'in_file')
            ]),

            (self.infosource, self.anat_transform, [
                ('t1', 'in_file'),
                ('standard', 'reference')
            ]),
            (self.infosource, self.normalize_anat, [
                ('t1', 'in_file'),
                ('standard', 'reference')
            ]),
            (self.infosource, self.normalize_func, [
                ('t2_files', 'in_file'),
                ('standard', 'reference')
            ]),
            (self.infosource, self.normalize_motion_ref, [
                ('t2_ref', 'in_file'),
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
            ]),
            (self.normalize_motion_ref, self.datasink, [
                ('out_file', 'normalized.motion_ref')
            ])
        ])

        return self

    def run(self, parallel=True, print_header=True, n_procs=4):

        if print_header:
            print('=' * 30 + 'SUBJECT {}'.format(self.sub_id) + '=' * 30)

        if parallel:
            self.workflow.run('MultiProc', plugin_args={'n_procs': n_procs})
        else:
            self.workflow.run()

        return self

    def make_reports(self):

        self.__report_dir = os.path.join(self._sub_output_dir, 'reports')

        if not os.path.exists(self.__report_dir):
            os.makedirs(self.__report_dir)

        raw_t1 = self.t1
        coreg_t2 = _get_file(os.path.join(self._sub_output_dir, 'registered'))
        normed_t1 = _get_file(os.path.join(self._sub_output_dir, 'normalized/anat'))
        normed_t2 = _get_file(os.path.join(self._sub_output_dir, 'normalized/motion_ref'))

        registration_report(os.path.join(self.__report_dir, 't1_to_mni.png'),
                            normed_t1, self.standard, title='T1w to MNI Normalization')
        registration_report(os.path.join(self.__report_dir, 't2_to_t1.png'),
                            coreg_t2, raw_t1, title='Coregistration')
        registration_report(os.path.join(self.__report_dir, 't2_to_mni.png'),
                            normed_t2, self.standard, title='T2w to MNI Normalization')


def apply_registration(self, files, output_dir, coreg_matrix, t1_affine,
                       field_file=None):
    """Take transformations matrices from previous coregistrations and/or
    normalizations and apply to set of files. Mainly useful for
    transforming model stat-maps
    """

    # if field_file none, do linear normalization, if not none, do
    # nonlinear transformation
    pass
