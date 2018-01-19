"""Normalization workflow using FSL's FLIRT and FNIRT"""

import os
import sys
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
                        title=None, return_fig=False):

    if target is None:
        target = '../../resources/MNI152_T1_2mm_brain.nii'

    fig, ax = plt.subplots(3, 1, figsize=(20, 25))
    ax[0].set_title(title, fontsize=30)

    for i, j in enumerate(['x', 'y', 'z']):
        plot_anat(
            in_file, draw_cross=False, cut_coords=nslices,
            display_mode=j, axes=ax[i]
        ).add_edges(target)

    # save off subplot figure into png
    return fig.savefig(fn)


def _registration_report_node(fn, target=None, title=None):
    report = Function(input_names=['in_file', 'target', 'title'],
                      output_names=['fn'],
                      function=registration_report)

    report.inputs.fn = fn

    if target is not None:
        report.target = target

    report.title = title

    return report

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

def _apply_warp():

    convert =



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
            self.standard = os.path.join(module_path, '../../resources/MNI152_T1_2mm_brain.nii')
        else:
            self.standard = standard


    def build_nonlinear(self, parameterize_output=False, fnirt_fwhm=[6, 4, 2, 2],
              fnirt_subsampling_scheme=[4, 2, 1, 1],
              fnirt_warp_resolution=(10, 10, 10)):

        self.parameterize_output = parameterize_output
        self.fnirt_fwhm = fnirt_fwhm
        self.fnirt_subsampling_scheme = fnirt_subsampling_scheme
        self.fnirt_warp_resolution = fnirt_warp_resolution

        nipype.config.set('execution', 'remove_unnecessary_outputs', 'true')
        self.workflow = Workflow(name='normalize')
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

        # -------------------
        # Normalization nodes
        # -------------------

        # nodes only necessary for coregistration
        self.coregister_transform = _get_linear_transform('coregister_transform')
        self.coregister = _apply_linear_transform('coregister')

        # generate affine transformation
        self.anat_transform = _get_linear_transform('anat_transform')
        #self.normalize_anat = _apply_linear_transform('normalize_anat')
        #self.concat = _concat_transforms('concat')
        # self.normalize_func = _apply_linear_transform('normalize_func')

        # generate nonlinear normalization transform
        # to be added in workflow: in_file, ref_file, affine_file
        self.nonlinear_transform = Node(
            fsl.FNIRT(
                in_fwhm=self.fnirt_fwhm,
                subsampling_scheme=self.fnirt_subsampling_scheme,
                warp_resolution=self.fnirt_warp_resolution,
                config_file='T1_2_MNI152_2mm' # see if needed/desired
            )
        )

        self.normalize_func = MapNode(fsl.ApplyWarp(), name='normalize_func',
                                      iterfield='in_file')
        self.normalize_anat = Node(fsl.ApplyWarp(), name='normalize_anat')

        # make registration reports (motion_ref to anat, anat to mni,
        # motion_ref to mni)

        t2_t1_report = _registration_report_node('t2_t1_report.png', title='T2w to T1w')
        t1_mni_report = _registration_report_node('t1_mni_report.png', title='T1w to MNI')
        t2_mni_report = _registration_report_node('t2_mni_report.png', title='T2w to MNI')

        # -------------------------------
        # Intra-normalization connections
        # -------------------------------

        self.workflow.connect([
            # transform anat to mni
            (self.anat_transform, self.nonlinear_transform, [
                'out_matrix_file', 'affine_file'
            ]),
            self.nonlinear_transform, self.normalize_anat, [
                'fieldcoeff_file', 'field_file'
            ],

            # transform funct to anat
            (self.coregister_transform, self.coregister, [
                ('out_matrix_file', 'in_matrix_file')
            ]),
            # transform funct to mni
            (self.coregister_transform, self.normalize_func, [
                'out_matrix_file', 'premat'
            ]),
            (self.nonlinear_transform, self.normalize_func, [
                'fieldcoeff_file', 'field_file'
            ]),
            (self.normalize_anat, t1_mni_report, [
                ('out_file', 'in_file')
            ]),
            (self.coregister, t2_t1_report, [
                ('out_file', 'in_file')
            ]),
            (self.normalize_func, t2_t1_report, [
                ('out_file', 'in_file')
            ])

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
            (self.select_files, self.nonlinear_transform, [
                ('t1', 'in_file'),
                ('standard', 'ref_file')
            ])
            (self.select_files, self.normalize_anat, [
                ('t1', 'in_file'),
                ('standard', 'ref_file')
            ]),
            (self.select_files, self.normalize_func, [
                ('t2_files', 'in_file'),
                ('standard', 'ref_file')
            ]),
            (self.select_files, t2_t1_report, [
                ('t1', 'target')
            ]),
            (self.select_files, t1_mni_report, [
                ('standard', 'target')
            ]),
            (self.select_files, t2_t1_report, [
                ('standard', 'target')
            ]),

            # output
            (self.coregister, self.datasink, [
                ('out_file', 'registered')
            ]),
            (self.coregister_transform, self.datasink, [
                ('out_matrix_file', 'registered.@mat')
            ]),
            (self.nonlinear_transform, self.datasink, [
                ('fieldcoeff_file', 'normalized.@field'),
            ])
            (self.normalize_anat, self.datasink, [
                ('out_file', 'normalized.anat')
            ]),
            (self.normalize_func, self.datasink, [
                ('out_file', 'normalized.@func')
            ]),
            (t2_t1_report, self.datasink, [
                ('fn', 'registered.@t2')
            ]),
            (t1_mni_report, self.datasink, [
                ('fn', 'normalized.reports.@t1')
            ]),
            (t2_mni_report, self.datasink, [
                ('fn', 'normalized.reports.@t2')
            ])
        ])

        return self


    def build_linear(self, parameterize_output=False):

        self.parameterize_output = parameterize_output

        nipype.config.set('execution', 'remove_unnecessary_outputs', 'true')
        self.workflow = Workflow(name='normalize')
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

        # -------------------
        # Normalization nodes
        # -------------------

        # nodes only necessary for coregistration
        self.coregister_transform = _get_transform('coregister_transform')
        self.coregister = _apply_transform('coregister')

        # extra nodes to complete normalization
        self.anat_transform = _get_transform('anat_transform')
        self.normalize_anat = _apply_transform('normalize_anat')
        self.concat = _concat_transforms('concat')
        self.normalize_func = _apply_transform('normalize_func')

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
