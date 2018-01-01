"""Normalization workflow using FSL's FLIRT and FNIRT"""

import os
import sys
import numpy as np
import nipype
from nipype import logging
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces import fsl, spm
from nipype.interfaces.utility import IdentityInterface, Function
from utils import set_datasink


def _get_transform(node_name, dof=12, bins=None):
    """ FLIRT node for getting transformation matrix for coregistration"""
    transform = fsl.FLIRT()
    transform.inputs.dof = dof
    transform.inputs.interp = 'trilinear'
    transform.inputs.terminal_output = 'file'
    if bins is not None:
        transform.inputs.bins = bins
    transform.inputs.cost_func = 'mutualinfo'
    return Node(transform, name=node_name)


def _apply_transform(node_name):
    """FLIRT node for apply an existing transform"""
    transform = fsl.FLIRT()
    transform.inputs.apply_xfm = True
    transform.inputs.terminal_output = 'file'
    return MapNode(transform, name=node_name, iterfield='in_file')


def _concat_transforms(node_name):
    convert = fsl.ConvertXFM()
    convert.inputs.concat_xfm = True
    return Node(convert, name=node_name)

class Normalizer:

    def __init__(self, sub_id, data_dir, working_dir, datasink_dir,
                 t1, t2_ref, t2_files, t2_files_dir, standard=None):

        self.sub_id = sub_id
        self.data_dir = os.path.abspath(data_dir)
        self.datasink_dir = os.path.abspath(datasink_dir)
        self.working_dir = os.path.abspath(working_dir)

        # typically ends with *CNS_SAG_MPRAGE_*.nii.gz
        self.t1 = os.path.abspath(t1)

        self.t2_ref = os.path.abspath(t2_ref)
        self.t2_files_dir = os.path.abspath(t2_files_dir)
        self.t2_files = t2_files

        if standard is None:
            # default to MNI
            self.standard = '../../resources/MNI152_T1_2mm_brain.nii'
        else:
            self.standard = standard

    def build(self, parameterize_output=False):

        self.parameterize_output = parameterize_output

        nipype.config.set('execution', 'remove_unnecessary_outputs', 'true')
        self.workflow = Workflow(name='normalize')
        self.workflow.base_dir = self.working_dir

        # ----------
        # Data Input
        # ----------

        self.infosource = Node(
            IdentityInterface(
                fields=['t2_files']
            ),
            name='infosource'
        )
        self.infosource.iterables = [('t2_files', self.t2_files)]

        self.select_files = Node(
            SelectFiles(
                {'t1': self.t1,
                 't2_ref': self.t2_ref
                 't2_files': os.path.join(self.t2_files_dir, '{t2_files}'),
                 'standard': self.standard},
                name='select_files'
            )
        )

        # -----------
        # Data Output
        # -----------

        # setup subject's data folder
        self.__sub_output_dir = os.path.join(
            self.datasink_dir
            self.sub_id
        )

        if not os.path.exists(self.__sub_output_dir):
            os.makedirs(self.__sub_output_dir)

        self.datasink = Node(
            DataSink(
                base_dir=self.datasink_dir,
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
                ('t2_files', 'in_file')
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

    def run(self, parallel=True, print_header=True, n_procs=8):

        if print_header:
            print('=' * 30 + 'SUBJECT {}'.format(self.sub_id) + '=' * 30)

        if parallel:
            self.workflow.run('MultiProc', plugin_args={'n_procs': n_procs})
        else:
            self.workflow.run()
