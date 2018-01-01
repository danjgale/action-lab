"""Encapsulation of a standard preprocessing pipeline that uses nipype to
connect to SPM and FSL.
"""

import os
import sys
import argparse
import numpy as np
import nipype
from nipype import logging
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces import fsl, spm
from nipype.interfaces.fsl.utils import ExtractROI
from nipype.interfaces.utility import IdentityInterface, Function
from nipype.algorithms.misc import Gunzip


class Preprocessor:

    def __init__(self, sub_id, data_dir, functionals, working_dir, datasink_dir,
                 anatomical='*CNS_SAG_MPRAGE_*.nii.gz'):

        self.sub_id = sub_id
        self.data_dir = os.path.abspath(data_dir)
        self.functionals = functionals
        self.working_dir = os.path.abspath(working_dir)
        self.datasink_dir = os.path.abspath(datasink_dir)
        self.anatomical = anatomical

    @staticmethod
    def _get_motion_params(plot_type, name='motion_plot'):
        return MapNode(
            fsl.PlotMotionParams(
                in_source='fsl',
                plot_type=plot_type,
            )
            name=name,
            iterfield='in_file'
        )

    def build(self, TR, parameterize_output=False, fwhm=5.0,
              skullstrip_frac=0.5, skullstrip_gradient=0):

        self.TR = TR
        self.fwhm = fwhm
        self.skullstrip_frac = skullstrip_frac,
        self.skullstrip_gradient = skullstrip_gradient
        self.parameterize_output = parameterize_output


        nipype.config.set('execution', 'remove_unnecessary_outputs', 'true')
        self.workflow = Workflow(name='preprocessing')
        self.workflow.base_dir = self.working_dir

        # ----------
        # Data Input
        # ----------

        self.infosource = Node(
            IdentityInterface(
                fields=['sub_id', 'functionals', 'anatomical', 'first_funct']
            ),
            iterables=['functionals'],
            name='infosource'
        )
        self.infosource.inputs.sub_id = self.sub_id
        self.infosource.inputs.anatomical = self.anatomical
        self.infosource.inputs.first_funct = self.functionals[0]
        self.infosource.iterables = [('functionals', self.functionals)]


        self.select_files = Node(
            SelectFiles(
                {'funct': os.path.join(self.data_dir, '{sub_id}/{functionals}'),
                 'anat': os.path.join(self.data_dir, '{sub_id}/{anatomical}'),
                 'first_funct': os.path.join(self.data_dir, '{sub_id}/{first_funct}')},
                name='select_files'
            )
        )

        # -----------
        # Data Output
        # -----------

        # setup subject's data folder
        self._sub_output_dir = os.path.join(
            config.data_sink_dir,
            config.sub_id
        )

        if not os.path.exists(self._sub_output_dir):
            os.makedirs(self._sub_output_dir)

        self.datasink = Node(
            DataSink(
                base_dir=self.datasink_dir,
                container=self._sub_output_dir,
                substitutions=[('_subject_id_', ''), ('sub_id_', '')],
                parameterization=self.parameterize_output
            ),
            name='datasink'
        )

        # -----------------
        # Motion Correction
        # -----------------

        self.motion_ref = Node(
            fsl.utils.ExtractROI(
                t_min=config.motion_ref_volume,
                t_size=1
            ),
            name='motion_ref'
        )

        self.motion_correction = MapNode(
            fsl.MCFLIRT(
                cost='mutualinfo',
                save_plots=True
            ),
            name='motion',
            iterfield='in_file'
        )

        self.plot_disp = self._get_motion_params('displacement', 'disp_plot')
        self.plot_rotation = self._get_motion_params('rotations', 'rot_plot')
        self.plot_translation = self._get_motion_params('translations', 'trans_plot')

        # ---------------------
        # Slice Time Correction
        # ---------------------

        self.slicetime = MapNode(
            fsl.SliceTimer(
                interleaved=True,
                time_repetition=self.TR,
                output_type='NIFTI' # for SPM
            ),
            iterfield='in_file',
            name='slicetime'
        )

        # -----------------
        # Spatial Smoothing
        # -----------------

        self.smooth = MapNode(
            spm.Smooth(
                fwhm=self.fwhm
            ),
            iterfield='in_files',
            name='smooth'
        )

        # ---------------
        # Skull Stripping
        # ---------------

        self.skullstrip = Node(
            fsl.BET(
                robust=True,
                mask=True,
                frac=self.skullstrip_frac,
                gradient=self.skullstrip_gradient,
            ),
            name='skullstrip'
        )

        # -------------------------
        # Build preprocessing nodes
        # -------------------------

        # make these accessible so that they can be changed alongside changes
        # to actual nodes (i.e. using SPM for motion correction instead of fsl
        # uses different input/output names) should this be desired.
        self.motion_ref_to_motion_correction = [('roi_file', 'ref_file')]
        self.motion_to_slicetime = [('out_file', 'in_file')]
        self.motion_correction_to_plot = [('par_file', 'in_file')]
        self.slicetime_to_smooth = [('slice_time_corrected_file', 'in_files')]

        self.workflow.connect([
            (self.motion_ref, self.motion, self.motion_ref_to_motion_correction),
            (self.motion_correction, self.slicetime, self.motion_to_slicetime),
            (self.motion_correction, self.plot_disp, self.motion_correction_to_plot),
            (self.motion_correction, self.plot_rot, self.motion_correction_to_plot),
            (self.motion_correction, self.plot_trans, self.motion_correction_to_plot),
            (self.slicetime, self.smooth, self.slicetime_to_smooth)
        ])

        # ---------------
        # Data flow nodes
        # ---------------

        # make class methods for same reason as above
        self.selectfiles_to_skullstrip = [('anat', 'in_file')]
        self.selectfiles_to_motion_ref = [('first_run', 'in_file')]
        self.select_files_to_motion_correction = [('func', 'in_file')]

        self.motion_ref_to_datasink = [('roi_file', 'motion_corrected.ref')]
        self.skullstrip_to_datasink = [
            ('out_file', 'structual'),
            ('mask_file', 'structual.mask')
        ]
        self.motion_correction_to_datasink = [
            ('out_file', 'motion_corrected'),
            ('par_file', 'motion_corrected.par')
        ]
        self.plot_disp_to_datasink = [('out_file', 'motion_corrected.disp_plots')]
        self.plot_rot_to_datasink = [('out_file', 'motion_corrected.rot_plots')]
        self.plot_trans_to_datasink = [('out_file', 'motion_corrected.trans_plots')]
        self.slicetime_to_datasink = [('slice_time_corrected_file', 'time_corrected')]
        self.smooth_to_datasink = [('smoothed_files', 'smoothed')]


        self.workflow.connect([
            (self.infosource, self.select_files, [
                ('sub_id', 'sub_id'),
                ('task_runs', 'task_runs'),
                ('first_run', 'first_run'),
                ('rest_runs', 'rest_runs')
            ]),
            # inputs
            (self.select_files, self.skullstrip, self.selectfiles_to_skullstrip),
            (self.select_files, self.motion_ref, self.selectfiles_to_motion_ref),
            (self.select_files, self.motion_correction,
             self.select_files_to_motion_correction),
            # outputs
            (self.motion_ref, self.datasink, self.motion_ref_to_datasink),
            (self.skullstrip, self.datasink, self.skullstrip_to_datasink),
            (self.motion_correction, self.datasink, self.motion_correction_to_datasink),
            (self.plot_disp, self.datasink, self.plot_disp_to_datasink),
            (self.plot_translation, self.datasink, self.plot_trans_to_datasink),
            (self.plot_rotation, self.datasink, self.plot_rot_to_datasink),
            (self.slicetime, self.datasink, self.slicetime_to_datasink),
            (self.smooth, self.datasink, self.smooth_to_datasink)
        ])

    def run(self, parallel=True, print_header=True, n_procs=8):

        if print_header:
            print('=' * 30 + 'SUBJECT {}'.format(self.sub_id) + '=' * 30)

        if parallel:
            self.workflow.run('MultiProc', plugin_args={'n_procs': n_procs})
        else:
            self.workflow.run()
