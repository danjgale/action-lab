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
from nilearn.image import smooth_img
from nibabel import nifti1

from .base import BaseProcessor


class Preprocessor(BaseProcessor):

    def __init__(self, sub_id, input_data, output_path,
                 anatomical='*CNS_SAG_MPRAGE_*.nii.gz', zipped=True,
                 input_file_endswith=None, sort_input_files=True):

        BaseProcessor.__init__(self, sub_id, input_data, output_path, zipped,
                               input_file_endswith,
                               sort_input_files=sort_input_files)

        self.anatomical = anatomical


    @staticmethod
    def _get_motion_params(plot_type, name='motion_plot'):
        return MapNode(
            fsl.PlotMotionParams(
                in_source='fsl',
                plot_type=plot_type,
            ),
            name=name,
            iterfield='in_file'
        )

    def build(self, TR, fwhm=5.0, bet_center=None, bet_frac=0.5, bet_gradient=0,
              motion_ref_volume=4, workflow_name='preprocessing'):

        self.TR = TR
        self.bet_center = bet_center # x y z in voxel coordinates
        self.fwhm = fwhm
        self.bet_frac = bet_frac
        self.bet_gradient = bet_gradient
        self.parameterize_output = parameterize_output
        self.motion_ref_volume = motion_ref_volume
        self.workflow_name = workflow_name


        nipype.config.set('execution', 'remove_unnecessary_outputs', 'true')
        self.workflow = Workflow(name=self.workflow_name)
        self.workflow.base_dir = self._working_dir

        # ----------
        # Data Input
        # ----------

        self.infosource = Node(
            IdentityInterface(
                fields=['functionals', 'anatomical', 'first_funct']
            ),
            name='infosource'
        )
        self.infosource.inputs.anatomical = self.anatomical
        self.infosource.inputs.first_funct = self._input_files[0]
        self.infosource.iterables = [('functionals', self._input_files)]

        self.select_files = Node(
            SelectFiles(
                {'funct': '{functionals}',
                 'anat': '{anatomical}',
                 'first_funct': '{first_funct}'}
            ),
            name='select_files'
        )

        # -----------------
        # Motion Correction
        # -----------------

        self.motion_ref = Node(
            fsl.utils.ExtractROI(
                t_min=self.motion_ref_volume,
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
        self.plot_rot = self._get_motion_params('rotations', 'rot_plot')
        self.plot_trans = self._get_motion_params('translations', 'trans_plot')

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
                frac=self.bet_frac,
                vertical_gradient=self.bet_gradient,
            ),
            name='skullstrip'
        )

        if self.bet_center is not None:
            self.skullstrip.inputs.center = self.bet_center

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
            (self.motion_ref, self.motion_correction, self.motion_ref_to_motion_correction),
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
        self.selectfiles_to_motion_ref = [('first_funct', 'in_file')]
        self.select_files_to_motion_correction = [('funct', 'in_file')]

        self.motion_ref_to_datasink = [('roi_file', 'motion_corrected.ref')]
        self.skullstrip_to_datasink = [
            ('out_file', 'anatomical'),
            ('mask_file', 'anatomical.mask')
        ]
        self.motion_correction_to_datasink = [
            ('out_file', 'motion_corrected'),
            ('par_file', 'motion_corrected.par')
        ]
        self.plot_disp_to_datasink = [('out_file', 'motion_corrected.disp_plots')]
        self.plot_rot_to_datasink = [('out_file', 'motion_corrected.rot_plots')]
        self.plot_trans_to_datasink = [('out_file', 'motion_corrected.trans_plots')]
        self.slicetime_to_datasink = [('slice_time_corrected_file', 'slice_time_corrected')]
        self.smooth_to_datasink = [('smoothed_files', 'smoothed')]


        self.workflow.connect([
            (self.infosource, self.select_files, [
                ('functionals', 'functionals'),
                ('anatomical', 'anatomical'),
                ('first_funct', 'first_funct'),
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
            (self.plot_trans, self.datasink, self.plot_trans_to_datasink),
            (self.plot_rot, self.datasink, self.plot_rot_to_datasink),
            (self.slicetime, self.datasink, self.slicetime_to_datasink),
            (self.smooth, self.datasink, self.smooth_to_datasink)
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


def compute_fsl_sigma(cutoff, TR, const=2):
    """Convert filter cutoff from seconds to sigma required by fslmaths"""
    return cutoff / (TR*const)


def spatially_smooth(input_files, fwhm, output_dir=None):

    image_list = [smooth_img(i, fwhm) for i in input_files]

    if output_dir is None:
        return image_list
    else:
        [nifti1.save(j, os.path.join(output_dir, 'smoothed_{}'.format(input_files[i])))
        for i, j in enumerate(image_list)]


class Filter(BaseProcessor):

    def __init__(self, sub_id, input_data, output_path, zipped=True, smooth=True,
                 input_file_endswith=None, sort_input_files=True):
        """Spatially and temporally filter data using a separate workflow

        Using a separate workflow permits spatial/temporal filtering after
        normalization (i.e. an SPM-style workflow).
        """

        BaseProcessor.__init__(self, sub_id, input_data, output_path, zipped,
                               input_file_endswith,
                               sort_input_files=sort_input_files)
        self.smooth = smooth


    def build(self, fwhm=[5, 5, 5], highpass=100, TR=2, workflow_name='filter'):

        nipype.config.set('execution', 'remove_unnecessary_outputs', 'true')
        self.workflow = Workflow(name=workflow_name)
        self.workflow.base_dir = self._working_dir

        self.fwhm = fwhm
        self.highpass_sigma = compute_fsl_sigma(highpass, TR)

        # ----------
        # Data Input
        # ----------

        self.infosource = Node(
            IdentityInterface(
                fields=['functionals']
            ),
            name='infosource'
        )
        self.infosource.iterables = [('functionals', self._input_files)]

        # -----------------------
        # Basic Filtering Workflow
        # -----------------------

        # nodes for unsmoothed data pipeline
        self.temp_filter = Node(
            fsl.maths.TemporalFilter(
                highpass_sigma = self.highpass_sigma,
                output_type='NIFTI',
            ),
            name='temp_filter'
        )

        self.workflow.connect([
            (self.infosource, self.temp_filter, [
                ('functionals', 'in_file')
            ]),
            (self.temp_filter, self.datasink, [
                ('out_file', 'filtered')
            ])
        ])

        # -----------------------
        # Smoothing Workflow
        # -----------------------

        if self.smooth:

            self.postsmooth_temp_filter = Node(
                fsl.maths.TemporalFilter(
                    highpass_sigma = self.highpass_sigma,
                    output_type='NIFTI',
                ),
                name='postsmooth_temp_filter'
            )

            if self.zipped:
                self.gunzip = Node(
                    Gunzip(),
                    name='gunzip'
                )
                self.spatial_smooth = Node(
                    spm.Smooth(
                        fwhm=self.fwhm
                    ),
                    name='spatial_smooth',
                )
                self.workflow.connect([
                    (self.infosource, self.gunzip, [
                        ('functionals', 'in_file')
                    ]),
                    (self.gunzip, self.spatial_smooth, [
                        ('out_file', 'in_files')
                    ]),
                    (self.spatial_smooth, self.postsmooth_temp_filter, [
                        ('smoothed_files', 'in_file')
                    ]),
                    (self.postsmooth_temp_filter, self.datasink, [
                        ('out_file', 'smoothed_filtered')
                    ])
                ])

            else:
                self.spatial_smooth = Node(
                    spm.Smooth(
                        fwhm=self.fwhm
                    ),
                    name='spatial_smooth'
                )
                self.workflow.connect([
                    (self.infosource, self.spatial_smooth, [
                        ('functionals', 'in_files')
                    ]),
                    (self.spatial_smooth, self.postsmooth_temp_filter, [
                        ('smoothed_files', 'in_file')
                    ]),
                    (self.postsmooth_temp_filter, self.datasink, [
                        ('out_file', 'smoothed_filtered')
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


