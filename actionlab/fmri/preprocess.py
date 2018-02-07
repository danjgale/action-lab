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


class Preprocessor:

    def __init__(self, sub_id, data_dir, functionals, output_dir,
                 anatomical='*CNS_SAG_MPRAGE_*.nii.gz',
                 prepend_functional_path=None):

        self.sub_id = sub_id
        self.data_dir = data_dir

        if prepend_functional_path is not None:
            # add prepended path (full path required if not present)
            self.functionals = [os.path.join(prepend_functional_path, i) for i in functionals]
        else:
            # assumes that full path is already there
            self.functionals = functionals

        self.output_dir = output_dir
        self.__working_dir =  os.path.join(self.output_dir, 'working')
        self.__datasink_dir = os.path.join(self.output_dir, 'output')

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

    def build(self, TR, parameterize_output=False, bet_center=None, fwhm=5.0,
              skullstrip_frac=0.5, skullstrip_gradient=0,
              motion_ref_volume=4):

        self.TR = TR
        self.bet_center = bet_center # x y z in voxel coordinates
        self.fwhm = fwhm
        self.skullstrip_frac = skullstrip_frac
        self.skullstrip_gradient = skullstrip_gradient
        self.parameterize_output = parameterize_output
        self.motion_ref_volume = motion_ref_volume


        nipype.config.set('execution', 'remove_unnecessary_outputs', 'true')
        self.workflow = Workflow(name='preprocessing')
        self.workflow.base_dir = self.__working_dir

        # ----------
        # Data Input
        # ----------

        self.infosource = Node(
            IdentityInterface(
                fields=['sub_id', 'functionals', 'anatomical', 'first_funct']
            ),
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
                 'first_funct': os.path.join(self.data_dir, '{sub_id}/{first_funct}')}
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
                frac=self.skullstrip_frac,
                vertical_gradient=self.skullstrip_gradient,
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
                ('sub_id', 'sub_id'),
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


def spatially_smooth(input_files, fwhm, output_dir=None):

    image_list = [smooth_img(i, fwhm) for i in input_files]

    if output_dir is None:
        return image_list
    else:
        [nifti1.save(j, os.path.join(output_dir, 'smoothed_{}'.format(input_files[i])))
        for i, j in enumerate(image_list)]


class Filter:

    def __init__(self, sub_id, data_dir, functionals, output_dir,
                 prepend_functional_path=None, zipped=True, smooth=True):
        """Spatially and temporally filter data using a separate workflow

        Using a separate workflow permits spatial/temporal filtering after
        normalization (i.e. an SPM-style workflow).
        """

        self.sub_id = sub_id
        self.data_dir = data_dir
        self.zipped = zipped

        if self.zipped:
            file_extension = '.nii.gz'
        else:
            file_extension = '.nii'

        if isinstance(functionals, str):
            self.functionals = [os.path.join(os.path.join(functionals), i)
                       for i in os.listdir(os.path.join(self.data_dir, self.sub_id, functionals))
                       if i.endswith(file_extension)]
        else:
            # assume as list (to be specified in docs)
            self.functionals = functionals


        self.smooth = smooth

        self.output_dir = output_dir
        self.__working_dir =  os.path.join(self.output_dir, 'working')
        self.__datasink_dir = os.path.join(self.output_dir, 'output')

    def build(self, fwhm=[5, 5, 5], highpass=100, TR=2, highpass_units='secs',
              workflow_name='filter'):

        nipype.config.set('execution', 'remove_unnecessary_outputs', 'true')
        self.workflow = Workflow(name=workflow_name)
        self.workflow.base_dir = self.__working_dir

        self.fwhm = fwhm

        if highpass_units == 'secs':
            self.highpass = highpass / 2
        elif highpass_units == 'vols':
            self.highpass = highpass
        else:
            raise ValueError("Value for highpass_units must be either 'secs' or 'vols'")

        # ----------
        # Data Input
        # ----------

        self.infosource = Node(
            IdentityInterface(
                fields=['sub_id', 'functionals']
            ),
            name='infosource'
        )
        self.infosource.inputs.sub_id = self.sub_id
        self.infosource.iterables = [('functionals', self.functionals)]


        self.select_files = Node(
            SelectFiles(
                {'funct': os.path.join(self.data_dir, self.sub_id, '{functionals}')}
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

        # -----------------------
        # Basic Filtering Workflow
        # -----------------------

        # nodes for unsmoothed data pipeline
        self.temp_filter = Node(
            fsl.maths.TemporalFilter(
                highpass_sigma = self.highpass,
                output_type='NIFTI',
            ),
            name='presmooth_temp_filter',
            iterables='in_file'
        )

        self.workflow.connect([
            (self.infosource, self.select_files, [('functionals', 'functionals')]),
            (self.select_files, self.temp_filter, [
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
                    highpass_sigma = self.highpass,
                    output_type='NIFTI',
                ),
                name='postsmooth_temp_filter'
            )

            if self.zipped:
                self.gunzip = Node(
                    Gunzip(),
                    iterables='in_file',
                    name='gunzip'
                )
                self.spatial_smooth = Node(
                    spm.Smooth(
                        fwhm=self.fwhm
                    ),
                    name='spatial_smooth',
                )
                self.workflow.connect([
                    (self.select_files, self.gunzip, [
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
                    name='spatial_smooth',
                    iterables='in_file'
                )
                self.workflow.connect([
                    (self.select_files, self.spatial_smooth, [
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


