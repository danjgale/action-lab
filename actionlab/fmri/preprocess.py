"""Encapsulation of a standard preprocessing pipeline that uses nipype to
connect to SPM and FSL.
"""

import os
import sys
import argparse
import glob
import gzip
import shutil
import pandas as pd
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
from nilearn.input_data import MultiNiftiMasker

from .base import BaseProcessor
from .roi import binarize_mask_array


class Preprocessor(BaseProcessor):

    def __init__(self, sub_id, input_data, anatomical, output_path, zipped=True,
                 input_file_endswith=None, sort_input_files=True, save_all=False):

        BaseProcessor.__init__(self, sub_id, input_data, output_path, zipped,
                               input_file_endswith,
                               sort_input_files=sort_input_files)

        self.anatomical = anatomical
        self._save_all = save_all


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
                fields=['funct', 'anat', 'first_funct']
            ),
            name='infosource'
        )
        self.infosource.inputs.anat = self.anatomical
        self.infosource.inputs.first_funct = self._input_files[0]
        self.infosource.iterables = [('funct', self._input_files)]

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

        self.motion_correction = Node(
            fsl.MCFLIRT(
                cost='mutualinfo',
                save_plots=True
            ),
            name='motion'
        )

        self.plot_disp = self._get_motion_params('displacement', 'disp_plot')
        self.plot_rot = self._get_motion_params('rotations', 'rot_plot')
        self.plot_trans = self._get_motion_params('translations', 'trans_plot')

        # ---------------------
        # Slice Time Correction
        # ---------------------

        self.slicetime = Node(
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

        self.smooth = Node(
            spm.Smooth(
                fwhm=self.fwhm
            ),
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
        self.infosource_to_skullstrip = [('anat', 'in_file')]
        self.infosource_to_motion_ref = [('first_funct', 'in_file')]
        self.infosource_to_motion_correction = [('funct', 'in_file')]

        self.motion_ref_to_datasink = [('roi_file', 'motion_corrected.ref')]
        self.skullstrip_to_datasink = [
            ('out_file', 'anatomical'),
            ('mask_file', 'anatomical.mask')
        ]
        self.motion_correction_to_datasink = [
            ('out_file', 'motion_corrected')
        ]
        self.motion_correction_params_to_datasink = [
            ('par_file', 'motion_corrected.par')
        ]
        self.plot_disp_to_datasink = [('out_file', 'motion_corrected.disp_plots')]
        self.plot_rot_to_datasink = [('out_file', 'motion_corrected.rot_plots')]
        self.plot_trans_to_datasink = [('out_file', 'motion_corrected.trans_plots')]
        self.slicetime_to_datasink = [('slice_time_corrected_file', 'slice_time_corrected')]
        self.smooth_to_datasink = [('smoothed_files', 'smoothed')]


        self.workflow.connect([
            # inputs
            (self.infosource, self.skullstrip, self.infosource_to_skullstrip),
            (self.infosource, self.motion_ref, self.infosource_to_motion_ref),
            (self.infosource, self.motion_correction,
             self.infosource_to_motion_correction),
            # default outputs
            (self.motion_ref, self.datasink, self.motion_ref_to_datasink),
            (self.skullstrip, self.datasink, self.skullstrip_to_datasink),
            (self.plot_disp, self.datasink, self.plot_disp_to_datasink),
            (self.plot_trans, self.datasink, self.plot_trans_to_datasink),
            (self.plot_rot, self.datasink, self.plot_rot_to_datasink),
            (self.motion_correction, self.datasink, self.motion_correction_params_to_datasink),
            (self.slicetime, self.datasink, self.slicetime_to_datasink),
            (self.smooth, self.datasink, self.smooth_to_datasink)
        ])


        if self._save_all:
            # save off motion correction data (prior to slicetime correction/smoothing)
            self.workflow.connect([
                (self.motion_correction, self.datasink, self.motion_correction_to_datasink)
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

    if any([i.endswith('.gz') for i in input_files]):
        # uncompress nifti files for SPM
        compressed = True
        tmp_file_list = []
        for i in input_files:

            # Nipype gunzip workaround ----------------------------------------
            # gunzip doesn't allow you to save to path of input file...
            if i[-3:].lower() == ".gz":
                save_filename = i[:-3]
            else:
                # file is already uncompressed; skip
                continue

            print('converting {} to {}'.format(i, save_filename))
            with gzip.open(i, 'rb') as in_file:
                with open(save_filename, 'wb') as out_file:
                    shutil.copyfileobj(in_file, out_file)
            # -----------------------------------------------------------------

            tmp_file_list.append(save_filename)
        input_files = tmp_file_list
    else:
        compressed = False

    if output_dir is None:
        smooth = spm.Smooth(in_files=input_files, fwhm=fwhm, out_prefix='smoothed_')
    else:
        smooth = spm.Smooth(in_files=input_files, fwhm=fwhm, paths=output_dir,
                            out_prefix='smoothed_')
    smooth.run()

    if compressed:
        # removed temporary uncompressed nifti files if created
        [os.remove(i) for i in tmp_file_list]


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

        if self.smooth:
            # -------------------------------------
            # Smoothing Workflow
            #
            # Only occurs if smoothing is specified
            # -------------------------------------

            self.spatial_smooth = Node(
                spm.Smooth(
                    fwhm=self.fwhm
                ),
                name='spatial_smooth'
            )

            # filter nodes (same process as above, different names)
            self.mean_img_smooth = Node(
                fsl.maths.MeanImage(), name="mean_img"
            )
            self.temp_filter_smooth = Node(
                fsl.maths.TemporalFilter(
                    highpass_sigma=self.highpass_sigma,
                    output_type='NIFTI',
                ),
                name='temp_filter'
            )
            self.filter_with_mean_smooth = Node(
                fsl.ImageMaths(op_string='-add', output_type='NIFTI'),
                name="filter_with_mean"
                )


            # connect smoothed files to filter workflow
            self.workflow.connect([
                (self.spatial_smooth, self.mean_img_smooth, [
                    ('smoothed_files', 'in_file')
                ]),
                (self.spatial_smooth, self.temp_filter_smooth, [
                    ('smoothed_files', 'in_file')
                ])
            ])

            # filter workflow done post-smoothing
            self.workflow.connect([
                (self.temp_filter_smooth, self.filter_with_mean_smooth, [
                    ('out_file', 'in_file')
                ]),
                (self.mean_img_smooth, self.filter_with_mean_smooth, [
                    ('out_file', 'in_file2')
                ]),
                (self.filter_with_mean_smooth, self.datasink, [
                    ('out_file', 'smoothed_filtered')
                ])
            ])

            # ----------------------- Data input handling ---------------------
            if self.zipped:
                # need to unzip for SPM's smoothing
                self.gunzip = Node(
                    Gunzip(),
                    name='gunzip'
                )
                self.workflow.connect([
                    (self.infosource, self.gunzip, [
                        ('functionals', 'in_file')
                    ]),
                    (self.gunzip, self.spatial_smooth, [
                        ('out_file', 'in_files')
                    ]),
                ])

            else:
                self.workflow.connect([
                    (self.infosource, self.spatial_smooth, [
                        ('functionals', 'in_files')
                    ])
                ])
            # -----------------------------------------------------------------

        else:
            # -----------------------------
            # Basic Filtering Workflow
            #
            # If smoothing is not specified
            # -----------------------------

            self.mean_img = Node(fsl.maths.MeanImage(), name="mean_img")

            # nodes for unsmoothed data pipeline
            self.temp_filter = Node(
                fsl.maths.TemporalFilter(
                    highpass_sigma=self.highpass_sigma,
                    output_type='NIFTI',
                ),
                name='temp_filter'
            )

            self.filter_with_mean = Node(
                fsl.ImageMaths(op_string='-add', output_type='NIFTI'),
                name="filter_with_mean"
                )

            self.workflow.connect([
                (self.infosource, self.mean_img, [
                    ('functionals', 'in_file')
                ]),
                (self.infosource, self.temp_filter, [
                    ('functionals', 'in_file')
                ]),
                (self.temp_filter, self.filter_with_mean, [
                    ('out_file', 'in_file')
                ]),
                (self.mean_img, self.filter_with_mean, [
                    ('out_file', 'in_file2')
                ]),
                (self.filter_with_mean, self.datasink, [
                    ('out_file', 'filtered')
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


def _segment_anat(fn, output_dir):

    fast = fsl.FAST(in_files=fn, img_type=1, segments=True,
                    verbose=True, out_basename=os.path.join(output_dir, 'fast'),
                    ignore_exception=True)
    fast.base_dir = output_dir
    fast.run()

    # only way to get around nipype issue is to ignore the exception and
    # manually check if the correct files are ouputted. Only need the WM and CSF maps, but
    # 9 files are generated each time if run correctly
    output_files = [i for i in os.listdir(output_dir) if i.endswith('.nii.gz')]
    if len(output_files) != 9:
        raise Exception

    # return binary masks for WM and CSF
    return os.path.join(output_dir, 'fast_seg_2.nii.gz'), os.path.join(output_dir, 'fast_seg_0.nii.gz')


def _normalize_segment(transform, mask, mat_file=None, nonlinear=False,
                       standard='MNI152_T1_2mm_brain.nii.gz'):
    """Normalize binary segment mask. `transform` is either a coeff nifti file
    if nonlinear, or a matrix file if linear.
    """

    if nonlinear:
        # assume nonlinear normalization
        norm = fsl.ApplyWarp(
            in_file=mask,
            ref_file=fsl.Info.standard_image(standard),
            field_file=transform,
            out_file=mask
        )
    else:
       # assume linear transformation
        norm = fsl.ApplyXFM(
            in_file=mask,
            reference=fsl.Info.standard_image(standard),
            in_matrix_file=transform,
            out_file=mask,
            out_matrix_file=mat_file
        )

    norm.run()

    return None


def _binarize(img, thresh):
    binarize = fsl.ImageMaths(in_file=img, op_string='-thr {} -bin'.format(thresh), out_file=img)
    binarize.run()


def _extract_matter(runs, mask):

    masker = MultiNiftiMasker(mask, n_jobs=-1)
    voxels = masker.fit_transform(runs)

    # return average intensity of each time point
    return [np.mean(i, axis=1) for i in voxels]


class SubjectConfounds(object):

    def __init__(self, anatomical, functional_runs, output_path, motion_parameters=None):
        """ Generate confound files

        Creates confound.csv files containing regressors for mask extraction.
        Each file corresponds to one run with shape of (volumes, n regressors).
        """

        self.functional_runs = functional_runs
        self.anatomical = anatomical
        self.output_path = output_path

        if motion_parameters is not None:
            self.confounds = [pd.read_csv(i, sep="\s+", header=None) for i in motion_parameters]


            for i in self.confounds:
                i.columns = ['motion1', 'motion2', 'motion3', 'motion4',
                             'motion5', 'motion6']
        else:
            self.confounds = None


    def make_segments(self, subfolder=None):

        if subfolder is not None:
            self._outdir = os.path.join(self.output_path, subfolder)
        else:
            self._outdir = self.output_path

        if not os.path.exists(self._outdir):
            os.makedirs(self._outdir)

        print('Segmenting to {} ...'.format(self._outdir))
        # get tissue masks
        self.WM, self.CSF = _segment_anat(self.anatomical, self._outdir)

    def extract_segments(self, WM=None, CSF=None, transform=None, nonlinear=False,
                         binarization_threshold=.6):
        """Extract segment time series from functional runs. Can either use
        make_segments to get WM and CSF masks directly, or enter them as args
        if they already exist. Transform is a transformation to apply to the
        masks in order to extract functional data. Must specify if this is a
        linear or nonlinear transform.
        """
        self.binarization_threshold = binarization_threshold
        self.transform = transform

        if WM is not None:
            self.WM = WM
        if CSF is not None:
            self.CSF = CSF

        # file-based operations done in place
        if transform is not None:
            for i in [self.WM, self.CSF]:
                _normalize_segment(self.transform, i, nonlinear=nonlinear)
                _binarize(i, self.binarization_threshold)

        # get timeseries of WM and CSF for each run (returns list of runs)
        print('extracting {}'.format(self.WM))
        self.WM_timeseries = _extract_matter(self.functional_runs, self.WM)
        print('extracting {}'.format(self.CSF))
        self.CSF_timeseries = _extract_matter(self.functional_runs, self.CSF)

        # add timeseries to confounds
        if self.confounds is None:
            self.confounds = [
                pd.DataFrame({'wm': self.WM_timeseries[i], 'csf': self.CSF_timeseries[i]})
                for i in np.arange(len(self.WM_timeseries))
            ]
        else:
            for i, j in enumerate(self.confounds):
                j['wm'] = self.WM_timeseries[i]
                j['csf'] = self.CSF_timeseries[i]


    def add_constant(self):
        if self.confounds is None:
            raise Exception('Other confound regressors must be present')
        else:
            for i, j in enumerate(self.confounds):
                j['const'] = 1


    def add_linear_drift(self):
        if self.confounds is None:
            raise Exception('Other confound regressors must be present')
        else:
            for i, j in enumerate(self.confounds):
                j['lin'] = np.arange(len(j))


    def write(self):

        if self.confounds is None:
            raise AttributeError('Confounds currently not set; cannot write files.')
        else:
            for i, j in enumerate(self.confounds):
                j.to_csv(os.path.join(self.output_path, 'confound{}.csv'.format(i + 1)), index=False)
