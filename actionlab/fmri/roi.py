"""Utilities for extracting and using ROI masks"""


import os
import sys
import re
import subprocess
import json
import numpy as np
import pandas as pd
from nipype.interfaces.fsl import ImageMaths, Info
import nilearn
from nilearn.input_data import NiftiMasker, MultiNiftiMasker
from nibabel import nifti1


def binarize_mask_array(x, threshold=None):

    if threshold is None:
        return np.where(x > 0, 1, 0)
    else:
        return np.where(x >= threshold, 1, 0)


class GlasserAtlas:

    def __init__(self, roi_dir, hemisphere):

        self.roi_dir = roi_dir
        self.hemisphere = hemisphere


    def collect(self, roi_numbers):

        if isinstance(roi_numbers, int):
            roi_numbers = [roi_numbers]

        self.files = []
        self.rois = {}
        for i in roi_numbers:
            fn = os.path.join(
                self.roi_dir, '{}_ROI_{}_2mm.nii.gz'.format(self.hemisphere, i))
            self.files.append(fn)
            self.rois[i] = nifti1.load(fn)

        return self


    def binarize(self, threshold=None, inplace=True):

        dict_ = {}
        for k, v in self.rois.items():
            dict_[k] = nifti1.Nifti1Pair(
                binarize_mask_array(v.get_data(), threshold),
                v.affine,
                v.header
            )

        if inplace:
            self.rois = dict_
        else:
            return dict_


def extract_voxels(roi_img, data, output_fn=None, confounds=None):
    """Get timecourse for every voxel in a single ROI"""
    masker = MultiNiftiMasker(roi_img, n_jobs=-1)
    voxels = np.vstack(masker.fit_transform(data, confounds=confounds))

    if output_fn is None:
        return voxels
    else:
        print('Writing {}...'.format(output_fn))
        np.savetxt(output_fn, voxels, delimiter=',', fmt='%1.3f')


def voxels_to_df(fn, labels):
    roi_name = os.path.splitext(os.path.basename(fn))[0]
    df = pd.read_csv(labels, header=None)
    df['roi'] = roi_name
    df.rename(columns={0: 'run', 1: 'label'}, inplace=True)

    voxels = pd.Series(list(np.loadtxt(fn,  delimiter=',')), name='voxels')

    if voxels.shape[0] != df.shape[0]:
        raise ValueError('Rows in voxels and time labels do not match.')

    df['voxels'] = voxels

    return df


class ROIDirectory(object):

    def __init__(self, path):

        if not os.path.exists(path):
            os.mkdir(path)

        self.path = path


    def create_from_masks(self, roi_imgs, data_imgs, timecourse_labels=None,
                          confounds=None):
        """Generate ROI voxel arrays and store in directory.

        roi_imgs is a dict with ROI label as the key and a nifti-like img
        as the value
        """
        for k, v in roi_imgs.items():

            extract_voxels(
                v,
                data_imgs,
                os.path.join(self.path, '{}.csv'.format(k)),
                average,
                confounds
            )

        timecourse_labels.to_csv(
            os.path.join(self.path, timecourse_labels),
            header=False,
            index=False
        )


    def create_from_coords(self):
        pass


    def create_from_df(self):
        pass


    def load(self, rois=None, label_file='labels.csv', concat=False, filetype='.csv'):

        if filetype is not None and rois is not None:
            # add file extension so rois can be inputted as just labels
            rois = [i + filetype for i in rois]

        # check if file labels
        if os.path.isfile(os.path.join(self.path, label_file)):
            self.labels = os.path.join(self.path, label_file)
        else:
            raise ValueError("No label_file found.")

        if rois is not None:
            roi_list = [
                voxels_to_df(os.path.join(self.path, i), self.labels)
                for i in os.listdir(self.path) if i in rois
            ]
        else:
            roi_list = [
                voxels_to_df(os.path.join(self.path, i), self.labels)
                for i in os.listdir(self.path) if i != label_file
            ]

        if rois is None:
            pass
        elif len(rois) == 0:
            raise ValueError("No ROIs found.")
        elif len(rois) == 1:
            return roi_list[0]
        else:
            pass


        if concat:
            roi_list = pd.concat(roi_list)

        return roi_list


def MNI_to_voxels(x, y, z):
    """Convert MNI mm coordinates into voxel space on a 2mm MNI template.

    From https://www.jiscmail.ac.uk/cgi-bin/webadmin?A2=fsl;d95418af.1308
    """
    return (-x + 90)/2, (y + 126)/2, (z + 72)/2


def sphere_mask(coordinates, radius, fn, in_file=None):

    # set up temp files
    output_dir = os.path.dirname(fn)
    point_file = os.path.join(output_dir, 'point.nii.gz')
    sphere_file = os.path.join(output_dir, 'sphere.nii.gz')

    if in_file is None:
        in_file = Info.standard_image('MNI152_T1_2mm_brain.nii.gz')

    # make point from coordinate
    point_string = '-mul 0 -add 1 -roi %d 1 %d 1 %d 1 0 1' % coordinates
    point = ImageMaths(in_file=in_file, op_string=point_string, out_file=point_file,
                       out_data_type='float')
    point.run()
    # make sphere from point
    sphere = ImageMaths(in_file=point_file, out_file=sphere_file,
                        op_string='-kernel sphere %d -fmean' % radius,
                        out_data_type='float')
    sphere.run()
    # binarize sphere mask
    binarize = ImageMaths(in_file=sphere_file, op_string='-bin', out_file=fn,
                          out_data_type='float')
    binarize.run()

    # remove temp files
    os.remove(point_file)
    os.remove(sphere_file)