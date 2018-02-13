"""Utilities for extracting and using ROI masks"""


import os
import sys
import re
import subprocess
import json
import numpy as np
import pandas as pd
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


def extract_voxels(roi_img, data, output_fn=None):
    """Get timecourse for every voxel in a single ROI"""
    masker = MultiNiftiMasker(roi_img, n_jobs=-1)
    voxels = np.vstack(masker.fit_transform(data))

    if output_fn is None:
        return voxels
    else:
        print('Writing {}...'.format(output_fn))
        np.savetxt(output_fn, voxels, delimiter=',')


def voxels_to_df(fn, labels):
    roi_name = os.path.splitext(os.path.basename(labels))[0]
    df = pd.read_csv(labels)

    voxels = pd.Series(list(np.loadtxt(fn)))

    if voxels.shape[0] != df.shape[0]:
        raise ValueError('Rows in voxels and time labels do not match.')

    df['voxels'] = voxels

    return df


class ROIDirectory(object):

    def __init__(self, path):

        if not os.path.exists(path):
            os.mkdir(path)

        self.path = path


    def create_from_masks(self, roi_imgs, data_imgs, timecourse_labels=None):
        """Generate ROI voxel arrays and store in directory.

        roi_imgs is a dict with ROI label as the key and a nifti-like img
        as the value
        """
        for k, v in roi_imgs.items():
            extract_voxels(v, data_imgs, os.path.join(self.path, '{}.csv'.format(k)))

        if timecourse_labels is not None:
            timecourse_labels.to_csv(
                os.path.join(self.path, timecourse_labels),
                header=False,
                index=False
            )


    def create_from_coords(self):
        pass


    def create_from_df(self):
        pass


    def load(self, rois=None, label_file='labels.csv'):

        self.__label_file_exists = False

        if os.path.exists(os.path.join(self.path, label_file)):
            self.__label_file_exists = True
            self.labels = os.path.join(self.path, label_file)

        if rois is not None:
            roi_list = [
                voxels_to_df(i, self.labels)
                for i in os.listdir(self.path) if i in rois
            ]
        else:
            roi_list = [
                voxels_to_df(i, self.labels)
                for i in os.listdir(self.path) if i is not label_file
            ]

        if concat=True:
            roi_list = pd.concat(roi_list)

        return roi_list


# class ROIExtractor:

#     def __init__(self):
#         pass

#     def extract(self):
#         pass
