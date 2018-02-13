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


def extract_voxels(roi_img, data, output_fn=None, average=False):
    """Get timecourse for every voxel in a single ROI"""
    masker = MultiNiftiMasker(roi_img, n_jobs=-1)
    voxels = np.vstack(masker.fit_transform(data))

    if average:
        voxels = np.mean(voxels, axis=1)

    if output_fn is None:
        return voxels
    else:
        print('Writing {}...'.format(output_fn))
        np.savetxt(output_fn, voxels, delimiter=',')


def voxels_to_df(fn, labels):
    roi_name = os.path.splitext(os.path.basename(fn))[0]
    df = pd.read_csv(labels, header=None)
    df['roi'] = roi_name
    print(df.shape)

    voxels = pd.Series(list(np.loadtxt(fn,  delimiter=',')), name='voxels')
    print(voxels.shape)

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
                          average=False):
        """Generate ROI voxel arrays and store in directory.

        roi_imgs is a dict with ROI label as the key and a nifti-like img
        as the value
        """
        for k, v in roi_imgs.items():
            extract_voxels(v, data_imgs,
                           os.path.join(self.path, '{}.csv'.format(k)), average)


        timecourse_labels.to_csv(
            os.path.join(self.path, timecourse_labels),
            header=False,
            index=False
        )


    def create_from_coords(self):
        pass


    def create_from_df(self):
        pass


    def load(self, rois=None, label_file='labels.csv', concat=True, filetype='.csv'):

        if filetype is not None:
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
                for i in os.listdir(self.path) if i is not label_file
            ]

        if len(rois) == 1:
            return roi_list[0]

        if concat:
            roi_list = pd.concat(roi_list)

        return roi_list


# class ROIExtractor:

#     def __init__(self):
#         pass

#     def extract(self):
#         pass
