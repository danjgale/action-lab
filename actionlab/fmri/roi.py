"""Utilities for extracting and using ROI masks"""


import os
import sys
import re
import subprocess
import json
import numpy as np
import nilearn
from nilearn.input_data import NiftiMasker, MultiNiftiMasker
from nibabel import nifti1


def binarize_mask_array(x, threshold=None):

    if threshold is None:
        return np.where(x > 0, 1, 0)
    else:
        return np.where(x >= threshold, 1, 0)


class GlasserExtractor:

    def __init__(self, roi_dir, hemisphere):

        self.roi_dir = roi_dir
        self.hemisphere = hemisphere


    def extract(self, roi_numbers):

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


class _VoxelArrayOutput:

    def __init__(self, name, voxels, labels):
        self.name = name
        self.voxels = voxels.tolist()
        self.labels = labels

    def save(self, fn):
        with open(fn, 'w') as f:
            json.dump(self.__dict__, f, sort_keys=True, indent=2)


class VoxelArray:

    def __init__(self, fn=None):

        if fn is not None:
            self = self.load(fn)
        else:
            self.labels = None
            self.voxels = None
            self.name = None


    def create(self, mask_img, data, name, n_jobs=1):
        """Create a concatenated (volumes * runs) by voxel numpy array for all data files
        included.
        """

        try:
            len(data)
        except TypeError:
            data = [data]

        self.__data_shape = len(data)
        self.__data_ix = np.arange(self.__data_shape)

        if self.__data_shape > 1:
            masker = MultiNiftiMasker(mask_img, n_jobs=n_jobs)
            self.voxels = np.vstack(masker.fit_transform(data))
        else:
            masker = NiftiMasker(mask_img)
            self.voxels = masker.fit_transform(data[0])

        self.name = name


    def load(self, fn):

        with open(fn) as f:
            input_json = json.load(f)

        self.voxels = np.array(input_json['voxels'])
        self.labels = input_json['labels']
        self.name = input_json['name']


    def label(self, time_labels, label_column):

        if not isinstance(time_labels, list):
            self.__time_labels = [time_labels]

        if not self.__data_shape == len(self.time_labels):
            raise ValueError('The length of time label list provided does not '
                             'match the length of data list provided.')

        for i in self.__time_labels:
            i['run'] = i
            i['roi'] = self.name

        self.labels = pd.concat([i[['roi', 'run', label_column]] for i in self.__time_labels])


    def to_dataframe(self):

        if self.labels is not None:
            return pd.concat([self.labels, pd.Series(list(self.voxels), name='voxels')])
        else:
            return pd.Series(list(self.voxels), name='voxels')

    def save(self, fn):
        output = _VoxelArrayOutput(self.name, self.voxels, self.labels)
        output.save(fn)


# class ROIExtractor:

#     def __init__(self):
#         pass

#     def extract(self):
#         pass
