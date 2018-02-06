"""Utilities for extracting and using ROI masks"""


import os
import sys
import re
import subprocess
import numpy as np
import nilearn
from nibabel import nifti1


def binarize_mask_array(x, threshold=None):

    if threshold is None:
        return np.where(x > 0, 1, 0)
    else:
        return np.where(x >= threshold, 1, 0)


class GlasserExtractor:

    def __init__(self, roi_dir, hemisphere='bilateral'):

        self.roi_dir = roi_dir
        self.hemisphere = hemisphere

        if self.hemisphere == 'bilateral':
            self.__hem_list = ['L', 'R']
        elif self.hemisphere == 'r':
            self.__hem_list = ['R']
        elif self.hemisphere == 'l':
            self.__hem_list = ['L']
        else:
            raise ValueError("Hemisphere argument must be either 'bilateral',"
                             " 'r', or 'l' ")


    def extract(self, roi_numbers):

        if isinstance(roi_numbers, int):
            roi_numbers = [roi_numbers]

        self.files = []
        self.rois = {}
        for i in self.__hem_list:

            hem_dict = {}
            for j in roi_numbers:

                fn = os.path.join(self.roi_dir, '{}_ROI_{}_2mm.nii.gz'.format(i, j))
                self.files.append(fn)
                hem_dict[j] = nifti1.load(fn)

            self.rois[i] = hem_dict

        return self


    def binarize(self, threshold=None, inplace=True):

        dict_ = {}
        for k, v in self.rois.items():
            tmp_dict = {}
            for inner_k, inner_v in v.items():
                tmp_dict[inner_k] = (
                    nifti1.Nifti1Pair(
                        binarize_mask_array(inner_v.get_data(), threshold),
                        inner_v.affine,
                        inner_v.header
                    )
                )

            dict_[k] = tmp_dict

        if inplace:
            self.rois = dict_
        else:
            return dict_


# class ROIExtractor:

#     def __init__(self):
#         pass

#     def extract(self):
#         pass
