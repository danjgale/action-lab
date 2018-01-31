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


    def extract(self, roi_numbers, load=True):

        if isinstance(roi_numbers, int):
            roi_numbers = [roi_numbers]

        self.files = []
        if load:
            self.masks = {}
        else:
            self.masks = None

        for i in self.__hem_list:

            hem_dict = {}
            for j in roi_numbers:

                fn = os.path.join(self.roi_dir, '{}_ROI_{}_2mm.nii.gz'.format(i, j))
                self.files.append(fn)

                if load:
                    hem_dict[j] = nifti1.load(fn)

            self.masks[i] = hem_dict

        return self







# class ROIExtractor:

#     def __init__(self):
#         pass

#     def extract(self):
#         pass
