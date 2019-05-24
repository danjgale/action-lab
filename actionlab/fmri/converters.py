

import sys
import os
import re
import numpy as np
import nipype
from nipype.interfaces.dcm2nii import Dcm2niix
from traitlets import TraitError

def _make_path(path, return_str=False):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass

    if return_str:
        return tmp_path

# Use dcm2bids package (pip installable) for dicom to bids format

def convert_to_nifti(input_dir, output_dir, compress=True):

    if compress:
        compress_flag = 'i'
    else:
        compress_flag = 'n'

    if not os.path.exists(output_dir):
        os.makedirs(output_path)

    try:
        Dcm2niix(
            source_dir=input_dir,
            output_dir=output_dir,
            out_filename='%t%p%s',
            compress=compress_flag,
            single_file=False
        ).run()
    except Exception as e:
        # Will get a trait error after each participant
        print('{} occured; passing...'.format(e))
        pass
