""" Handy utility functions for various fMRI procedures"""

import json
import os
from datetime import datetime
import re
import pandas as pd
import subprocess
import sys

sys.path.append('../')
from utils import convert_file

def is_motion_corrected(fn):

    with open(fn) as f:
        metadata = json.load(f)

    try:
        if metadata['SeriesDescription'] == 'MoCoSeries':
            return True
        else:
            return False
    except KeyError:
        print("Please verify JSON key.")


def _scan_subject(path, file_pattern=None, use_moco=True):

    if file_pattern is not None:
        json_files = [
            i for i in os.listdir(path)
            if all(j in i for j in (file_pattern, '.json'))
        ]
    else:
        # file/string must be json and contain "Retinotopy"
        json_files = [
            i for i in os.listdir(path)
            if i.endswith('.json'))
        ]

    if use_moco:
        runs = [i for i in json_files
                if is_motion_corrected(os.path.join(path, i))]
    else:
        runs = [i for i in json_files
                if not is_motion_corrected(os.path.join(path, i))]
    return runs


def get_volumes(fn):
    """Return number of volumes in nifti"""
    return int(subprocess.check_output(['fslnvols', fn]))


def _filter_runs(data_path, run_list, vols):
    """Remove niftis from list if they do not have the correct number of volumes.
    Returns list of nifti file names.
    """

    run_files = [convert_file(i, '.json', '.nii.gz') for i in run_list]
    list_ = []
    for i, run_file in enumerate(run_files):
        nvols = get_volumes(os.path.join(data_path, run_file))

        # remove runs without specified volume(s)
        if isinstance(vols, list):
            # check if nvols is not any of the ones in the list
            if any(nvols == v for v in vols):
                list_.append(run_files[i])
        else:
            if nvols == vols:
                list_.append(run_file)

    print("n runs: {}".format(len(list_)))
    return list_


def get_run_time(fn, as_nifti=True):

    with open(fn) as f:
        metadata = json.load(f)

    time = datetime.strptime(metadata['AcquisitionTime'], '%H:%M:%S.%f')

    if as_nifti:
        fn = convert_file(fn, '.json', 'nii.gz')

    return fn, metadata['AcquisitionTime'], time


def _sort_run_times(x, show_time=True):
    """Sort runs based on acquisition times"""
    ordered_runs = sorted(x, key=lambda x: x[2])

    if show_time:
        return dict([(i[0], i[1]) for i in ordered_runs])
    else:
        return [i[0] for i in ordered_runs]


class RunManager:

    def __init__(self, subjects, data_dir, n_vols, use_moco=True,
                 file_pattern=None):

        if isinstance(subjects, str):
            subjects = [subjects]

        self.subjects = subjects
        self.subject_dirs = []

        self.n_vols = n_vols
        self.use_moco = use_moco
        self.file_pattern = file_pattern

        return self

    def gather(self):

        # dictionary containing list of nifti filenames for each subject
        self.subject_runs = {}
        for i in self.subjects:
            subject_dir = os.path.join(self.data_dir, i)
            runs = _scan_subject(subject_dir, self.file_pattern, self.use_moco)
            runs = _filter_runs(subject_dir, runs, self.n_vols)

            self.subject_runs[i] = [os.path.join(subject_dir, j) for j in runs]

        return self


    def sort(self):

        for k, v in self.subject_runs.items():
            json_files = [convert_file(i, '.nii.gz', '.json') for i in v]
            run_times = [get_run_time(i) for i in json_files]
            self.subject_runs[k] = _sort_run_times(run_times, show_time=False)

        return self

    def get_run_numbers(self, pattern=r'Vols\d+.nii'):

        self.subject_run_numbers = {}
        for k, v in self.subjects_runs:

            # extract run number out of pattern the number for
            # each run
            run_nums = [
                int(list(filter(str.isdigit, re.search(pattern, i).group())))
                for i in v
            ]

            self.subject_run_numbers[k] = run_nums


    def export(self, fn, as_numbers=False):

        if as_numbers:
            write_obj = self.subject_run_numbers
        else:
            write_obj = self.subject_runs

        with open(fn, 'w') as f:
            f.write(json.dumps(self.subject_runs, indent=2))

        return self
