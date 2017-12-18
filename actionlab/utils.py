"""General/miscellaneous utility functions and classes"""

import os
import sys
import numpy as np
import pandas as pd

### helper functions for data I/O

def _get_trial(df, block, trial_number, block_col='BlockNumber',
               trial_col='TrialNumber'):
    """Returns specified trial from specified block. Assumes the block and trial
     columns are consistent across experiments"""
    return df.groupby([block_col, trial_col]).get_group((block, trial_number))


def _get_data_start(fn, verify_with='HeaderLines', add=1, sep='\t'):
    """Reads first line of a data file to determine header rows in file.

    The first line of a data file is typically a 'HeaderLines' variable, which
    indicates the number of lines dedicated to headers. This may vary from
    trial to trial (i.e. 'task' trials versus 'pause' trials).

    Parameters:
    -----------
    fn : str
        File name/path.
    verify_with : str
        Identifier to verify that the first line refers to the number of header
        lines (default is 'HeaderLines', which is standard for data files).
    add : int
        Number of lines/rows to add to header lines so that it refers to the
        line in which the data appears rather than the headers stop (default is
        1, which is most common for data files).
    sep : str
        Delimiter type. Default is tab-delimited.

    Returns:
    --------
    int
        Line number in the data file in which the data column headers
        begin
    """
    header_df = pd.read_csv(fn, header=None, nrows=1, sep=sep)

    if header_df[0].loc[0] != verify_with:
        raise Exception(
            'No {} found in first line. Check file.'.format(verify_with)
        )

    try:
        lines = int(header_df[1])
    except ValueError:
        raise ValueError("Number of lines not able to convert to an integer. "
                         "Check file to verify value.")

    # add so that lines now indicates the row in which the data starts
    return lines + add


class DataFile:
    def __init__(self, fn, sep='\t', data_start=None,
                 data_start_verify='HeaderLines', data_start_add=1):
        """ Handles single-trial data from a single .dat file.

        Note that the headers attribute is a Series containing all strings by
        default. self.change_dtype() or SubjectData.get_header() can
        conveniently chnage the data type of a header if a string is not
        desired.

        Parameters:
        -----------
        fn : str
            File name/path.
        sep : str
            Delimiter type. Default is tab-delimited.
        data_start : int, optional
            Line number in the data file in which the data column headers
            begin. If None, the line number is inferred from the first line of
            the file (default is None).
        data_start_verify : str
            Identifier to verify that the first line refers to the number of
            header lines (default is 'HeaderLines', which is standard for data
            files).
        data_start_add : int
            Number of lines/rows to add to header lines so that it refers to
            the line in which the data appears rather than the headers stop
            (default is 1, which is most common for data files).

        Attributes:
        -----------
        headers : Series
            Pandas Series containing all header data found in file.
        data : DataFrame
            Pandas DataFrame containing time-varying trial data.
        """

        self.file_name = fn
        self.data_start = data_start

        if self.data_start is None:
            # infer data start number from file
            self.data_start = _get_data_start(self.file_name, data_start_verify,
                                              data_start_add, sep=sep)

        # make correction (TODO: find way to infer this)
        header_rows = self.data_start - 4
        # drop excess column almost all filled with NANs
        headers = (pd.read_table(self.file_name, sep='\t', nrows=header_rows))
        # remove any duplicate indices
        headers = headers.drop(headers.columns[1], axis=1).squeeze()
        self.headers = headers[~headers.index.duplicated(keep='first')]

        self.data = pd.read_csv(self.file_name, sep=sep,
                                skiprows=self.data_start - 2)


class SubjectData:
    def __init__(self, path, sep='\t', data_start=None,
                 data_start_verify='HeaderLines', data_start_add=1):
        """ Class containing all trial data belonging to a single subject.

        Main class that assembles all of a subject's data, with variety of
        methods to access and combine various trials. The current
        implementation assumes that all data files have the same data start
        number. This is a safe assumption, as this is largely the
        case.

        Parameters:
        -----------
        path : str
            Path to subject folder/directory.
        sep = str
            Delimiter type. Default is tab-delimited.
        data_start : int, optional
            Line number in the data file in which the data column headers
            begin. If None, the line number is inferred from the first line of
            the file (default is None).
        data_start_verify : str
            Identifier to verify that the first line refers to the number of
            header lines (default is 'HeaderLines', which is standard for data
            files).
        data_start_add : int
            Number of lines/rows to add to header lines so that it refers to
            the line in which the data appears rather than the headers stop
            (default is 1, which is most common for data files).

        Attributes:
        -----------
        path : str
            Path to subject folder/directory.
        id : str
            Subject id as defined as the last folder by the path.
        files : list
            List of file names containing data for each trial.
        data_list : list
            List of DataFile objects for each trial.
        """

        self.path = path
        self.id = os.path.basename(os.path.normpath(path))
        self.files = [i for i in os.listdir(self.path) if i.endswith('.dat')]
        self.data_start = data_start

        # read in list of data file objects
        self.data_list = [DataFile(
            os.path.join(self.path, i),
            sep=sep,
            data_start=self.data_start,
            data_start_verify=data_start_verify,
            data_start_add=data_start_add
        )
            for i in self.files]

    def select_trials(self, header, value):
        """Identify trials based on a value in the file headers and return list
        of objects of only those trials. Important that value is a string."""
        return [i for i in self.data_list if i.headers[header] == value]

    def combine_all(self, reset_index=True):
        """Concatenates all loaded data into a single DataFrame"""
        self.data = pd.concat([i.data for i in self.data_list],
                              ignore_index=reset_index)

    def combine_subset(self, subset, reset_index=True):
        """Concatenates subset from select_trials into a single DataFrame"""
        if len(subset) == 0:
            # if subset is only one trial, cannot concatenate. So just return
            # the trial
            return subset[0].data
        else:
            return pd.concat([i.data for i in subset], ignore_index=reset_index)

    def get_trial(self, block, trial_number, block_col='BlockNumber',
                  trial_col='TrialNumber'):
        """Method implication of get_trial()"""
        return _get_trial(self.data, block, trial_number, block_col='BlockNumber',
                          trial_col='TrialNumber')

    def get_header(self, name, dtype=None, index=None):
        """Get header value from every trial"""

        if index is None:
            data = self.data_list
        else:
            data = self.data_list[index]

        list_ = []
        for i in data:
            list_.append(i.headers[name])

        if dtype is not None:
            list_ = list(map(dtype, list_))

        return list_
