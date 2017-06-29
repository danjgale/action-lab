"""Project utility functions and classes"""

import os
import sys
import numpy as np
import pandas as pd

__all__ = ['get_trial', 'DataFile', 'SubjectData']
### General functions

def get_trial(df, block, trial_number, block_col='BlockNumber', 
              trial_col='TrialNumber'):
    """Returns specified trial from specified block. Assumes the block and trial
     columns are consistent across experiments"""
    return df.groupby([block_col, trial_col]).get_group((block, trial_number))

class DataFile:
    def __init__(self, fn, data_start, sep='\t'):
        """ Handles single-trial data from a single .dat file. Note that the
        headers attribute is a Series containing all strings."""

        header_rows = data_start - 4 
        # drop excess column almost all filled with NANs
        headers = (pd.read_table(fn, sep='\t', nrows=header_rows))
        # remove any duplicate indices, important for SubjectData.select_trials()
        headers = headers.drop(headers.columns[1], axis=1).squeeze()
        self.headers = headers[~headers.index.duplicated(keep='first')]
        self.block = self.headers['BlockNumber']
        self.trial = self.headers['TrialNumber']
        self.frequency = self.headers['DataRecordFrequency']
        
        self.data = pd.read_csv(fn, sep=sep, skiprows=data_start - 2)
        
class SubjectData:
    def __init__(self, path, subject, data_start, sep='\t'):
        """ Class to represent all data belonging to a single subject """
        self.id = subject
        self.data_path = os.path.join(path, subject)
        self.files = [i for i in os.listdir(self.data_path) if i.endswith('.dat')]
        # read in list of data file objects
        self.data_file_list = [DataFile(os.path.join(self.data_path, i), data_start) 
                               for i in self.files]
        
    def select_trials(self, header, value):
        """Identify trials based on a value in the file headers and return list
        of objects of only those trials. Important that value is a string."""
        return [i for i in self.data_file_list if i.headers[header] == value]

    def combine_all(self, reset_index=True):
        """Concatenates all loaded data into a single DataFrame"""
        self.data = pd.concat([i.data for i in self.data_file_list], 
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
        return get_trial(self.data, block, trial_number, block_col='BlockNumber', 
                         trial_col='TrialNumber')

    def get_header(self, name, dtype=None, index=None):
        """Get header value from every trial"""
        
        if index is None:
            data = self.data_file_list
        else:
            data = self.data_file_list[index]
        
        list_ = []
        for i in data:
            list_.append(i.headers[name])

        if dtype is not None:
            list_ = list(map(dtype, list_))
            
        return list_







