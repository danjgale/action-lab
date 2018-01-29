"""Module that contains tools to setup protocol files used in fMRI first-level
modelling and time-series indexing.
"""

import pandas as pd
import numpy as np


def _write_event_file(df, event, event_column_names, run, file_pattern,
                      save_path, split=False):

    all_events = df.groupby(event).get_group(event)

    if split:
        for i, j in enumerate(all_events.index.values):
            all_events.loc[i].to_csv(
                os.path.join(save_path, '{}_run{}_{}.txt'.format(event, run, i), sep='\t')
            )
    else:
        all_events.to_csv(
            os.path.join(save_path, '{}_run{}.txt'.format(event, run), sep='\t')
        )

def _compute_durations(start, stop):
    """Get duration of event based on BVs. Add 1 because duration is inclusive"""
    return (stop - start) + 1

def get_event_durations(df, start, stop):
    df['Duration'] = (
        df.apply(lambda x: _compute_durations(x[start], x[stop]), axis=1)
    )
    return df

def expand_events(df, duration_col):

    return df.reindex(
        np.repeat(df.index.values, df[duration_col].astype(int)), method='ffill'
    ).reset_index(drop=True)


# class ProtocolBase:

#     def __init__(self, onset_col, as_seconds=False):

#         self.onset_col = onset_col


#         self.__as_seconds = as_seconds

#     def insert_amplitudes(self, value=1):

#         for i in self.protocols:
#             i['amp'] = value

#         return self

#     def get_event_durations():



#     def convert_to_secs(self, columns=):

#         self.__as_seconds = True

#         for i in self.protocols:
#             i[self.onset_col] = (i[self.onset_col] * 2) - 2
#             i[self.duration_col] = self.duration_col * 2

#         return self


#     def combine(self):

#         concat_list = []
#         total = 0
#         for i in protocol_list:
#             i[self.onset_col] = i[self.onset_col] + total_seconds
#             total += (i.iloc[-1][self.onset_col] + i.iloc[-1][self.duration_col])
#             concat_list.append(i)

#         return pd.concat(concat_list, axis=0)

#     def make_event_files(self, event, individual_events=False):

#         for i in self.protocols:
#             _write_event_file(i, event, [self.onset_col, self.duration_col, 'amp'],
#                               run, file_pattern, save_path, split=False)






class ProtocolManager:

    def __init__(self, protocols, onset_col='onset', duration_col='duration',
                 event_col='event', sort=False, as_seconds=False):

        self.protocols = [pd.read_csv(i) for i in protocols]
        self.onset_col = onset_col
        self.duration_col = duration_col
        self.event_col = event_col

        if sort:
            self.protocols = sorted(protocols)

        self.__as_seconds = as_seconds

    def insert_amplitudes(self, value=1):

        for i in self.protocols:
            i['amp'] = value

        return self

    def convert_to_secs(self):

        self.__as_seconds = True

        for i in self.protocols:
            i[self.onset_col] = (i[self.onset_col] * 2) - 2
            i[self.duration_col] = self.duration_col * 2

        return self


    def combine(self):

        concat_list = []
        total = 0
        for i in protocol_list:
            i[self.onset_col] = i[self.onset_col] + total_seconds
            total += (i.iloc[-1][self.onset_col] + i.iloc[-1][self.duration_col])
            concat_list.append(i)

        return pd.concat(concat_list, axis=0)

    def make_event_files(self, event, individual_events=False):

        for i in self.protocols:
            _write_event_file(i, event, [self.onset_col, self.duration_col, 'amp'],
                              run, file_pattern, save_path, split=False)




# class BVProtocol:

#     def __init__(self, start_col, end_col, condition_col):
