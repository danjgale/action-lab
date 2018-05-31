
import os
import sys
import argparse
import itertools
import numpy as np
import pandas as pd
import nipype
from nipype import logging
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces import fsl, spm, freesurfer
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces.base import Bunch
from nipype.algorithms.modelgen import SpecifyModel, SpecifySPMModel
from nipype.interfaces.utility import IdentityInterface, Function

from .base import BaseProcessor


def _check_design_input(design_list):
    if all(isinstance(i, str) for i in design_list):
        design_tables = [pd.read_csv(i, sep='\t') for i in design_list]
    elif all(isinstance(i, pd.DataFrame) for i in design_list):
        design_tables = design_list
    else:
        raise ValueError('Ensure that designs are a list of str or list of pd.DataFrame')

    return design_tables


def stack_designs(design_list, onset_col='onset', duration_col='duration'):
    """Vertically concatenate designs for concatenated runs in model fitting.

    Expected that the design files are in the same order as the functional runs.
    Onsets from each run are adjusted to reflect new timing of concatenation.
    """

    design_tables = _check_design_input(design_list)

    concat_list = []
    total_seconds = 0
    for i in design_tables:
        i[onset_col] = i[onset_col] + total_seconds
        total_seconds += i.iloc[-1][onset_col] + i.iloc[-1][duration_col]
        concat_list.append(i)
    return pd.concat(concat_list, axis=0)


def bunch_single_design(df, condition_col, onset_col='onset', duration_col='duration'):
    """Create bunch object from a design dataframe"""
    grouped = df.groupby(condition_col)
    # each condition has a list of onsets, durations, and amplitudes associated
    # with it
    names = []
    onsets = []
    durations = []
    for name, g in grouped:
        names.append(name)
        onsets.append(g[onset_col].tolist())
        durations.append(g[duration_col].tolist())

    return Bunch(conditions=names, onsets=onsets, durations=durations)


def _bunch_single_design_wrapper(design, condition_col, onset_col='onset', duration_col='duration'):
    """ Wrapper to work within a Nipype pipeline"""
    import pandas as pd
    from nipype.interfaces.base import Bunch
    df = pd.read_csv(design, sep='\t')
    return bunch_single_design(df, condition_col, onset_col, duration_col)



def bunch_designs(design_list, condition_col, onset_col='onset',
                  duration_col='duration'):
    """Create bunch object (event, start, duration, amplitudes) for
    SpecifyModel() using existing event files
    """
    design_tables = _check_design_input(design_list)
    bunch_list = []
    for i in design_tables:
        bunch_list.append(bunch_single_design(i, condition_col, onset_col, duration_col))

    return bunch_list


class GLM(BaseProcessor):

    def __init__(self, sub_id, input_data, output_path, zipped=True,
                 input_file_endswith=None, sort_input_files=True):


        BaseProcessor.__init__(self, sub_id, input_data, output_path, zipped,
                               input_file_endswith,
                               sort_input_files=sort_input_files,
                               datasink_parameterization=True)


    def build(self, design, contrasts, realign_params, output_path=None,
              high_pass_filter_cutoff=100, TR=2.0, input_units='secs',
              workflow_name='glm', design_onset_col='onset',
              design_duration_col='duration', design_condition_col='condition'):

        # note that this concatenates runs, so for a typical experiment in which
        # all runs have the same conditions but in different orders, you input
        # the concatenated protocol file for ALL runs, and include ALL runs in
        # the order that they appear in the protocol file
        # ^^^NO LONGER TRUE!!!!

        # build sets workflow for individual runs (with different protocols)
        self.design = design
        self.workflow_name = workflow_name
        self.contrasts = contrasts
        self.realign_params = realign_params # must be in same order as runs
        self.high_pass_filter_cutoff = high_pass_filter_cutoff
        self.TR = TR
        self.input_units = input_units

        if output_path is not None:
            # update data sink directory for a specific build
            self._datasink_dir = os.path.abspath(
                os.path.join(output_path, 'output')
            )
            self._working_dir = os.path.abspath(
                os.path.join(output_path, 'working')
            )

        # -----------
        # Model Nodes
        # -----------

        self.sub_info = bunch_designs(design_files, design_condition_col,
                                      onset_col=design_onset_col,
                                      duration_col=design_duration_col)

        self.model_spec = Node(
            SpecifySPMModel(
                subject_info = self.sub_info,
                high_pass_filter_cutoff = self.high_pass_filter_cutoff,
                time_repetition=self.TR,
                input_units=self.input_units,
                output_units=self.input_units,
                functional_runs = self._input_files,
                realignment_parameters = self.realign_params,
                concatenate_runs=False
            ),
            name='model_spec'
        )

        self.design = Node(
            spm.Level1Design(
                bases={'hrf': {'derivs': [1, 0]}},
                timing_units=self.input_units,
                interscan_interval=self.TR
            ),
            name='design'
        )

        self.estimate_model = Node(
            spm.EstimateModel(
                estimation_method={'Classical': 1}
            ),
            name='model_est'
        )

        self.estimate_contrast = Node(
            spm.EstimateContrast(
                contrasts=self.contrasts
            ),
            name='con_est'
        )

        self.workflow=Workflow(name = self.workflow_name)
        self.workflow.base_dir = self._working_dir

        # -----------
        # Work Flow
        # -----------
        # intra-modelling flow
        self.workflow.connect([
            (self.model_spec, self.design, [('session_info', 'session_info')]),
            (self.design, self.estimate_model, [('spm_mat_file', 'spm_mat_file')]),
            (self.estimate_model, self.estimate_contrast, [
                ('spm_mat_file', 'spm_mat_file'),
                ('beta_images', 'beta_images'),
                ('residual_image', 'residual_image')
            ])
        ])
        # input and output flow
        self.workflow.connect([
            #(self.infosource, self.model_spec, [
            #    ('funct', 'functional_runs'),
            #    ('realign_params', 'realignment_parameters')
            #]),
            (self.design, self.datasink, [('spm_mat_file', 'model.pre-estimate')]),
            (self.estimate_model, self.datasink, [
                ('spm_mat_file', 'model.@spm'),
                ('beta_images', 'model.@beta'),
                ('residual_image', 'model.@res'),
                ('RPVimage', 'model.@rpv')
            ]),
            (self.estimate_contrast, self.datasink, [
                ('con_images', 'contrasts'),
                ('spmT_images', 'contrasts.@T')
            ])
        ])

        return self

    def run(self, parallel=True, print_header=True, n_procs=8):

        if print_header:
            print('=' * 30 + 'SUBJECT {}'.format(self.sub_id) + '=' * 30)

        if parallel:
            self.workflow.run('MultiProc', plugin_args = {'n_procs': n_procs})
        else:
            self.workflow.run()

        return self


class GroupGLM:
    def __init__(self, output_path):

        # does not inherit BaseProcessor because this is a group-level operation
        # rather than a single subject operation.


        self.output_path = output_path
        self._working_dir =  os.path.join(self.output_path, 'working')
        self._datasink_dir = os.path.join(self.output_path, 'output')


    def build(self, input_data, name=None, thresholding=True, cluster_thresholding=True,
              primary_threshold=.005, extent_threshold=.05, minimum_voxels=0,
              primary_threshold_fwe_correct=False):

        self.input_data = input_data

        # place several group-level analyses (i.e. contrasts) in same output
        # directory
        self.name = name
        # Set up datasink
        self.datasink = Node(
            DataSink(
                base_directory=self._datasink_dir,
                container=os.path.join(self._datasink_dir, self.name),
                parameterization=True
            ),
            name='datasink'
        )

        # largely taken from
        # https://miykael.github.io/nipype_tutorial/notebooks/example_2ndlevel.html

        # OneSampleTTestDesign - creates one sample T-Test Design
        self.ttest = Node(
            spm.OneSampleTTestDesign(
                in_files=self.input_data,
            ),
            name="ttest"
        )

        # EstimateModel - estimates the model
        self.level2estimate = Node(spm.EstimateModel(estimation_method={'Classical': 1}),
                            name="level2estimate")

        # EstimateContrast - estimates group contrast
        self.level2conestimate = Node(spm.EstimateContrast(group_contrast=True),
                                name="level2conestimate")
        cont1 = ['Group', 'T', ['mean'], [1]]
        self.level2conestimate.inputs.contrasts = [cont1]

        # -----------
        # Work Flow
        # -----------
        self.workflow=Workflow(name=self.name)
        self.workflow.base_dir = self._working_dir

        self.workflow.connect([
            (self.ttest, self.level2estimate, [('spm_mat_file', 'spm_mat_file')]),
            (self.level2estimate, self.level2conestimate, [
                ('spm_mat_file', 'spm_mat_file'),
                ('beta_images', 'beta_images'),
                ('residual_image', 'residual_image')
            ])
        ])

        # output
        self.workflow.connect([
            (self.level2estimate, self.datasink, [
                ('spm_mat_file', 'model'),
                ('beta_images', 'model.@beta'),
                ('residual_images', 'model.@resid')
            ]),
            (self.level2conestimate, self.datasink, [
                ('spm_mat_file', 'contrast'),
                ('spmT_images', 'contrast.@T'),
                ('con_images', 'contrast.@con')
            ])
        ])

        if thresholding:

            # NOTE: spm calls extent threshold the number of voxels, but extent
            # threshold, as per literature, refers to cluster p value threshold
            # (which is what we'll stay true to here)

            if cluster_thresholding:
                # threshold map and clusters
                self.level2thresh = Node(
                        spm.Threshold(
                            contrast_index=1,
                            use_topo_fdr=True,
                            use_fwe_correction=primary_threshold_fwe_correct,
                            extent_threshold=minimum_voxels, # this specifies n VOXELS
                            height_threshold=primary_threshold,
                            height_threshold_type='p-value',
                            extent_fdr_p_threshold=extent_threshold # this specifies cluser p value
                        ),
                        name="level2thresh"
                )
            else:
                # threshold map but not clusters
                self.level2thresh = Node(
                        spm.Threshold(
                            contrast_index=1,
                            use_topo_fdr=False,
                            extent_threshold=minimum_voxels,
                        ),
                        name="level2thresh"
                )

            self.workflow.connect([
                (self.level2conestimate, self.level2thresh, [
                    ('spm_mat_file', 'spm_mat_file'),
                    ('spmT_images', 'stat_image')
                ]),
                (self.level2thresh, self.datasink, [
                    ('thresholded_map', 'contrast.@threshold')
                ])
            ])


    def run(self, parallel=True, n_procs=8):

        if parallel:
            self.workflow.run('MultiProc', plugin_args = {'n_procs': n_procs})
        else:
            self.workflow.run()

        return self


def single_trial_design(design, condition_col='condition', lm_type='lss'):

    if isinstance(design, str):
        design = pd.read_csv(design, sep='\t')

    design_list = []
    for ix, row in design.iterrows():

        df = design.copy()

        # label condition based on trial vs rest
        df.drop(ix, inplace=True)
        df[condition_col] = 'other'

        design_list.append(df.append(row))

    return design_list

def _unpack_run_map(x):
    list_ = []
    for k, v in x.items():
        for i in v:
            list_.append((k, i))
    return list_


def _save_designs(path, design_map):
    dict_ = []
    for i, (k, v) in enumerate(design_map):
        list_ []
        for j in v:
            fn = os.path.join(path, 'run{}_trial{}.csv')
            j.to_csv()
            list_.append(fn)
        dict_[k] = list_

    return dict_


class LSS(BaseProcessor):


    def __init__(self, sub_id, input_data, output_path, zipped=True,
                 input_file_endswith=None, sort_input_files=True):

        BaseProcessor.__init__(self, sub_id, input_data, output_path, zipped,
                               input_file_endswith,
                               sort_input_files=sort_input_files,
                               datasink_parameterization=True)


    def build(self, design, realign_params=None, output_path=None,
              high_pass_filter_cutoff=100, TR=2.0, input_units='secs',
              workflow_name='glm', design_onset_col='onset',
              design_duration_col='duration', design_condition_col='condition'):

        # build sets workflow for individual runs (with different protocols)
        self.design = design
        self.workflow_name = workflow_name
        self.realign_params = realign_params # must be in same order as runs
        self.high_pass_filter_cutoff = high_pass_filter_cutoff
        self.TR = TR
        self.input_units = input_units

        self.design_onset_col = design_onset_col
        self.design_duration_col = design_duration_col
        self.design_condition_col = design_condition_col

        # create list with each element as a list containing single-trial design matrices for
        # a given run

        design_matrices = [single_trial_design(i, design_condition_col) for i in self.design]

        # create mapping between run and the design matrices for that run
        self.run_design_map = dict(zip(self._input_files, design_matrices))

        self.run_design_pairs = _unpack_run_map(self.run_design_map)

        # create mapping between run and realign params
        if self.realign_params is not None:
            self.run_realign_map = dict(zip(self._input_files, self.realign_params))
        else:
            self.run_realign_map = None

        # create/save design files
        design_dir = os.path.join(self._datasink_dir, 'design')
        if not os.path.exist(design_dir):
            os.makedir(design_dir)
        self._design_file_map = _save_designs(design_dir, self.design_map)


    def run(self, parallel=True, print_header=True, n_procs=8):

        for i, (run, design) in enumerate(self._design_file_map.items()):

            self.infosource = Node(
                IdentityInterface(fields=['design']),
                name='infosources'
            )
            self.infosource.iterables = [('design', design)]

            self.subject_info = Node(
                Function(
                    input_names=['design', 'condition_col', 'onset_col', 'duration_col'],
                    output_names=['bunch'],
                    function=_bunch_single_design_wrapper
                ),
                name='subject_info'
            )
            self.subject_info.inputs.condition_col = self.design_condition_col
            self.subject_info.inputs.onset_col = self.design_onset_col
            self.subject_info.inputs.duration_col = self.design_duration_col

            if self.run_realign_map is None:
                # exclude realignment parameters
                self.model_spec = Node(
                    SpecifySPMModel(
                        functional_runs=run,
                        high_pass_filter_cutoff = self.high_pass_filter_cutoff,
                        time_repetition=self.TR,
                        input_units=self.input_units,
                        output_units=self.input_units,
                        concatenate_runs=False
                    ),
                    name='model_spec'
                )
            else:
                # include realignment parameters
                self.model_spec = Node(
                    SpecifySPMModel(
                        high_pass_filter_cutoff = self.high_pass_filter_cutoff,
                        time_repetition=self.TR,
                        input_units=self.input_units,
                        output_units=self.input_units,
                        realignment_parameters = self.realign_params[run],
                        concatenate_runs=False
                    ),
                    name='model_spec'
                )

            # ------------
            # GLM Workflow
            # ------------

            self.design = Node(
                spm.Level1Design(
                    bases={'hrf': {'derivs': [1, 0]}},
                    timing_units=self.input_units,
                    interscan_interval=self.TR
                ),
                name='design'
            )

            self.estimate_model = Node(
                spm.EstimateModel(
                    estimation_method={'Classical': 1}
                ),
                name='model_est'
            )

            # temporary functions. TODO: come up with better way to place these so that
            # they're hidden but specific to class
            # def _get_run(x):
            #     return x[0]

            def _get_design(x):
                grouped = x.groupby(self.design_condition_col)
                # each condition has a list of onsets, durations, and amplitudes associated
                # with it
                names = []
                onsets = []
                durations = []
                for name, g in grouped:
                    names.append(name)
                    onsets.append(g[self.design_onset_col].tolist())
                    durations.append(g[self.design_duration_col].tolist())

                return Bunch(conditions=names, onsets=onsets, durations=durations)


                #bunch = bunch_single_design(x[1], self.design_onset_col,
                #    self.design_duration_col, self.design_condition_col)
                #return bunch


            self.workflow=Workflow(name=self.workflow_name)
            self.workflow.base_dir = self._working_dir
            self.workflow.connect([
                (self.infosource, self.subject_info, [('design', 'design')]),
                (self.subject_info, self.model_spec, [('info', 'subject_info')]),
                (self.model_spec, self.design, [('session_info', 'session_info')]),
                (self.design, self.estimate_model, [('spm_mat_file', 'spm_mat_file')])
            ])

            self.workflow.connect([
                (self.design, self.datasink, [('spm_mat_file', 'run{}.pre-estimate'.format(i))]),
                (self.estimate_model, self.datasink, [
                    ('spm_mat_file', 'run{}.@spm'.format(i)),
                    ('beta_images', 'run{}.@beta'.format(i)),
                    ('residual_image', 'run{}.@res'.format(i)),
                    ('RPVimage', 'run{}.@rpv'.format(i))
                ])
            ])

            if parallel:
                self.workflow.run('MultiProc', plugin_args = {'n_procs': n_procs})
            else:
                self.workflow.run()
        return self
