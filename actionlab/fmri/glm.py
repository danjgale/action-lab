
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
from nipype.interfaces.utility import IdentityInterface

from .base import BaseProcessor


def stack_designs(design_files, onset_col='onset', duration_col='duration'):
    """Vertically concatenate designs for concatenated runs in model fitting.

    Expected that the design files are in the same order as the functional runs.
    Onsets from each run are adjusted to reflect new timing of concatenation.
    """

    design_list = [pd.read_csv(i, sep='\t') for i in design_files]

    concat_list = []
    total_seconds = 0
    for i in design_list:
        i[onset_col] = i[onset_col] + total_seconds
        total_seconds += i.iloc[-1][onset_col] + i.iloc[-1][duration_col]
        concat_list.append(i)
    return pd.concat(concat_list, axis=0)


def bunch_designs(design_files, condition_col, onset_col='onset',
                  duration_col='duration'):
    """Create bunch object (event, start, duration, amplitudes) for
    SpecifyModel() using existing event files
    """

    bunch_list = []
    for i in design_files:
        df = pd.read_csv(i, sep='\t')

        grouped = df.groupby(condition_col)

        # each condition has a list of onsets, durations, and amplitudes associated
        # with it
        names = []
        onsets = []
        durations = []
        amplitudes = []
        for name, g in grouped:
            names.append(name)
            onsets.append(g[onset_col].tolist())
            durations.append(g[duration_col].tolist())
            amplitudes.append(
                np.ones(len(g[onset_col].tolist()), dtype=np.int8).tolist()
            )

        bunch_list.append(Bunch(conditions=names, onsets=onsets, durations=durations))

    return bunch_list


class GLM(BaseProcessor):

    def __init__(self, sub_id, input_data, output_path, zipped=True,
                 input_file_endswith=None, sort_input_files=True):


        BaseProcessor.__init__(self, sub_id, input_data, output_path, zipped,
                               input_file_endswith,
                               sort_input_files=sort_input_files,
                               datasink_parameterization=True)


    def build(self, design_files, contrasts, realign_params, output_path=None,
              high_pass_filter_cutoff=100, TR=2.0, input_units='secs',
              workflow_name='glm', design_onset_col='onset',
              design_duration_col='duration', design_condition_col='condition'):

        # note that this concatenates runs, so for a typical experiment in which
        # all runs have the same conditions but in different orders, you input
        # the concatenated protocol file for ALL runs, and include ALL runs in
        # the order that they appear in the protocol file
        # ^^^NO LONGER TRUE!!!!

        # build sets workflow for individual runs (with different protocols)
        self.design_files = design_files
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


    def build(self, input_data, name=None):

        self.input_data = input_data

        # place several group-level analyses (i.e. contrasts) in same output
        # directory
        self.name = name
        # Set up datasink
        self.datasink = Node(
            DataSink(
                base_directory=self._datasink_dir,
                container=self.name
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
                ('spmT_images', 'contrast@.T'),
                ('con_images', 'contrast.@con')
            ])
        ])


    def run(self, parallel=True, print_header=True, n_procs=8):

        if print_header:
            print('=' * 30 + 'SUBJECT {}'.format(self.sub_id) + '=' * 30)

        if parallel:
            self.workflow.run('MultiProc', plugin_args = {'n_procs': n_procs})
        else:
            self.workflow.run()

        return self

