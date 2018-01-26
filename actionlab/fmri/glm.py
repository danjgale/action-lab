
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


def bunch_protocols(protocol, nruns, condition_col):
    """Create bunch object (event, start, duration, amplitudes) for
    SpecifyModel() using existing event files
    """

    if isinstance(protocol, str):
        df = pd.read_csv(protocol)
    else:
        df = protocol

    grouped = df.groupby(condition_col)

    # each bunch corresponds to ONE RUN, not one condition.
    bunches = []
    for i in range(nruns):

        names = []
        onsets = []
        durations = []
        amplitudes = []
        for name, g in grouped:

            names.append(name)
            onsets.append(g['start'].tolist())
            durations.append(g['duration_s'].tolist())
            amplitudes.append(
                np.ones(len(g['duration_s'].tolist()), dtype=np.int8).tolist()
            )

        # each element corresponds to a run
        bunches.append(
            Bunch(conditions=names, onsets=onsets, durations=durations,
                  amplitudes=amplitudes)
        )

    return bunches


class GLM:

    def __init__(self, sub_id, data_dir, output_dir):

        self.sub_id = sub_id
        self.data_dir = data_dir
        self.__working_dir = os.path.abspath(
            os.path.join(self.output_dir, 'working')
        )
        self.__datasink_dir = os.path.abspath(
            os.path.join(self.output_dir, 'output')
        )

        


    def build(self, protocol_file, contrasts, runs, run_template,
              realign_template, parameterize_output=False,
              output_dir=None):

        # note that this concatenates runs, so for a typical experiment in which
        # all runs have the same conditions but in different orders, you input
        # the concatenated protocol file for ALL runs, and include ALL runs in
        # the order that they appear in the protocol file

        # build sets workflow for individual runs (with different protocols)
        self.protocol_file = protocol_file
        self.contrasts = contrasts
        self.run_template = run_template
        self.realign_params = realign_params
        self.realign_template = realign_template
        self.runs = runs
        self.parameterize_output = parameterize_output

        if output_dir is not None:
            # update data sink directory for a specific build
            self.__datasink_dir = os.path.abspath(
                os.path.join(output_dir, 'output')
            )

            self.__working_dir = os.path.abspath(
                os.path.join(output_dir, 'working')
            )


        self.__input_runs = [run_template.format(run=i) for i in self.runs]
        self.__realign = [run_template.format(run=i) for i in self.runs]


        # ----------
        # Data Input
        # ----------

        self.infosource = Node(
            IdentityInterface(
                fields=['selected_runs', 'realign_params']
            ),
            name='infosource'
        )
        self.infosource.selected_runs = self.__input_runs
        self.infosource.realign_params = self.__realign

        self.select_files = Node(
            SelectFiles(
                {'runs': '{selected_runs}',
                 'realignment_parameters': '{realign_params}'},
                name='select_files'
            )
        )

        # -----------
        # Data Output
        # -----------

        # setup subject's data folder
        self.__sub_output_dir = os.path.join(
            self.__datasink_dir,
            self.sub_id
        )

        if not os.path.exists(self.__sub_output_dir):
            os.makedirs(self.__sub_output_dir)

        self.datasink = Node(
            DataSink(
                base_dir=self.__datasink_dir,
                container=self.__sub_output_dir,
                substitutions=[('_subject_id_', ''), ('sub_id_', '')],
                parameterization=self.parameterize_output
            ),
            name='datasink'
        )

        # initialize model nodes
        self.sub_info = bunch_protocols(protocol_file, len(run_num))


        self.model_spec = Node(
            SpecifySPMModel(
                subject_info = self.sub_info,
                high_pass_filter_cutoff = self.high_pass_filter_cutoff,
                time_repetition=self.TR,
                input_units=self.input_units,
                concatenate_runs=True
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


        self.workflow=Workflow(name = 'glm')
        self.workflow.base_dir = self.__working_dir

        # intra-modelling flow
        self.workflow.connect([
            (self.model_spec, self.design, [('session_info', 'session_info')]),
            (self.design, self.model_est, [('spm_mat_file', 'spm_mat_file')]),
            (self.model_est, self.con_est, [
                ('spm_mat_file', 'spm_mat_file'),
                ('beta_images', 'beta_images'),
                ('residual_image', 'residual_image')
            ])
        ])

        # input and output flow
        self.workflow.connect([
            (self.infosource, self.select_files, [
                ('selected_runs', 'selected_runs'),
                ('realign_params', 'realign_params')
            ]),
            (self.select_files, self.model_spec, [
                ('runs', 'functional_runs'),
                ('realignment_parameters', 'realignment_parameters')
            ]),
            (self.design, self.datasink, [('spm_mat_file', 'model.pre-estimate')]),
            (self.model_est, self.datasink, [
                ('spm_mat_file', 'model.@spm'),
                ('beta_images', 'model.@beta'),
                ('residual_image', 'model.@res'),
                ('RPVimage', 'model.@rpv')
            ]),
            (self.con_est, self.datasink, [
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
