
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


class GLM(BaseProcessor):

    def __init__(self, sub_id, input_data, output_path, zipped=True,
                 input_file_endswith=None, sort_input_files=True):


        BaseProcessor.__init__(self, sub_id, input_data, output_path, zipped,
                               input_file_endswith,
                               sort_input_files=sort_input_files,
                               datasink_parameterization=True)


    def build(self, protocol_file, contrasts, realign_params, output_path=None,
              workflow_name='glm'):

        # note that this concatenates runs, so for a typical experiment in which
        # all runs have the same conditions but in different orders, you input
        # the concatenated protocol file for ALL runs, and include ALL runs in
        # the order that they appear in the protocol file

        # build sets workflow for individual runs (with different protocols)
        self.protocol_file = protocol_file
        self.contrasts = contrasts
        self.realign_params = realign_params # must be in same order as runs


        if output_path is not None:
            # update data sink directory for a specific build
            self._datasink_dir = os.path.abspath(
                os.path.join(output_path, 'output')
            )
            self._working_dir = os.path.abspath(
                os.path.join(output_path, 'working')
            )

        # ----------
        # Data Input
        # ----------

        self.infosource = Node(
            IdentityInterface(
                fields=['funct', 'realign_params']
            ),
            name='infosource'
        )
        self.infosource.funct = self._input_files
        self.infosource.realign_params = self.realign_params

        # -----------
        # Model Nodes
        # -----------

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
        self.workflow.base_dir = self._working_dir

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
            (self.infosource, self.model_spec, [
                ('funct', 'functional_runs'),
                ('realign_parameters', 'realignment_parameters')
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
