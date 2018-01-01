"""Normalization workflow using FSL's FLIRT and FNIRT"""

import os
import sys
import numpy as np
import nipype
from nipype import logging
from nipype.pipeline.engine import Workflow, Node, MapNode
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces import fsl, spm
from nipype.interfaces.utility import IdentityInterface, Function
from utils import set_datasink

class Normalizer:

    def __init__(self, sub_id, data_dir, working_dir, datasink_dir,
                 anatomical='*CNS_SAG_MPRAGE_*.nii.gz', functionals=None,
                 mni_template=True):

        self.sub_id = sub_id
        self.data_dir = os.path.abspath(data_dir)
        self.datasink_dir = os.path.abspath(datasink_dir)
        self.working_dir = os.path.abspath(working_dir)
        self.functionals = functionals
        self.anatomical = anatomical

    def build(self):
        pass

    def run(self):
        pass
