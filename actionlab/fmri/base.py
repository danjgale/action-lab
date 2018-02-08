"""Base class for processing subclasses"""


import os
from nipype.interfaces.io import DataSink
from nipype.pipeline.engine import Node

class BaseProcessor(object):

    """ Base class to set up input/outputs for subclasses.

    BaseProcessor sets up a sub_id, input_data (either a list of files, or a
    directory containing NiFTI files), and an output_path.

    Specifying the output_path specifies a top-level directory that contains
    Nipype's a) workflow output directory and b) the workflow's working
    directory. The actual directory containing the processed files is a
    subdirectory under the Nipype workflow output directory (a). As an example,
    if you specify output_path = 'output_data', subject 'AA' data will be
    stored in `output_data/output/AA/`. Furthermore, the output data will
    have it's own directory structure according to the DataSink used in the
    Nipype workflow. Under `output_data/output/AA/`, there could be a
    motion_corrected/ folder, for instance.

    Using BaseProcessor ensures consistency among main processing classes:
    Preprocessor, Normalizer, Filter, and GLM.

    Attributes
    ----------
    sub_id : str
        Subject's id
    input_data : str or list of str
        Either a directory containing files for input (str) or a list of
        filenames. Importantly, for either case, absolute paths must be used.
    output_path : str
        Output directory path. Does not need to exist, but must be an absolute
        path.
    zipped: bool
        Specify if NiFTI input files are compressed. Default is True.
    input_file_endswith: str, optional
        If input_data is a directory and only a subset of NiFTI files are of
        interest, this specifies a pattern that marks those files.
    datasink: obj
        Nipype Datasink object.

    """

    def __init__(self, sub_id, input_data, output_path, zipped=True,
                 input_file_endswith=None, datasink_parameterization=False,
                 sort_input_files=True):

        self.sub_id = sub_id
        self.input_data = input_data
        self.zipped = zipped

        if input_file_endswith is None:
            # infer based on zipped
            input_file_endswith = '.nii.gz' if self.zipped else '.nii'


        # set up input files, which are a list of file names to be used by
        # the SelectFiles interface
        if isinstance(self.input_data, str):
            self._input_files = [os.path.join(input_data, i)
                               for i in os.listdir(input_data)
                               if i.endswith(input_file_endswith)]
        elif isinstance(self.input_data, list):
            self._input_files = self.input_data

        if sort_input_files:
            # no guarantees this works for all cases...
            self._input_files = sorted(self._input_files)

        self.output_path = output_path
        self._working_dir =  os.path.join(self.output_path, 'working')
        self._datasink_dir = os.path.join(self.output_path, 'output')
        self._sub_output_dir = os.path.join(
            self._datasink_dir,
            self.sub_id
        )

        if not os.path.exists(self._sub_output_dir):
            os.makedirs(self._sub_output_dir)

        # Set up datasink
        self.datasink = Node(
            DataSink(
                base_directory=self._datasink_dir,
                container=self._sub_output_dir,
                substitutions=[('_subject_id_', ''), ('sub_id_', '')],
                parameterization=datasink_parameterization
            ),
            name='datasink'
        )
