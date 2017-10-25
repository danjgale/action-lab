# action-lab
General lab library for everyday tasks.

# Data Handling

Typically, all of a single subject's data is stored in a single directory, with each trial stored as an individual `.dat` file. `actionlab` contains two classes to handle this structure: `DataFile`, and `SubjectData`. `DataFile` is a single-file parser for trial-by-trial files. Here, you can access header information, along with trial data. `SubjectData` generates a list of `DataFile` objects for every trial of a single subject, from which you can access individual trials (`SubjectData.get_trial()` or `SubjectData.select_trials()`), etc.

**Example**

```
from actionlab.data import DataFile, SubjectData

data = DataFile('example.dat', datastart=30)

# get actual data of trial
print(data.data)

# show headers attached to trial
print(data.headers)

# whole subject
data_path = 'data/subject1'
subject1 = SubjectData(data_path)

# see data of an individual trial
trial11 = subject1.data_list[10].data

# create a large DataFrame from all trial data
subject1_df = subject1.combine_all()

# get some sort of header value (trial_type, etc)
conditions = subject1.get_header('Condition')
```


