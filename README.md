# Diagnostics project

Scripts go in the `scripts` directory.

Library code (Python modules) goes in the `findoutlie` directory.

You should put the code in this `findoutlie` directory on your Python PATH.

This README file has instructions on how to get, validate and process the data.

## Get the data

```
cd data
curl -L https://figshare.com/ndownloader/files/34951602 -o group_data.tar
tar xvf group_data.tar
```

Add the hash_list file to Git:

```
git add data/group-*/hash_list.txt
git commit -m "Add hash list file"
```

Change directory back to root of repository

```
cd ..
```

## Check the data

```
python3 scripts/validate_data.py data
```

## Find outliers

```
python3 scripts/find_outliers.py data
```

This should print output to the terminal of form:

```
<filename>, <outlier_index>, <outlier_index>, ...
<filename>, <outlier_index>, <outlier_index>, ...
```

Where `<filename>` is the name of the image that has outlier scans, and
`<outlier_index>` is an index to the volume in the 4D image that you have
identified as an outlier.  0 refers to the first volume.  For example (these
outlier IDs are completely random, for illustration):

```
data/sub-01/func/sub-01_task-taskzero_run-01_bold.nii.gz, 3, 21, 22, 104
data/sub-01/func/sub-01_task-taskzero_run-02_bold.nii.gz, 11, 33, 91
data/sub-03/func/sub-03_task-taskzero_run-02_bold.nii.gz, 101, 102, 132
data/sub-08/func/sub-08_task-taskzero_run-01_bold.nii.gz, 0, 1, 2, 166, 167
data/sub-09/func/sub-08_task-taskzero_run-01_bold.nii.gz, 3
```
