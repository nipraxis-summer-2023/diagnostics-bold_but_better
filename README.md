# Diagnostics Project by *Bold But Better Group*

## Overview

Welcome to the Diagnostics project! This repository contains Python scripts and modules designed for data validation and outlier detection in 4D images. Scripts are found in the `scripts` directory, and library code (Python modules) reside in the `findoutlie` directory. Scroll down for instructions on how to get your hands on the data and make it dance to your tunes.

## Requirements

- Python 3.x
- Internet access to download the data

## Get the Data

First, let's get the data like we get our morning newspaper, fresh and quick!

Change to the `data` directory:

```bash
cd data
```

Download and extract the data using:

```bash
curl -L https://figshare.com/ndownloader/files/34951602 -o group_data.tar
tar xvf group_data.tar
```

And don't forget to navigate back to the root of the repository:

```bash
cd ..
```

## Check the Data

Run this command like you're checking your tea for the right colour:

```bash
python3 scripts/validate_data.py <path_to_data>
```

### Example

```bash
python3 scripts/validate_data.py data
```

## Find Outliers

Let's catch those outliers, shall we? Like hunting for Waldo but in 4D.

```bash
python3 scripts/find_outliers.py <path_to_data>
```

### Example

```bash
python3 scripts/find_outliers.py data
```

You should see an output like this:

```
<filename>, <outlier_index>, <outlier_index>, ...
...
```

### What is going on?

The script tries 3 different outlier detection methods and uses mean root sum of squares (MRSS) as the criteria for "best" method. The method that gives the biggest reduction on MRSS is selected and indices of outliers per image file returned, based on that method. To see what is going on under the hood, turn on the verbose parameter:

```bash
python3 scripts/find_outliers.py data -v
```
or
```bash
python3 scripts/find_outliers.py data --verbose
```

You should see a 3 by 2 subplots of t, p and multiple comparison (p_adjusted) values of a brain slice, before and after applying the outlier detection method. Which slice you see and which multiple comparison method is applied can be configured by using the glm function directly from the findoutlie/outfind.py module.
You should also see logs from the "best" outlier detection method selection process.


## Contributing

Contributions are like clotted cream on scones, always welcome!

## License

This project is as open as the British skies, but check with @matthew-brett first
