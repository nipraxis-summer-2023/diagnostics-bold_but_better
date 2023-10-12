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

The script tries 3 different outlier detection methods and uses mean root sum of squares (MRSS) as the criteria for "best" method. The method that gives the biggest reduction on MRSS is selected and indices of outliers per image file returned, based on that method.

The script writes out a file called educated_guess.txt which makes an educated guess about the nature of outliers per image file, based on the outliers found with the three outlier dection methods (z-score detector, interquartile range detector and DIVARs).

```bash
educated_guess.txt
```
To see what is going on under the hood whilst the script is running, turn on the verbose parameter, with -v or --verbose:

```bash
python3 scripts/find_outliers.py data --verbose
```

### show images
Turn on images with -s or --show
```bash
python3 scripts/find_outliers.py data --show
```
You should see a 3 by 2 subplots of t statistic, p value and p_adj (multiple comparison adjusted p value) values of a brain slice, before and after applying each outlier detection method.
The selected slice and multiple comparison method used, can be configured by using the glm function directly from the findoutlie/outfind.py module.

### Conservative approach
For more specificity, outliers from all methods can be joined. To do so, set the -c or --conservative flag
```bash
python3 scripts/find_outliers.py data --conservative
```

### Example
Setting all flags: combine outliers from all methods, print logs and display images
```bash
python3 scripts/find_outliers.py data -c -v -s
```

## Contributing

Contributions are like clotted cream on scones, always welcome!

## License

This project is as open as the British skies, but check with @matthew-brett first
