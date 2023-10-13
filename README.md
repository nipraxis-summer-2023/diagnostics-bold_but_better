# Diagnostics Project by *Bold But Better Group*

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Get the Data](#get-the-data)
- [Check the Data](#check-the-data)
- [Find Outliers](#find-outliers)
- [Conservative Approach](#conservative-approach)
- [What is going on?](#what-is-going-on)
- [Show images](#show-images)
- [Example with flags](#example-with-flags)
- [Contributing](#contributing)
- [License](#license)

## Overview

Welcome aboard the Diagnostics train! 🚂 This depot is stocked with Python scripts and modules all prepped to detect outliers in 4D images. Your journey begins with `scripts` nestled in the scripts directory and takes you through Python modules residing in the `findoutlie` directory. Ready to conduct your data symphony? Keep reading!

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
The below script will apply three different outlier detection methods on the data: Z-score, Interquartile range and DIVAR
The General Linear Method (GLM) is then applied with convolved hemodynamic response function as activation model. From the GLM model, Mean Root Sum of Squares (MRSS) is calculated, before and after removing outliers detected by each method. The method that shows the biggest reduction in MRSS is then selected as the method of choice.

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

### Conservative Approach
Desiring a more exhaustive outlier search? 🔍 You can combine the findings from all methods. However, this comprehensive net might flag more data points as outliers. Beware, this could include some data points that aren't genuine outliers, termed as "false positives". To cast this comprehensive net, set the -c or --conservative flag.
```bash
python3 scripts/find_outliers.py data --conservative
```

### What is going on?

The script tries 3 different outlier detection methods and uses Mean Root Sum of Squares (MRSS) as the criteria for "best" method. The method that gives the biggest reduction on MRSS is selected and indices of outliers per image file returned, based on that method.

The script writes out a file called `educated_guess.txt` which makes an educated guess about the nature of outliers per image file, based on the outliers found with the three outlier dection methods (z-score detector, interquartile range detector and DIVARs).

```bash
educated_guess.txt
```
To get more details and see what is going on under the hood whilst the script is running, you can turn on the verbose parameter, with -v or --verbose:

```bash
python3 scripts/find_outliers.py data --verbose
```

### Show images
Turn on images with -s or --show
```bash
python3 scripts/find_outliers.py data --show
```
Setting the `show` flag, displays 3x2 subplots of the t-statistic, p-value and p_adj (multiple comparison adjusted p value) values of a brain slice – before and after applying each outlier detection method.

The selected slice and multiple comparison method used, can be configured by using the glm function directly from the `findoutlie/outfind.py` module.

### Example with flags
You can skip or set as many flags as your mind desires. Setting all flags will tell the script to combine outliers from all methods, print logs and display images
```bash
python3 scripts/find_outliers.py data -c -v -s
```

## Contributing

Contributions are like clotted cream on scones, always welcome!

## License

This project is as open as the British skies, but check with @matthew-brett first
