# Raman Data Generator

[![Generic badge](https://img.shields.io/badge/python-v3.6+-<COLOR>.svg)]() [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project aims to offer a fast :zap: and reliable data augmentation generator of **Raman spectra**

## Arguments

### Basic
Param|Type|Description
---|---|---
df|pandas.DataFrame|A pandas dataframe with shift's values as columns + a column called "labels" for the categories
batch_size|int|batch size of samples
max_classes|int|categories in the labels

### Advanced
The standard paramenter were validated on a Raman task, however if you need a greater customization you can still tweak them!

The augmentation process works as follow.
For each $sample_i$ of the current batch, takes another sample of the same class $sample_j$ (randomly) and performes:
1. __roll__ (shift horizontally, i used the _roll_ term because it's easy to misunderstand the _horizontal shift_ with the _Raman's shift_) $sample_j$ of some __roll_factor__ (Raman's shift values).

2. a __weighted sum__ with respect of some $a$ probability variable

$$ sample_k = a·sample_i + (1-a)·sample_j $$

This augmentation step is based on the assumption that two samples of the same class are semantically equal (natural class variability) + some sensor noise.

3. on $sample_k$ apply a __slope__ of some __slope factor__, which is baseline linear error that emulates the fluorescence issue of some sensors.
4. on $sample_k$ apply addittive white gaussian noise to the signal

Param|Type|Description
-|---|---
roll|bool|Enable/disable the __roll__ step during the augmentation
roll_factor|int|The signal is rolled(horizontal shifted) of this amount of shifts. It rolls along the dataframe columns. If a signal has a precision of 10 Raman's shifts, wich means that the columns increase 10 shifts at time, using a roll factor of 5, it actually shifts 10*5 = 50 shifts 
slope|bool|Enable/disable the __slope__ step during the augmentation
slope_factor|float|It's the slope angle of the baseline linear error
noise|bool|Enable/disable the __noise__ step during the augmentation
noise_range|tuple|The noise factor is sampled in this range. e.g. (min, max)


## Requirements
The python libraries needed are:

    random
    dataclasses
    pandas
    numpy
    tensorflow

---
The code is documented for more insightful informations :wink: !

Contributors are welcome :thumbsup:
