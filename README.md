> This repository aims to build a text encoder network which can be used to encode texts into vectors and calculate the semantic textual similarity between these texts.

## Introduction

`**Step1: Training**`

This method chooses classification tasks as the optimization objective rather than optimizing textual similarity directly so that we do not need to sample negative samples from datasets that is done by many typical methods like **Siamese network**.Then, we choose **amsoftmax** as loss function which can ensure that smaples in the same class are more cohesive.

`**Step2: Applying**`

After training, we take the hidden layer before classification layer as textual representation vectors and calculate the semantic similarity.

## Requirements
- tensorflow >=1.0.4
- python3
- tqdm
- yaml

## Structure

- data_helper.py 

    including data process and batch generation 

- sent_simtf.py 

    including main model

- train.py

    including train process and evaluation process

- config.yaml

    including all hyper-parameters





