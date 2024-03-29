> This repository aims to build a text encoder network which can be used to encode texts into vectors and calculate the **semantic textual similarity (STS)** between these texts.

## Introduction

`Step1: Training`

This method will choose classification tasks as the optimization objective rather than optimizing textual similarity directly so that we do not need to sample negative samples from datasets which is always taken by many typical methods like **Siamese network**.Then, we choose [amsoftmax](https://arxiv.org/abs/1801.05599) as loss function which can ensure that smaples in the same class are more cohesive.
These two advantages greatly improve the performance.

`Step2: Applying`

After training, we take the hidden layer before classification layer as textual representation vectors and calculate the semantic similarity.

## Requirements
- tensorflow >=1.4.0
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





