# LG4AV: Combining Language Models and Graph Neural Networks for Author Verification

This repository contains code to run LG4AV. This is the code of the version which was accepted to the *Symposium on Intelligent Data Analysis 2022*.

For the version which is availablle at [arXiv](https://arxiv.org/abs/2109.01479), checkout the branch ```arxiv```.


## Setup
To run the experiments in this repository, you need Python3.7.
You can install the needed dependencies via `pip install -r requirements.txt`

## Preprocess Data
First, you need to replace the files *data/german_ai_community_dataset.json* and *data/ai_community_dataset.json* with the files which can be found [here](https://zenodo.org/record/3930390).

Then you can do all data preprocessing via

```python
PYTHONHASHSEED=42 python preprocessing/data.py
PYTHONHASHSEED=42 python preprocessing/features.py
```

## Run LG4AV
Then you can run the models of LG4AV-0, LG4AV-2 and LG4AV-F via

```python
PYTHONHASHSEED=42 python -m models.gai_model
PYTHONHASHSEED=42 python -m models.kdd_model
```

**Warning** Running LG4AV on a specific dataset will store all 30 models with checkpoints which can take up to 3GB per model.
Note, that the results depend on the GPU and the exact setup which is used. Hence, the results are expected to differ from the one reported in the paper.

Note, that the results depend on the GPU and the exact setup which is used. Hence, the results are expected to differ from the one reported in the paper.
