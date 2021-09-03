# Author Verification

This repository contains code to run LG4AV (Link to paper follows soon!).

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

## Experiments on unseen authors
You can run LG4AV on authors not seen at training time via

```python
PYTHONHASHSEED=42 python new_authors/preprocessing.py
PYTHONHASHSEED=42 python - m new_authors.features
PYTHONHASHSEED=42 python - m new_authors.gai_classify
PYTHONHASHSEED=42 python - m new_authors.kdd_classify
```

Note, that the results depend on the GPU and the exact setup which is used. Hence, the results are expected to differ from the one reported in the paper.
