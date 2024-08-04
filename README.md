# EOL: Transductive Few-Shot Open-Set Recognition by Enhancing Outlier Logits

![Python Versions](https://img.shields.io/badge/python-3.8-%23EBBD68.svg)

# Getting started

## Installation

```bash
virtualenv venv --python=python3.8 
source venv/bin/activate
pip install -r requirements.txt
```

On top of common packages, this project uses [pyod](https://pyod.readthedocs.io/en/latest/), [timm](https://github.com/rwightman/pytorch-image-models) and [easyfsl](https://pypi.org/project/easyfsl/). For more details, follow the instructions from the repository for ["Open-Set Likelihook Maximization for Few-Shot Learning"](https://github.com/ebennequin/few-shot-open-set). 

## Download features

This code uses several models but you can download manually the extracted features from this [annonymized folder](https://drive.google.com/drive/folders/1UmeVoDTnJoQ6zazjMmzhVTTRJeR6Qy3P?usp=drive_link) (1.32GB uncompressed). 

Place the features inside `data/features/<dataset>/<dataset>/test/standard/<model>_<dataset>_feat.pickle`, where `<dataset>` is one of the datasets used (`mini_imagenet`, `cub`, `aircraft`) and model is one of the models used (`resnet12`, `AGWR18ibn`, `AGWR50ibn`, `clip_vit_base_patch16`, `vit_base_patch16_224`, `efficientnet_b0`). 

## Running EOL on standard setting

All commands to reproduce any results in the paper is contained as a recipe in the `Makefile`.

On the default task run our method like this:
```
make run_eol
```

To change the task settings, modify `Makefile` or though command line as shown below. To change the hyperparameter settings for any of the algorithms, change them inside the `.yaml` files inside the `configs/`. 

## Reproducing results

Balanced: OOD_QUERY=15 ID_QUERY=15
```
make EXP=benchmark N_TASKS=1000 SHOTS=5 OVERRIDE=FALSE SEEDS="0 1 2 3 4 5 6" benchmark_backbone_dataset
```

Imbalanced: OOD_QUERY=5 ID_QUERY=25
```
make EXP=imbalanced_5ood DETECTOR_CONFIG_FILE=configs/detectors_5ood.yaml OOD_QUERY=5 ID_QUERY=25 N_TASKS=1000 SHOTS=5 OVERRIDE=FALSE benchmark_backbone_dataset
```

Imbalanced: OOD_QUERY=25 ID_QUERY=5
```
make EXP=imbalanced_25ood DETECTOR_CONFIG_FILE=configs/detectors_25ood.yaml OOD_QUERY=25 ID_QUERY=5 N_TASKS=1000 SHOTS=5 OVERRIDE=FALSE benchmark_backbone_dataset
```

# Acknowledgements

The code was adapted from ["Open-Set Likelihook Maximization for Few-Shot Learning"](https://github.com/ebennequin/few-shot-open-set).
