# AmeliaInference

This repository contains the inference code for the model introduced in the paper below:

***Amelia: A Large Dataset and Model for Airport Surface Movement Forecasting [[paper](https://arxiv.org/pdf/2407.21185)]***

[Ingrid Navarro](https://navars.xyz) *, [Pablo Ortega-Kral](https://paok-2001.github.io) *, [Jay Patrikar](https://www.jaypatrikar.me) *, Haichuan Wang,
Zelin Ye, Jong Hoon Park, [Jean Oh](https://cmubig.github.io/team/jean_oh/) and [Sebastian Scherer](https://theairlab.org/team/sebastian/)

## Overview

**AmeliaInference**: Tool intended to be a standalone version of [AmeliaTF](github.com/AmeliaCMU/AmeliaTF) used only for inference purposes without the training/evaluation/testing overhead code.

If you're interested in training trajectory forecasting models, please refer to [AmeliaTF](github.com/AmeliaCMU/AmeliaTF).

## Pre-requisites

#### Dataset

To run this repository, you first need to download the amelia dataset. Follow the instructions [here](https://ameliacmu.github.io/amelia-dataset/) to download the dataset.

Once downloaded, create a symbolic link into  ```datasets```:

```bash
cd datasets
ln -s /path/to/amelia .
```

### Installation

Make sure that you have [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) installed.

**Recommended:** Use the  [`install.sh`](https://github.com/AmeliaCMU/AmeliaScenes/blob/main/install.sh) to download and install the Amelia Framework:

```bash
chmod +x install.sh
./install.sh amelia
```

This will create a conda environment named `amelia` and install all dependencies.

Alternatively, refer to [`INSTALL.md`](https://github.com/AmeliaCMU/AmeliaScenes/blob/main/INSTALL.md) for manual installation.

**Note:** AmeliaInference requires the Amelia dataset and AmeliaTF dependencies to run, refer to AmeliaTF's and AmeliaInference's installation.

## How to use

Activate your amelia environment (**Please follow the installation instructions above**):

```bash
conda activate amelia
```

Run the testing script:

```bash
python run_inference.py -m tests=<test_name>
```

Where:

- `<test_name>` is the name of the test to run. The tests files are in a `yaml` format and are located in the `configs/test` directory. The default test is `default.yaml`. The test files contain the configuration for the test, including the model, the dataloader, the dataset, the device, the seed, and the output directory.

### Example

The next example shows how to change the test file to `example_kbos_critical.yaml`, which is a test file that uses the KBOS dataset and the critical model and it is included in the repository as an example.

```bash
python run_inference.py -m tests=example_kbos_critical
```

Produces the following output in the `output` directory:

```bash
output
  |-- KBOS_26_1672610400_critical_ego
    |-- .hydra
    |-- kbos_scene_*.png
  |-- KBOS_26_1672621200_critical_ego
    |-- .hydra
    |-- kbos_scene_*.png
```

where:

- `KBOS_26_1672610400_critical_ego` and `KBOS_26_1672621200_critical_ego` are the directories with the specifications configured in the `example_kbos_critical.yaml` file.
- `kbos_scene_*.png` files are the images with the predictions generated by the model for each scene.

## BibTeX

If you find our work useful in your research, please cite us!

```bibtex
@inbook{navarro2024amelia,
  author = {Ingrid Navarro and Pablo Ortega and Jay Patrikar and Haichuan Wang and Zelin Ye and Jong Hoon Park and Jean Oh and Sebastian Scherer},
  title = {AmeliaTF: A Large Model and Dataset for Airport Surface Movement Forecasting},
  booktitle = {AIAA AVIATION FORUM AND ASCEND 2024},
  chapter = {},
  pages = {},
  doi = {10.2514/6.2024-4251},
  URL = {https://arc.aiaa.org/doi/abs/10.2514/6.2024-4251},
  eprint = {https://arc.aiaa.org/doi/pdf/10.2514/6.2024-4251},
}
```
<!--
TODO: install from git
'amelia_tf @ git+https://github.com/AmeliaCMU/AmeliaTF@main' -->
