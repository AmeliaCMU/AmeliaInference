# AmeliaInference

## Overview

AmeliaInference is a repository that contains the code to run the inference of the Amelia Framework. This repository is part of the Amelia Framework, a framework to generate and evaluate synthetic data for the task of object detection in indoor scenes.

## Pre-requisites

### Dataset

To run this repository, you first need to download the amelia dataset. Follow the instructions [here](https://github.com/AmeliaCMU/AmeliaScenes/DATASET.md) to download and setup the dataset.

Once downloaded, create a symbolic link into  ```datasets```:

```bash
cd datasets
ln -s /path/to/the/amelia/dataset .
```

### Installation

<!-- This repository requires the Amelia Framework, it can be installed following the instructions [here](https://github.com/AmeliaCMU/AmeliaInference/INSTALL.md). However, you can do so following the instructions [here](https://github.com/AmeliaCMU/AmeliaScenes/INSTALL.md) -->

## How to use

Activate your amelia environment (**Please follow the installation instructions above**):

```bash
conda activate amelia
```

Run the testing script:

```bash
python test_bed.py --in_file [in_file]
```

Where `[in_file]` is the path to the test `yaml` file. By default it is set to `default` wich is `configs/test/default.yaml` the. It can be changed to any other file in the `configs/test` directory to run diferent tests.

For some quick changes it can also be set directly in the command line with the following options.

```bash
python test_bed.py --in_file [in_file] --out_dir [out_dir] --dataset_dir [dataset_dir]--dataloader [dataloader] --device [device] --seed [seed]
```

Where:

- `[out_dir]`: The output directory where the results will be saved. By default it is set to `out`.
- `[dataset_dir]`: The directory where the dataset is located. By default it is set to `datasets/amelia`.
- `[dataloader]`: The dataloader to use. By default it is set to `default`.


Other optional arguments are:
-


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
