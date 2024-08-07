# AmeliaInference

## Overview

AmeliaInference is a repository that contains the code to run the Amelia Framework. It is a framework that allows to run tests on the amelia dataset based on th hydra library, allowing to run tests with different configurations and dataloaders.

## Pre-requisites

### Dataset

To run this repository, you first need to download the amelia dataset. Follow the instructions [here](https://github.com/AmeliaCMU/AmeliaScenes/DATASET.md) to download and setup the dataset.

Once downloaded, create a symbolic link into  `datasets`:

```bash
cd datasets
ln -s /path/to/the/amelia/dataset .
```

### Installation

This repository requires the AmeliaTF, it can be installed following the instructions [here](https://github.com/AmeliaCMU/AmeliaTF/INSTALL.md). Once installed, in the same environment follow the instructons in [INSTALL.md](https://github.com/AmeliaCMU/AmeliaInference/INSTALL.md). However, you can intall the Amelia Framework following the instructions [here](https://github.com/AmeliaCMU/AmeliaScenes/INSTALL.md).

## How to use

Activate your amelia environment (**Please follow the installation instructions above**):

```bash
conda activate amelia
```

Run the testing script:

```bash
python run_inference.py -m test=[test_name]
```

Where `[test_name]` is the name of the test to run. The tests files are in a `yaml` format and are located in the `configs/test` directory. The default test is `default.yaml`. The test files contain the configuration for the test, including the model, the dataloader, the dataset, the device, the seed, and the output directory.

For some quick changes you may refer to hydras' [documentation](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/). The next example shows how to change the test file to `example_kbos_critical.yaml`, which is a test file that uses the KBOS dataset and the critical model and it is included in th erepository as an example.

```bash
python run_inference.py -m test=example_kbos_critical
```

Giving the following output in the `output` directory:

```bash
output
  |-- KBOS_26_1672610400_critical_ego
    |-- .hyra
    |-- kbos_scene_*.png
  |-- KBOS_26_1672621200_critical_ego
    |-- .hyra
    |-- kbos_scene_*.png
```

Where:

- `KBOS_26_1672610400_critical_ego` and `KBOS_26_1672621200_critical_ego` are the directories with the specifications configured in the `example_kbos_critical.yaml` file.
- `kbos_scene_*.png` files are the images with the predictions generated by the model for each scene.

## BibTeX

Our paper:

**Amelia: A Large Dataset and Model for Airport Surface Movement Forecasting [[paper](https://arxiv.org/pdf/2407.21185)]**

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
