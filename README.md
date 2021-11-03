# Infilling spatial precipitation recordings with a memory assisted CNN

**[Applied implementation] (https://github.com/naoto0804/pytorch-inpainting-with-partial-conv)**

**[Official implementation](https://github.com/NVIDIA/partialconv) is released by the authors.**

**Note that this is an ongoing re-implementation and is designed for climate reconstructions using numerial model input and output!**

This is an unofficial pytorch implementation of a paper, [Image Inpainting for Irregular Holes Using Partial Convolutions](https://arxiv.org/abs/1804.07723) [Liu+, arXiv2018].

## Requirements
- Python 3.7+

```
pip install -r requirements.txt
```

## Usage

### Preprocess 
Download climate data. The dataset should contain `data_large`, `val_large`, and `test_large` as the subdirectories and should be in netcdf file format.

### Training
The training process can be started by executing 

`python train.py`

To specify additional args such as the data root directory, use `--arg arg_value`.
Here are some important args:
- `--root-dir` -> root directory of the data
- `--snapshot-dir` -> directory of training checkpoints
- `--mask-dir` -> directory of mask file
- `--device` -> cuda or cpu
- `--data-type` -> type of variable in netCDF file
- `--prev-next` -> Number of previous and following states that should be considered for training process
- `--encoding-layers` -> number of encoding layers in the CNN
- `--pooling-layers` -> number of pooling layers in the CNN
- `--image-size` -> size of image, must be of shape NxN

### Evaluate
The evaluation process can be started by executing

`python evaluate.py`

Important args:
- `--root-dir` -> root directory of the data
- `--evaluation-dir` -> directory where evaluations will be stored
- `--infill` -> 'test', if mask order is irrelevant, 'infill', if mask order is relevant
- `--create-images` -> creates images for time window in format 'YYY-MM-DD-HH:MM,YYYY-MM-DD-HH:MM'
- `--create-video` -> creates video. Images need to be created as well
- `--create-report` -> creates evaluation report for total test data set