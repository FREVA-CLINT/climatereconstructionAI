# Infilling spatial precipitation recordings with a memory assisted CNN

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

`python train_and_evaluate/train.py`

To specify additional args such as the data root directory, use `--arg arg_value`.
Here are some important args:
- `--data-root-dir` -> root directory of training and validation data
- `--snapshot-dir` -> directory of training checkpoints
- `--mask-dir` -> directory of mask files
- `--img-names` -> comma separated list of training data files stored in the data root directory, have to be same shape! First image is ground truth
- `--mask-names` -> comma separated list of mask files stored in the mask directory, need to correspond to order in img-names
- `--data-types` -> comma separated list of types of variable, need to correspond to order in img-names and mask-names
- `--device` -> cuda or cpu
- `--lstm-steps` -> Number of considered sequences for lstm, set to zero, if lstm module should be deactivated
- `--encoding-layers` -> number of encoding layers in the CNN
- `--pooling-layers` -> number of pooling layers in the CNN
- `--image-size` -> size of image, must be of shape NxN

### Evaluate
The evaluation process can be started by executing

`python train_and_evaluate/evaluate.py`

Important args:
- `--evaluation-dir` -> directory where evaluations will be stored
- `--data-root-dir` -> root directory of test data
- `--mask-dir` -> directory of mask files
- `--img-names` -> comma separated list of training data files stored in the data root directory, have to be same shape!
- `--mask-names` -> comma separated list of mask files stored in the mask directory, need to correspond to order in img-names
- `--data-types` -> comma separated list of types of variable, need to correspond to order in img-names and mask-names
- `--device` -> cuda or cpu
- `--lstm-steps` -> Number of considered sequences for lstm, set to zero, if lstm module should be deactivated
- `--infill` -> 'test', if mask order is irrelevant, 'infill', if mask order is relevant
- `--create-images` -> creates images for time window in format 'YYY-MM-DD-HH:MM,YYYY-MM-DD-HH:MM'
- `--create-video` -> creates video. Images need to be created as well
- `--create-report` -> creates evaluation report for total test data set