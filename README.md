# climatereconstructionAI

Software to train climate reconstruction technology (image inpainting with partial convolutions) with numerical model output and to re-fill missing values in observational datasets (e.g., HadCRUT4) using trained models.

## Dependencies
- python>=3.7
- pytorch>=1.8.0
- tqdm>=4.59.0
- torchvision>=0.2.1
- numpy>=1.20.1
- matplotlib>=3.4.3
- tensorboardX>=2.4.0
- tensorboard>=2.8.0
- xarray>=0.20.2
- netcdf4>=1.5.8
- setuptools==59.5.0

An Anaconda environment with all the required dependencies can be created using `environment.yml`:
```bash
conda env create -f environment.yml
```
To activate the environment, use:
```bash
conda activate crai
```

## Installation

`climatereconstructionAI` can be installed using `pip` in the current directory:
```bash
pip install .
```

## Usage

The software can be used to:
- train a model (**training**)
- infill climate datasets using a trained model (**evaluation**)

### Input data
The directory containing the climate datasets should have the following sub-directories:
- `data_large` and `val_large` for training
- `test_large` for evaluation

The climate datasets should be in netCDF format and placed in the corresponding sub-directories.

The missing values can be defined separately as masks. These masks should be in netCDF format and have the same dimension as the climate dataset.

A PyTorch model is required for the evaluation.

### Execution

Once installed, the package can be used as:
- a command line interface (CLI):
  - training:
  ```bash
  crai-train
  ```
  - evaluation:
  ```bash
  crai-evaluate
  ```
- a Python library:
  - training:
  ```python
  from climatereconstructionai import train
  train()
  ```
  - evaluation:
  ```python
  from climatereconstructionai import evaluate
  evaluate()
  ```

For more information about the arguments:
```bash
crai-train --help
usage: crai-train [-h] [--data-root-dir DATA_ROOT_DIR] [--mask-dir MASK_DIR] [--log-dir LOG_DIR] [--img-names IMG_NAMES] [--mask-names MASK_NAMES] [--data-types DATA_TYPES] [--device DEVICE] [--prev-next PREV_NEXT] [--lstm-steps LSTM_STEPS]
                  [--prev-next-steps PREV_NEXT_STEPS] [--encoding-layers ENCODING_LAYERS] [--pooling-layers POOLING_LAYERS] [--image-sizes IMAGE_SIZES] [--weights WEIGHTS] [--attention] [--channel-reduction-rate CHANNEL_REDUCTION_RATE]
                  [--disable-skip-layers] [--disable-first-last-bn] [--out-channels OUT_CHANNELS] [--snapshot-dir SNAPSHOT_DIR] [--resume-iter RESUME_ITER] [--batch-size BATCH_SIZE] [--n-threads N_THREADS] [--finetune] [--lr LR]
                  [--lr-finetune LR_FINETUNE] [--max-iter MAX_ITER] [--log-interval LOG_INTERVAL] [--save-snapshot-image] [--save-model-interval SAVE_MODEL_INTERVAL] [--loss-criterion LOSS_CRITERION] [--eval-timesteps EVAL_TIMESTEPS]
                  [--load-from-file LOAD_FROM_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --data-root-dir DATA_ROOT_DIR
                        Root directory containing the climate datasets
  --mask-dir MASK_DIR   Directory containing the mask datasets
  --log-dir LOG_DIR     Directory where the log files will be stored
  --img-names IMG_NAMES
                        Comma separated list of netCDF files (climate dataset)
  --mask-names MASK_NAMES
                        Comma separated list of netCDF files (mask dataset). If None, it extracts the masks from the climate dataset
  --data-types DATA_TYPES
                        Comma separated list of variable types, in the same order as img-names and mask-names
  --device DEVICE       Device used by PyTorch (cuda or cpu)
  --prev-next PREV_NEXT
  --lstm-steps LSTM_STEPS
                        Number of considered sequences for lstm (0 = lstm module is disabled)
  --prev-next-steps PREV_NEXT_STEPS
  --encoding-layers ENCODING_LAYERS
                        Number of encoding layers in the CNN
  --pooling-layers POOLING_LAYERS
                        Number of pooling layers in the CNN
  --image-sizes IMAGE_SIZES
                        Spatial size of the datasets (latxlon must be of shape NxN)
  --weights WEIGHTS     Initialization weight
  --attention           Enable the attention module
  --channel-reduction-rate CHANNEL_REDUCTION_RATE
                        Channel reduction rate for the attention module
  --disable-skip-layers
                        Disable the skip layers
  --disable-first-last-bn
                        Disable the batch normalization on the first and last layer
  --out-channels OUT_CHANNELS
                        Number of channels for the output image
  --snapshot-dir SNAPSHOT_DIR
                        Parent directory of the training checkpoints and the snapshot images
  --resume-iter RESUME_ITER
                        Iteration step from which the training will be resumed
  --batch-size BATCH_SIZE
                        Batch size
  --n-threads N_THREADS
                        Number of threads
  --finetune            Enable the fine tuning mode (use fine tuning parameterization and disable batch normalization
  --lr LR               Learning rate
  --lr-finetune LR_FINETUNE
                        Learning rate for fine tuning
  --max-iter MAX_ITER   Maximum number of iterations
  --log-interval LOG_INTERVAL
                        Iteration step interval at which a tensorboard summary log should be written
  --save-snapshot-image
                        Save evaluation images for the iteration steps defined in --log-interval
  --save-model-interval SAVE_MODEL_INTERVAL
                        Iteration step interval at which the model should be saved
  --loss-criterion LOSS_CRITERION
                        Index defining the loss function (0=original from Liu et al., 1=MAE of the hole region)
  --eval-timesteps EVAL_TIMESTEPS
                        Iteration steps for which an evaluation is performed
  --load-from-file LOAD_FROM_FILE
                        Load all the arguments from a text file
```

```bash
crai-evaluate --help
usage: crai-evaluate [-h] [--data-root-dir DATA_ROOT_DIR] [--mask-dir MASK_DIR] [--log-dir LOG_DIR] [--img-names IMG_NAMES] [--mask-names MASK_NAMES] [--data-types DATA_TYPES] [--device DEVICE] [--prev-next PREV_NEXT] [--lstm-steps LSTM_STEPS]
                     [--prev-next-steps PREV_NEXT_STEPS] [--encoding-layers ENCODING_LAYERS] [--pooling-layers POOLING_LAYERS] [--image-sizes IMAGE_SIZES] [--weights WEIGHTS] [--attention] [--channel-reduction-rate CHANNEL_REDUCTION_RATE]
                     [--disable-skip-layers] [--disable-first-last-bn] [--out-channels OUT_CHANNELS] [--model-dir MODEL_DIR] [--model-names MODEL_NAMES] [--dataset-name DATASET_NAME] [--evaluation-dirs EVALUATION_DIRS] [--eval-names EVAL_NAMES]
                     [--infill {infill,test}] [--create-graph] [--original-network] [--partitions PARTITIONS] [--maxmem MAXMEM] [--load-from-file LOAD_FROM_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --data-root-dir DATA_ROOT_DIR
                        Root directory containing the climate datasets
  --mask-dir MASK_DIR   Directory containing the mask datasets
  --log-dir LOG_DIR     Directory where the log files will be stored
  --img-names IMG_NAMES
                        Comma separated list of netCDF files (climate dataset)
  --mask-names MASK_NAMES
                        Comma separated list of netCDF files (mask dataset). If None, it extracts the masks from the climate dataset
  --data-types DATA_TYPES
                        Comma separated list of variable types, in the same order as img-names and mask-names
  --device DEVICE       Device used by PyTorch (cuda or cpu)
  --prev-next PREV_NEXT
  --lstm-steps LSTM_STEPS
                        Number of considered sequences for lstm (0 = lstm module is disabled)
  --prev-next-steps PREV_NEXT_STEPS
  --encoding-layers ENCODING_LAYERS
                        Number of encoding layers in the CNN
  --pooling-layers POOLING_LAYERS
                        Number of pooling layers in the CNN
  --image-sizes IMAGE_SIZES
                        Spatial size of the datasets (latxlon must be of shape NxN)
  --weights WEIGHTS     Initialization weight
  --attention           Enable the attention module
  --channel-reduction-rate CHANNEL_REDUCTION_RATE
                        Channel reduction rate for the attention module
  --disable-skip-layers
                        Disable the skip layers
  --disable-first-last-bn
                        Disable the batch normalization on the first and last layer
  --out-channels OUT_CHANNELS
                        Number of channels for the output image
  --model-dir MODEL_DIR
                        Directory of the trained models
  --model-names MODEL_NAMES
                        Model names
  --dataset-name DATASET_NAME
                        Name of the dataset for format checking
  --evaluation-dirs EVALUATION_DIRS
                        Directory where the output files will be stored
  --eval-names EVAL_NAMES
                        Prefix used for the output filenames
  --infill {infill,test}
                        Infill the climate dataset ('test' if mask order is irrelevant, 'infill' if mask order is relevant)
  --create-graph        Create a Tensorboard graph of the NN
  --original-network    Use the original network architecture (from Kadow et al.)
  --partitions PARTITIONS
                        Split the climate dataset into several partitions along the time coordinate
  --maxmem MAXMEM       Maximum available memory in MB (overwrite partitions parameter)
  --load-from-file LOAD_FROM_FILE
                        Load all the arguments from a text file
```

## Example

An example can be found in the directory `demo`.
The instructions to run the example are given in the README.md file.

## License

`climatereconstructionAI` is licensed under the terms of the BSD 3-Clause license.

## Contributions

`climatereconstructionAI` is maintained by the Climate Informatics and Technology group at DKRZ (Deutsches Klimarechenzentrum).
- Previous contributing authors: Naoto Inoue, Christopher Kadow, Stephan Seitz
- Current contributing authors: Johannes Meuer, Étienne Plésiat.
