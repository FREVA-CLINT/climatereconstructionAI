# CRAI (Climate Reconstruction AI)

Software to train/evaluate models to reconstruct missing values in climate data (e.g., HadCRUT4) based on a U-Net with partial convolutions.

## Dependencies
- pytorch>=1.11.0
- tqdm>=4.64.0
- torchvision>=0.12.0
- numpy>=1.21.6
- matplotlib>=3.5.1
- tensorboardX>=2.5
- tensorboard>=2.9.0
- xarray>=2022.3.0
- dask>=2022.7.0
- netcdf4>=1.5.8
- setuptools==59.5.0
- xesmf>=0.6.2
- cartopy>=0.20.2
- numba>=0.55.1

An Anaconda environment with all the required dependencies can be created using `environment.yml`:
```bash
conda env create -f environment.yml
```
To activate the environment, use:
```bash
conda activate crai
```

`environment-cuda.yml` should be used when working with GPUs using CUDA.

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
usage: crai-train [-h] [--data-root-dir DATA_ROOT_DIR] [--mask-dir MASK_DIR] [--log-dir LOG_DIR] [--data-names DATA_NAMES] [--mask-names MASK_NAMES] [--data-types DATA_TYPES]
                  [--n-target-data N_TARGET_DATA] [--device DEVICE] [--shuffle-masks] [--channel-steps CHANNEL_STEPS] [--lstm-steps LSTM_STEPS] [--gru-steps GRU_STEPS] [--encoding-layers ENCODING_LAYERS]
                  [--pooling-layers POOLING_LAYERS] [--conv-factor CONV_FACTOR] [--weights WEIGHTS] [--steady-masks STEADY_MASKS] [--loop-random-seed LOOP_RANDOM_SEED]
                  [--cuda-random-seed CUDA_RANDOM_SEED] [--deterministic] [--attention] [--channel-reduction-rate CHANNEL_REDUCTION_RATE] [--disable-skip-layers] [--disable-first-bn] [--masked-bn]
                  [--lazy-load] [--global-padding] [--normalize-data] [--n-filters N_FILTERS] [--out-channels OUT_CHANNELS] [--dataset-name DATASET_NAME] [--min-bounds MIN_BOUNDS]
                  [--max-bounds MAX_BOUNDS] [--profile] [--val-names VAL_NAMES] [--snapshot-dir SNAPSHOT_DIR] [--resume-iter RESUME_ITER] [--batch-size BATCH_SIZE] [--n-threads N_THREADS] [--multi-gpus]
                  [--finetune] [--lr LR] [--lr-finetune LR_FINETUNE] [--max-iter MAX_ITER] [--log-interval LOG_INTERVAL] [--lr-scheduler-patience LR_SCHEDULER_PATIENCE]
                  [--save-model-interval SAVE_MODEL_INTERVAL] [--n-final-models N_FINAL_MODELS] [--final-models-interval FINAL_MODELS_INTERVAL] [--loss-criterion LOSS_CRITERION]
                  [--eval-timesteps EVAL_TIMESTEPS] [-f LOAD_FROM_FILE] [--vlim VLIM]

options:
  -h, --help            show this help message and exit
  --data-root-dir DATA_ROOT_DIR
                        Root directory containing the climate datasets
  --mask-dir MASK_DIR   Directory containing the mask datasets
  --log-dir LOG_DIR     Directory where the log files will be stored
  --data-names DATA_NAMES
                        Comma separated list of netCDF files (climate dataset) for training/infilling
  --mask-names MASK_NAMES
                        Comma separated list of netCDF files (mask dataset). If None, it extracts the masks from the climate dataset
  --data-types DATA_TYPES
                        Comma separated list of variable types, in the same order as data-names and mask-names
  --n-target-data N_TARGET_DATA
                        Number of data-names (from last) to be used as target data
  --device DEVICE       Device used by PyTorch (cuda or cpu)
  --shuffle-masks       Select mask indices randomly
  --channel-steps CHANNEL_STEPS
                        Comma separated number of considered sequences for channeled memory:past_steps,future_steps
  --lstm-steps LSTM_STEPS
                        Comma separated number of considered sequences for lstm: past_steps,future_steps
  --gru-steps GRU_STEPS
                        Comma separated number of considered sequences for gru: past_steps,future_steps
  --encoding-layers ENCODING_LAYERS
                        Number of encoding layers in the CNN
  --pooling-layers POOLING_LAYERS
                        Number of pooling layers in the CNN
  --conv-factor CONV_FACTOR
                        Number of channels in the deepest layer
  --weights WEIGHTS     Initialization weight
  --steady-masks STEADY_MASKS
                        Comma separated list of netCDF files containing a single mask to be applied to all timesteps. The number of steady-masks must be the same as out-channels
  --loop-random-seed LOOP_RANDOM_SEED
                        Random seed for iteration loop
  --cuda-random-seed CUDA_RANDOM_SEED
                        Random seed for CUDA
  --deterministic       Disable cudnn backends for reproducibility
  --attention           Enable the attention module
  --channel-reduction-rate CHANNEL_REDUCTION_RATE
                        Channel reduction rate for the attention module
  --disable-skip-layers
                        Disable the skip layers
  --disable-first-bn    Disable the batch normalization on the first layer
  --masked-bn           Use masked batch normalization instead of standard BN
  --lazy-load           Use lazy loading for large datasets
  --global-padding      Use a custom padding for global dataset
  --normalize-data      Normalize the input climate data to 0 mean and 1 std
  --n-filters N_FILTERS
                        Number of filters for the first/last layer
  --out-channels OUT_CHANNELS
                        Number of channels for the output data
  --dataset-name DATASET_NAME
                        Name of the dataset for format checking
  --min-bounds MIN_BOUNDS
                        Comma separated list of values defining the permitted lower-bound of output values
  --max-bounds MAX_BOUNDS
                        Comma separated list of values defining the permitted upper-bound of output values
  --profile             Profile code using tensorboard profiler
  --val-names VAL_NAMES
                        Comma separated list of netCDF files (climate dataset) for validation
  --snapshot-dir SNAPSHOT_DIR
                        Parent directory of the training checkpoints and the snapshot images
  --resume-iter RESUME_ITER
                        Iteration step from which the training will be resumed
  --batch-size BATCH_SIZE
                        Batch size
  --n-threads N_THREADS
                        Number of workers used in the data loader
  --multi-gpus          Use multiple GPUs, if any
  --finetune            Enable the fine tuning mode (use fine tuning parameterization and disable batch normalization
  --lr LR               Learning rate
  --lr-finetune LR_FINETUNE
                        Learning rate for fine tuning
  --max-iter MAX_ITER   Maximum number of iterations
  --log-interval LOG_INTERVAL
                        Iteration step interval at which a tensorboard summary log should be written
  --lr-scheduler-patience LR_SCHEDULER_PATIENCE
                        Patience for the lr scheduler
  --save-model-interval SAVE_MODEL_INTERVAL
                        Iteration step interval at which the model should be saved
  --n-final-models N_FINAL_MODELS
                        Number of final models to be saved
  --final-models-interval FINAL_MODELS_INTERVAL
                        Iteration step interval at which the final models should be saved
  --loss-criterion LOSS_CRITERION
                        Index defining the loss function (0=original from Liu et al., 1=MAE of the hole region)
  --eval-timesteps EVAL_TIMESTEPS
                        Sample indices for which a snapshot is created at each iter defined by log-interval
  -f LOAD_FROM_FILE, --load-from-file LOAD_FROM_FILE
                        Load all the arguments from a text file
  --vlim VLIM           Comma separated list of vmin,vmax values for the color scale of the snapshot images
```

```bash
usage: crai-evaluate [-h] [--data-root-dir DATA_ROOT_DIR] [--mask-dir MASK_DIR] [--log-dir LOG_DIR] [--data-names DATA_NAMES] [--mask-names MASK_NAMES] [--data-types DATA_TYPES]
                     [--n-target-data N_TARGET_DATA] [--device DEVICE] [--shuffle-masks] [--channel-steps CHANNEL_STEPS] [--lstm-steps LSTM_STEPS] [--gru-steps GRU_STEPS]
                     [--encoding-layers ENCODING_LAYERS] [--pooling-layers POOLING_LAYERS] [--conv-factor CONV_FACTOR] [--weights WEIGHTS] [--steady-masks STEADY_MASKS]
                     [--loop-random-seed LOOP_RANDOM_SEED] [--cuda-random-seed CUDA_RANDOM_SEED] [--deterministic] [--attention] [--channel-reduction-rate CHANNEL_REDUCTION_RATE] [--disable-skip-layers]
                     [--disable-first-bn] [--masked-bn] [--lazy-load] [--global-padding] [--normalize-data] [--n-filters N_FILTERS] [--out-channels OUT_CHANNELS] [--dataset-name DATASET_NAME]
                     [--min-bounds MIN_BOUNDS] [--max-bounds MAX_BOUNDS] [--profile] [--model-dir MODEL_DIR] [--model-names MODEL_NAMES] [--evaluation-dirs EVALUATION_DIRS] [--eval-names EVAL_NAMES]
                     [--use-train-stats] [--create-graph] [--plot-results PLOT_RESULTS] [--partitions PARTITIONS] [--maxmem MAXMEM] [--split-outputs] [-f LOAD_FROM_FILE]

options:
  -h, --help            show this help message and exit
  --data-root-dir DATA_ROOT_DIR
                        Root directory containing the climate datasets
  --mask-dir MASK_DIR   Directory containing the mask datasets
  --log-dir LOG_DIR     Directory where the log files will be stored
  --data-names DATA_NAMES
                        Comma separated list of netCDF files (climate dataset) for training/infilling
  --mask-names MASK_NAMES
                        Comma separated list of netCDF files (mask dataset). If None, it extracts the masks from the climate dataset
  --data-types DATA_TYPES
                        Comma separated list of variable types, in the same order as data-names and mask-names
  --n-target-data N_TARGET_DATA
                        Number of data-names (from last) to be used as target data
  --device DEVICE       Device used by PyTorch (cuda or cpu)
  --shuffle-masks       Select mask indices randomly
  --channel-steps CHANNEL_STEPS
                        Comma separated number of considered sequences for channeled memory:past_steps,future_steps
  --lstm-steps LSTM_STEPS
                        Comma separated number of considered sequences for lstm: past_steps,future_steps
  --gru-steps GRU_STEPS
                        Comma separated number of considered sequences for gru: past_steps,future_steps
  --encoding-layers ENCODING_LAYERS
                        Number of encoding layers in the CNN
  --pooling-layers POOLING_LAYERS
                        Number of pooling layers in the CNN
  --conv-factor CONV_FACTOR
                        Number of channels in the deepest layer
  --weights WEIGHTS     Initialization weight
  --steady-masks STEADY_MASKS
                        Comma separated list of netCDF files containing a single mask to be applied to all timesteps. The number of steady-masks must be the same as out-channels
  --loop-random-seed LOOP_RANDOM_SEED
                        Random seed for iteration loop
  --cuda-random-seed CUDA_RANDOM_SEED
                        Random seed for CUDA
  --deterministic       Disable cudnn backends for reproducibility
  --attention           Enable the attention module
  --channel-reduction-rate CHANNEL_REDUCTION_RATE
                        Channel reduction rate for the attention module
  --disable-skip-layers
                        Disable the skip layers
  --disable-first-bn    Disable the batch normalization on the first layer
  --masked-bn           Use masked batch normalization instead of standard BN
  --lazy-load           Use lazy loading for large datasets
  --global-padding      Use a custom padding for global dataset
  --normalize-data      Normalize the input climate data to 0 mean and 1 std
  --n-filters N_FILTERS
                        Number of filters for the first/last layer
  --out-channels OUT_CHANNELS
                        Number of channels for the output data
  --dataset-name DATASET_NAME
                        Name of the dataset for format checking
  --min-bounds MIN_BOUNDS
                        Comma separated list of values defining the permitted lower-bound of output values
  --max-bounds MAX_BOUNDS
                        Comma separated list of values defining the permitted upper-bound of output values
  --profile             Profile code using tensorboard profiler
  --model-dir MODEL_DIR
                        Directory of the trained models
  --model-names MODEL_NAMES
                        Model names
  --evaluation-dirs EVALUATION_DIRS
                        Directory where the output files will be stored
  --eval-names EVAL_NAMES
                        Prefix used for the output filenames
  --use-train-stats     Use mean and std from training data for normalization
  --create-graph        Create a Tensorboard graph of the NN
  --plot-results PLOT_RESULTS
                        Create plot images of the results for the comma separated list of time indices
  --partitions PARTITIONS
                        Split the climate dataset into several partitions along the time coordinate
  --maxmem MAXMEM       Maximum available memory in MB (overwrite partitions parameter)
  --split-outputs       Do not merge the outputs when using multiple models and/or partitions
  -f LOAD_FROM_FILE, --load-from-file LOAD_FROM_FILE
                        Load all the arguments from a text file
```

## Example

An example can be found in the directory `demo`.
The instructions to run the example are given in the README.md file.

## License

`CRAI` is licensed under the terms of the BSD 3-Clause license.

## Contributions

`CRAI` is maintained by the Climate Informatics and Technology group at DKRZ (Deutsches Klimarechenzentrum).
- Previous contributing authors: Naoto Inoue, Christopher Kadow, Stephan Seitz
- Current contributing authors: Johannes Meuer, Étienne Plésiat.
