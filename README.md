# climatereconstructionAI

Software to train climate reconstruction technology (image inpainting with partial convolutions) with numerical model output and to re-fill missing values in observational datasets (e.g., HadCRUT4) using trained models.

## Requirements
- python>=3.7
- imageio>=2.9.0
- pandas>=1.2.5
- fpdf>=1.7.2
- torch>=1.8.0
- h5py>=3.2.1
- tqdm>=4.59.0
- torchvision>=0.2.1
- numpy>=1.20.1
- matplotlib>=3.4.3
- tensorboardX>=2.4.0
- xarray>=0.20.2
- netcdf4>=1.5.8

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
usage: crai-train [-h] [--data-root-dir DATA_ROOT_DIR]
                  [--snapshot-dirs SNAPSHOT_DIRS] [--mask-dir MASK_DIR]
                  [--img-names IMG_NAMES] [--mask-names MASK_NAMES]
                  [--data-types DATA_TYPES] [--device DEVICE]
                  [--prev-next PREV_NEXT] [--lstm-steps LSTM_STEPS]
                  [--prev-next-steps PREV_NEXT_STEPS]
                  [--encoding-layers ENCODING_LAYERS]
                  [--pooling-layers POOLING_LAYERS]
                  [--image-sizes IMAGE_SIZES] [--attention]
                  [--channel-reduction-rate CHANNEL_REDUCTION_RATE]
                  [--disable-skip-layers] [--out-channels OUT_CHANNELS]
                  [--log-dir LOG_DIR] [--resume-iter RESUME_ITER]
                  [--batch-size BATCH_SIZE] [--n-threads N_THREADS]
                  [--finetune] [--lr LR] [--lr-finetune LR_FINETUNE]
                  [--max-iter MAX_ITER] [--log-interval LOG_INTERVAL]
                  [--save-snapshot-image]
                  [--save-model-interval SAVE_MODEL_INTERVAL]
                  [--loss-criterion LOSS_CRITERION]
                  [--eval-timesteps EVAL_TIMESTEPS] [--weights WEIGHTS]
                  [--load-from-file LOAD_FROM_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --data-root-dir DATA_ROOT_DIR
                        Root directory containing the climate datasets
  --snapshot-dirs SNAPSHOT_DIRS
                        Directory where the training checkpoints will be
                        stored
  --mask-dir MASK_DIR   Directory containing the mask datasets
  --img-names IMG_NAMES
                        Comma separated list of netCDF files (climate dataset)
  --mask-names MASK_NAMES
                        Comma separated list of netCDF files (mask dataset).
                        If None, it extracts the masks from the climate
                        dataset
  --data-types DATA_TYPES
                        Comma separated list of variable types, in the same
                        order as img-names and mask-names
  --device DEVICE       Device used by PyTorch (cuda or cpu)
  --prev-next PREV_NEXT
  --lstm-steps LSTM_STEPS
                        Number of considered sequences for lstm (0 = lstm
                        module is disabled)
  --prev-next-steps PREV_NEXT_STEPS
  --encoding-layers ENCODING_LAYERS
                        Number of encoding layers in the CNN
  --pooling-layers POOLING_LAYERS
                        Number of pooling layers in the CNN
  --image-sizes IMAGE_SIZES
                        Spatial size of the datasets (latxlon must be of shape
                        NxN)
  --attention           Enable the attention module
  --channel-reduction-rate CHANNEL_REDUCTION_RATE
                        Channel reduction rate for the attention module
  --disable-skip-layers
                        Disable the skip layers
  --out-channels OUT_CHANNELS
                        Number of channels for the output image
  --log-dir LOG_DIR     Directory where the log files will be stored
  --resume-iter RESUME_ITER
                        Iteration step from which the training will be resumed
  --batch-size BATCH_SIZE
                        Batch size
  --n-threads N_THREADS
                        Number of threads
  --finetune            Enable the fine tuning mode (use fine tuning
                        parameterization and disable batch normalization
  --lr LR               Learning rate
  --lr-finetune LR_FINETUNE
                        Learning rate for fine tuning
  --max-iter MAX_ITER   Maximum number of iterations
  --log-interval LOG_INTERVAL
                        Iteration step interval at which a tensorboard summary
                        log should be written
  --save-snapshot-image
                        Save evaluation images for the iteration steps defined
                        in --log-interval
  --save-model-interval SAVE_MODEL_INTERVAL
                        Iteration step interval at which the model should be
                        saved
  --loss-criterion LOSS_CRITERION
                        Index defining the loss function (0=original from Liu
                        et al., 1=MAE of the hole region)
  --eval-timesteps EVAL_TIMESTEPS
                        Iteration steps for which an evaluation is performed
  --weights WEIGHTS     Initialization weight
  --load-from-file LOAD_FROM_FILE
                        Load all the arguments from a text file
```

```bash
crai-evaluate --help
usage: crai-evaluate [-h] [--data-root-dir DATA_ROOT_DIR]
                     [--snapshot-dirs SNAPSHOT_DIRS] [--mask-dir MASK_DIR]
                     [--img-names IMG_NAMES] [--mask-names MASK_NAMES]
                     [--data-types DATA_TYPES] [--device DEVICE]
                     [--prev-next PREV_NEXT] [--lstm-steps LSTM_STEPS]
                     [--prev-next-steps PREV_NEXT_STEPS]
                     [--encoding-layers ENCODING_LAYERS]
                     [--pooling-layers POOLING_LAYERS]
                     [--image-sizes IMAGE_SIZES] [--attention]
                     [--channel-reduction-rate CHANNEL_REDUCTION_RATE]
                     [--disable-skip-layers] [--out-channels OUT_CHANNELS]
                     [--evaluation-dirs EVALUATION_DIRS]
                     [--eval-names EVAL_NAMES] [--infill {infill,test}]
                     [--create-images CREATE_IMAGES] [--create-video]
                     [--fps FPS] [--create-report] [--eval-range EVAL_RANGE]
                     [--eval-threshold EVAL_THRESHOLD]
                     [--smoothing-factor SMOOTHING_FACTOR]
                     [--convert-to-netcdf] [--partitions PARTITIONS]
                     [--load-from-file LOAD_FROM_FILE]

optional arguments:
  -h, --help            show this help message and exit
  --data-root-dir DATA_ROOT_DIR
                        Root directory containing the climate datasets
  --snapshot-dirs SNAPSHOT_DIRS
                        Directory where the training checkpoints will be
                        stored
  --mask-dir MASK_DIR   Directory containing the mask datasets
  --img-names IMG_NAMES
                        Comma separated list of netCDF files (climate dataset)
  --mask-names MASK_NAMES
                        Comma separated list of netCDF files (mask dataset).
                        If None, it extracts the masks from the climate
                        dataset
  --data-types DATA_TYPES
                        Comma separated list of variable types, in the same
                        order as img-names and mask-names
  --device DEVICE       Device used by PyTorch (cuda or cpu)
  --prev-next PREV_NEXT
  --lstm-steps LSTM_STEPS
                        Number of considered sequences for lstm (0 = lstm
                        module is disabled)
  --prev-next-steps PREV_NEXT_STEPS
  --encoding-layers ENCODING_LAYERS
                        Number of encoding layers in the CNN
  --pooling-layers POOLING_LAYERS
                        Number of pooling layers in the CNN
  --image-sizes IMAGE_SIZES
                        Spatial size of the datasets (latxlon must be of shape
                        NxN)
  --attention           Enable the attention module
  --channel-reduction-rate CHANNEL_REDUCTION_RATE
                        Channel reduction rate for the attention module
  --disable-skip-layers
                        Disable the skip layers
  --out-channels OUT_CHANNELS
                        Number of channels for the output image
  --evaluation-dirs EVALUATION_DIRS
                        Directory where the output files will be stored
  --eval-names EVAL_NAMES
                        Prefix used for the ourput filenames
  --infill {infill,test}
                        Infill the climate dataset ('test' if mask order is
                        irrelevant, 'infill' if mask order is relevant)
  --create-images CREATE_IMAGES
                        Creates .jpg images for time window specified using
                        the format 'YYY-MM-DD-HH:MM,YYYY-MM-DD-HH:MM'
  --create-video        Creates .gif videos using the created images
  --fps FPS             Frame per seconds for the created videos
  --create-report       Create a report with plots and evaluation metrics
  --eval-range EVAL_RANGE
                        Range of indexes for the time axis used to create the
                        plots in the report
  --eval-threshold EVAL_THRESHOLD
                        Create the masks from the climate dataset using a
                        threshold value
  --smoothing-factor SMOOTHING_FACTOR
                        Size of the convolution window used to smooth the
                        evaluation metrics
  --convert-to-netcdf   Convert the output hdf5 files to netCDF
  --partitions PARTITIONS
                        Split the climate dataset into several partitions
                        along the time coordinate
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
