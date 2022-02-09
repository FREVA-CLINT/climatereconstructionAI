# Demo for climatereconstructionAI

The present demo aims at giving an example of the climate dataset infilling (**evaluation** process).

`climatereconstructionAI` must be installed first (see [README](https://github.com/FREVA-CLINT/climatereconstructionAI/tree/clint#readme))

## Structure

The `demo` folder contains:
- a directory `inputs` with the following sub-directories:
  - `test-large`: contains the climate datasets to be infilled:
    - medium dataset: `tas_hadcrut_185001-201812.nc` is a netCDF file containing the HadCRUT4 monthly global temperature anomaly (in ºC) from 1850 to 2018 (2028 time steps in total) with a spatial resolution of 2.5º×5º (lat×lon). The dataset can be downloaded from the UK MET office ([HadCRUT4](https://www.metoffice.gov.uk/hadobs/hadcrut4/data/current/download.html))
    - small dataset: `tas_hadcrut_187709_189308.nc` is a netCDF file containing the same HadCRUT4 data but for two dates only (September 1877 and August 1893)
  - `masks`: contains the masks used for the missing values (optional)
- two text files containing the input arguments of two examples:
    - `demo-1_args.txt` is used to infill the medium dataset
    - `demo-2_args.txt` is used to infill the small dataset
- a directory `outputs` where the output files will be stored
- a directory `images` containing some visualizations of the output files


## Usage

The paths for the input and output directories defined in `demo-*_args.txt` are relative to the `demo` directory. Hence, the software should be run in the current directory.

### CLI

```bash
# Use the medium dataset
crai-evaluate --load-from-file demo-1_args.txt

# Use the small dataset
crai-evaluate --load-from-file demo-2_args.txt
```

### Python module

```python
from climatereconstructionai import evaluate

# Use the medium dataset
evaluate("demo-1_args.txt")

# Use the small dataset
evaluate("demo-2_args.txt")
```

## Outputs

### The files

Each evaluation produces 5 netCDF files contained in the `output` folder:
- `demo-#_gt.nc` corresponds to the original dataset
- `demo-#_mask.nc` contains the masks corresponding to the missing values
- `demo-#_image.nc` is `demo-#_gt.nc` after applying the masks `demo-#_mask.nc`
- `demo-#_output.nc` is the infilled dataset (all values being infilled)
- `demo-#_output_comp.nc` is the **composite output dataset**: it is the original dataset `demo-#_gt.nc` where the missing values have been replaced by the values from `demo-#_output.nc`

### Visualization

We can visualize the infilling by comparing the original and the composite datasets for a specific date (here September 1877):

| Original dataset | Composite dataset |
| --------------------- | -------------------------- |
![Original dataset](images/demo-1_gt.png)  |  ![Composite dataset](images/demo-1_output_comp.png)
