
import xesmf as xe
import logging
import sys
import numpy as np
from .. import config as cfg

def check_ncformat(ds,data_type,image_size,basename):

    if not data_type in list(ds.keys()):
        logging.error('Variable name \'{}\' not found in file {}.'.format(data_type,basename))
        sys.exit()

    if not cfg.dataset_name is None:

        ds_dims = list(ds[data_type].dims)
        ndims = len(cfg.dataset_format["dimensions"])

        if ndims != len(ds_dims):
            logging.error('Inconsistent number of dimensions in file {}.\nThe number of dimensions should be: {}'.format(basename,ndims))
            sys.exit()

        for i in range(ndims):
            if cfg.dataset_format["dimensions"][i] != ds_dims[i]:
                logging.error('Inconsistent dimensions in file {}.\nThe list of dimensions should be: {}'.format(basename,cfg.dataset_format["dimensions"]))
                sys.exit()

        ds[data_type] = ds[data_type].transpose(*cfg.dataset_format["axes"])

        shape = ds[data_type].shape

        step = []
        regrid = False
        for i in range(2):
            coordinate = cfg.dataset_format["axes"][i+1]

            step.append(np.unique(np.gradient(ds[data_type][coordinate].values)))
            if len(step[i]) != 1:
                logging.error('The {} grid in file {} is not uniform.'.format(coordinate,basename))
                sys.exit()

            extent = cfg.dataset_format["grid"][i][1]-cfg.dataset_format["grid"][i][0]
            if abs( ds[data_type][coordinate].values[-1] - ds[data_type][coordinate].values[0] + step[i] - extent ) > 1e-2:
                logging.error('Incorrect {} extent in file {}.\nThe extent should be: {}'.format(coordinate,basename,extent))
                sys.exit()

            if shape[i+1] != image_size:
                step[i] *= shape[i+1]/image_size
                logging.warning('The length of {} does not correspond to the image size for file {}.'.format(coordinate,basename))
                regrid = True

        if regrid:
            logging.warning('The spatial coordinates have been interpolated using nearest_s2d in file {}.'.format(basename))
            grid = xr.Dataset({cfg.dataset_format["axes"][1]: ([cfg.dataset_format["axes"][1]], xe.util._grid_1d(*cfg.dataset_format["grid"][0][:2],step[0])[0]),
                              cfg.dataset_format["axes"][2]: ([cfg.dataset_format["axes"][2]], xe.util._grid_1d(*cfg.dataset_format["grid"][1][:2],step[1])[0])})
            ds = xe.Regridder(ds, grid, "nearest_s2d")(ds,keep_attrs=True)

        if ds[data_type].dtype != "float32":
            logging.warning('Incorrect data type for file {}.\nData type has been converted to float32.'.format(basename))
            ds[data_type] = ds[data_type].astype(dtype=np.float32)

    return ds