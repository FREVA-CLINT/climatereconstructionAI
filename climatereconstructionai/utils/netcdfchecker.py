import logging

import numpy as np
import xarray as xr
import xesmf as xe

from .. import config as cfg


def reformat_dataset(ds1, ds2, data_type):
    if cfg.dataset_name is not None:
        ds2[data_type] = ds2[data_type].transpose(*cfg.dataset_format["dimensions"])

        regrid = False
        for i in range(2):
            coordinate = cfg.dataset_format["dimensions"][i + 1]
            if not ds1[coordinate].equals(ds2[coordinate]):
                regrid = True

        if regrid:
            ds2 = xe.Regridder(ds2, ds1, "nearest_s2d")(ds2, keep_attrs=True)
            del ds2.attrs["regrid_method"]

    return ds2


def dataset_formatter(ds, data_type, basename):
    if data_type not in list(ds.keys()):
        raise ValueError('Variable name \'{}\' not found in {}.'.format(data_type, basename))

    if cfg.dataset_name is not None:

        ds_dims = list(ds[data_type].dims)
        ndims = len(cfg.dataset_format["dimensions"])

        if ndims != len(ds_dims):
            raise ValueError(
                'Inconsistent number of dimensions in {}.\nThe number of dimensions should be: {}.'.format(basename,
                                                                                                           ndims))

        for i in range(ndims):
            if cfg.dataset_format["dimensions"][i] != ds_dims[i]:
                raise ValueError('Inconsistent dimensions in {}.''\nThe input file should contain: {}.'
                                 .format(basename, cfg.dataset_format["dimensions"]))

        ds[data_type] = ds[data_type].transpose(*cfg.dataset_format["axes"])

        step = []
        regrid = False
        for i in range(2):
            coordinate = cfg.dataset_format["axes"][i + 1]

            step.append(np.unique(np.gradient(ds[data_type][coordinate].values)))
            if len(step[i]) != 1:
                raise ValueError('The {} grid is not uniform in {}.'.format(coordinate, basename))

            extent = cfg.dataset_format["grid"][i][1] - cfg.dataset_format["grid"][i][0]
            diff = abs(ds[data_type][coordinate].values[-1] - ds[data_type][coordinate].values[0] + step[i] - extent)
            if diff > 1e-2:
                raise ValueError('Incorrect {} extent in {}.\nThe extent should be: {}.'
                                 .format(coordinate, basename, extent))

            if step[i] != cfg.dataset_format["step"][i]:
                step[i] = cfg.dataset_format["step"][i]
                logging.warning(
                    'The {} step does not match the targeted dataset in {}.'.format(coordinate, basename))
                regrid = True

        if regrid:
            logging.warning('The spatial coordinates have been interpolated using nearest_s2d in {}.'.format(basename))
            grid = xr.Dataset({cfg.dataset_format["axes"][1]: ([cfg.dataset_format["axes"][1]],
                                                               xe.util._grid_1d(*cfg.dataset_format["grid"][0][:2],
                                                                                step[0])[0]),
                               cfg.dataset_format["axes"][2]: ([cfg.dataset_format["axes"][2]],
                                                               xe.util._grid_1d(*cfg.dataset_format["grid"][1][:2],
                                                                                step[1])[0])})
            ds = xe.Regridder(ds, grid, "nearest_s2d")(ds, keep_attrs=True)

        if ds[data_type].dtype != "float32":
            logging.warning('Incorrect data type for {}.\nData type has been converted to float32.'.format(basename))
            ds[data_type] = ds[data_type].astype(dtype=np.float32)

    return ds
