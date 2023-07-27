import torch
import torch.nn as nn
import xarray as xr
import numpy as np

radius_earth= 6371

def distance_on_sphere(lon1,lat1,lon2,lat2):
    d_lat = torch.abs(lat1-lat2)
    d_lon = torch.abs(lon1-lon2)
    asin = torch.sin(d_lat/2)**2

    d_rad = 2*torch.arcsin(torch.sqrt(asin + (1-asin-torch.sin((lat1+lat2)/2)**2)*torch.sin(d_lon/2)**2))
    return d_rad


def get_flat_coords(ds, coord_dict:dict):
    
    if 'lon' in coord_dict.values():
        lon = ds['lon'].values
        lat = ds['lat'].values
        lon,lat = np.meshgrid(np.deg2rad(lon),np.deg2rad(lat))
        lon = lon.flatten()
        lat = lat.flatten()
    else:
        lon = ds[coord_dict['lon']].values
        lat = ds[coord_dict['lat']].values

    return torch.tensor(lon),torch.tensor(lat)
    

def create_regular_grid(ds, coord_dict):
    
    if 'lon' in coord_dict.values() and 'lat' in coord_dict.values():
        n_grid_points = len(ds[coord_dict['lon']].values) * len(ds[coord_dict['lat']].values)
        deg=True
    else:
        deg=False
        n_grid_points = len(ds[coord_dict['lon']].values.flatten())
    lon_min, lon_max = ds[coord_dict['lon']].min().values,ds[coord_dict['lon']].max().values
    lat_min, lat_max = ds[coord_dict['lat']].min().values,ds[coord_dict['lat']].max().values

    R = (lon_max - lon_min)/(lat_max - lat_min)
    N_lon = np.sqrt(n_grid_points*R)
    N_lat = np.ceil(N_lon/R)

    if not deg:
        lon_min, lon_max = np.rad2deg(lon_min), np.rad2deg(lon_max)
        lat_min, lat_max = np.rad2deg(lat_min), np.rad2deg(lat_max)        

    lons = np.linspace((lon_min), (lon_max), N_lon.astype(int))
    lats = np.linspace((lat_min), (lat_max), N_lat.astype(int))

    Lons, Lats = np.meshgrid(lons, lats)

    lon_coord = xr.DataArray(lons, dims='lon', coords={'lon': lons})
    lat_coord = xr.DataArray(lats, dims='lat', coords={'lat': lats})

    var1 = xr.DataArray(Lons, dims=('lat', 'lon'), coords={'lat': lat_coord, 'lon': lon_coord})
    var2 = xr.DataArray(Lats, dims=('lat', 'lon'), coords={'lat': lat_coord, 'lon': lon_coord})

    # Create the xarray dataset
    reg_grid = xr.Dataset({coord_dict['lon'] + '_': var1,
                           coord_dict['lat'] + '_': var2})
    return reg_grid


def get_distance_matrix(ds1, ds2, coord_dict1: dict, coord_dict2:dict):
    #coord_dict = {'lon': 'var_name_lon','lat','var_name_lat'}"

    c1_lon, c1_lat = get_flat_coords(ds1, coord_dict1)
    c2_lon, c2_lat = get_flat_coords(ds2, coord_dict2)

    lat1 = c1_lat.reshape((1,(c1_lat).numel()))
    lon1 = c1_lon.reshape((1,(c1_lon).numel()))

    lat2 = c2_lat.reshape(((c2_lat).numel(),1))
    lon2 = c2_lon.reshape(((c2_lon).numel(),1))

    d = distance_on_sphere(lon1,lat1,lon2,lat2)
    
    coords = {'grid1':{'lat': lat1,'lon':lon1},'grid2':{'lat': lat2,'lon':lon2}}
    return d, coords


def get_coord_dict_from_var(ds, variable):
    dims = ds[variable].dims
    spatial_dim = dims[-1]

    if not 'lon' in spatial_dim and not 'lat' in spatial_dim:
        coords = list(ds[spatial_dim].coords.keys())
    else:
        coords = dims

    lon_c = [var for var in coords if 'lon' in var]
    lat_c = [var for var in coords if 'lat' in var]

    assert len(lon_c)==1, "no longitude variable was found"
    assert len(lat_c)==1, "no latitude variable was found"

    return {'lon':lon_c[0],'lat':lat_c[0]}



def get_neighbours_of_distmat(distance_matrix, neighbours=5, dim=0):

    #get indices for regular grid
    neighbours = torch.topk(distance_matrix, neighbours, dim=dim, largest=False)
    return neighbours


def get_neighbours_of_grids(ds1, ds2, variable=None, coord_dict1=None, coord_dict2=None, neighbours=5, dim=0):
    
    if variable is not None:
        coord_dict1 = get_coord_dict_from_var(ds1, variable)
        coord_dict2 = get_coord_dict_from_var(ds2, variable)

    dist_mat, coords = get_distance_matrix(ds1, ds2, coord_dict1, coord_dict2)

    #get indices for regular grid
    if dim!=-1:
        neighbours = get_neighbours_of_distmat(dist_mat, neighbours=neighbours, dim=dim)
    else:
        neighbours = tuple(get_neighbours_of_distmat(dist_mat, neighbours=neighbours, dim=0),
                           get_neighbours_of_distmat(dist_mat, neighbours=neighbours, dim=1))

    return neighbours


def get_regions(lons, lats, seeds_lon, seeds_lat, radius_region=None, n_points=None ,in_deg=True):


    if in_deg:
        seeds_lon = torch.deg2rad(seeds_lon)
        seeds_lat = torch.deg2rad(seeds_lat)

    lons = lons.view(len(lons),1)
    lats = lats.view(len(lats),1)

    seeds_lon = seeds_lon.view(1,len(seeds_lon))
    seeds_lat = seeds_lat.view(1,len(seeds_lat))

    d_mat = distance_on_sphere(lons, lats, seeds_lon, seeds_lat)

    if n_points is None:
        distances, indices = torch.topk(d_mat, k=d_mat.shape[0], dim=0, largest=False)
        distances *= radius_earth*2
        region_indices = distances.median(dim=1).values <= radius_region 

        distances_regions = distances[region_indices]
        indices_regions = indices[region_indices] 
    else:
        distances, indices = torch.topk(d_mat, k=n_points, dim=0, largest=False)
        indices_regions = indices
        distances *= radius_earth*2
        distances_regions = distances

    lons_regions = lons.view(-1)[indices_regions]
    lats_regions = lats.view(-1)[indices_regions]

    return distances_regions, indices_regions, lons_regions, lats_regions


class PositionCalculator(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, lons, lats, lons_t=None, lats_t=None):

        lons = lons.view(1,len(lons))
        lats = lats.view(1,len(lats))

        if lons_t is None:
            lons_t=lons.transpose(1,0)
            lats_t=lats.transpose(1,0)

            d_lons = distance_on_sphere(lons,lats,lons_t,lats_t.transpose(0,1))
            d_lats = distance_on_sphere(lons,lats,lons_t.transpose(0,1),lats_t)
        else:
            lons_t = lons_t.view(len(lons_t),1)
            lats_t = lats_t.view(len(lats_t),1)

        d_lats = distance_on_sphere(lons, lats, lons, lats_t)
        d_lons = distance_on_sphere(lons, lats, lons_t, lats)

        sgn = torch.sign(lats - lats_t)
        sgn[(lats - lats_t).abs()/torch.pi>1] = sgn[(lats - lats_t).abs()/torch.pi>1]*-1
        d_lats_s = d_lats*sgn

        sgn = torch.sign(lons - lons_t)
        sgn[(lons - lons_t).abs()/torch.pi>1] = sgn[(lons - lons_t).abs()/torch.pi>1]*-1
        d_lons_s = d_lons*sgn

        d_mat = torch.sqrt(d_lats**2+d_lons**2)

        phis = torch.atan2(d_lats_s,d_lons_s)

        return d_mat, phis, d_lons_s, d_lats_s
    


class random_region_generator():
    def __init__(self, lon_range, lat_range, lon_lr, lat_lr, lon_hr, lat_hr, radius_target, radius_factor, batch_size=1) -> None:
        self.lon_range = lon_range
        self.lat_range = lat_range
        self.lon_lr = lon_lr
        self.lat_lr = lat_lr
        self.lon_hr = lon_hr
        self.lat_hr = lat_hr

        self.radius_target = radius_target
        self.radius_factor = radius_factor

        self.batch_size=batch_size
        self.n_points_lr = None
        self.n_points_hr = None
    
    def generate(self):
        seeds_lon = torch.randint(self.lon_range[0],self.lon_range[1], size=(self.batch_size,1))
        seeds_lat = torch.randint(self.lat_range[0],self.lat_range[1], size=(self.batch_size,1))

  
        distances_regions, indices_regions, lons_regions, lats_regions = get_regions(self.lon_hr, self.lat_hr, seeds_lon, seeds_lat, self.radius_target, n_points=self.n_points_hr, in_deg=True)
        self.n_points_hr = len(distances_regions)

        out_dict_hr = {'distances': distances_regions,
                   'indices': indices_regions,
                   'lons': lons_regions,
                   'lats': lats_regions}
        
     
        distances_regions, indices_regions, lons_regions, lats_regions = get_regions(self.lon_lr, self.lat_lr, seeds_lon, seeds_lat, self.radius_target*self.radius_factor,n_points=self.n_points_lr, in_deg=True)
        self.n_points_lr = len(distances_regions)

        out_dict_lr = {'distances': distances_regions,
                   'indices': indices_regions,
                   'lons': lons_regions,
                   'lats': lats_regions}
    
        return {'hr':out_dict_hr,'lr': out_dict_lr, 'seeds': [torch.deg2rad(seeds_lon), torch.deg2rad(seeds_lat)]}