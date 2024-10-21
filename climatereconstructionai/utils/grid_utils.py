import torch
import torch.nn as nn
import xarray as xr
import numpy as np
import math

from scipy.interpolate import griddata

radius_earth= 6371

def mapping_to_(dic, to='numpy', dtype='int'):
    out_dic = {}
    for key, subdic in dic.items():
        out_sub_dic = {}
        for subkey, subsub_dic in subdic.items():
            if to =='numpy':
                if dtype == "int":
                    dtype=np.int32
                elif dtype == "bool":
                    dtype=bool
                out_sub_dic[subkey] = np.array(subsub_dic, dtype=dtype)
            else:
                if dtype == "int":
                    dtype=torch.int32
                elif dtype == "bool":
                    dtype=torch.bool
                out_sub_dic[subkey] = torch.as_tensor(subsub_dic, dtype=dtype)
        out_dic[key] = out_sub_dic
    return out_dic

def distance_on_sphere(lon1,lat1,lon2,lat2):
    d_lat = torch.abs(lat1-lat2)
    d_lon = torch.abs(lon1-lon2)
    asin = torch.sin(d_lat/2)**2

    d_rad = 2*torch.arcsin(torch.sqrt(asin + (1-asin-torch.sin((lat1+lat2)/2)**2)*torch.sin(d_lon/2)**2))
    return d_rad

def get_coords_as_tensor(grid, lon='vlon', lat='vlat', grid_type=None):

    if grid_type == 'cell':
        lon, lat = 'clon', 'clat'

    elif grid_type == 'vertex':
        lon, lat = 'vlon', 'vlat'

    elif grid_type == 'edge':
        lon, lat = 'elon', 'elat'
    
    lons = torch.tensor(grid[lon].values)
    lats = torch.tensor(grid[lat].values)
    coords = torch.stack((lons, lats))
    return coords

def invert_dict(dict):
    dict_out = {}
    unique_values = np.unique(np.array(list(dict.values())))

    for uni_value in unique_values:
        dict_out[uni_value] = [key for key,value in dict.items() if value==uni_value]
    return dict_out

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

    if len(coords)==0:
        coords = list(ds[variable].coords.keys())
        
    lon_c = [var for var in coords if 'lon' in var]
    lat_c = [var for var in coords if 'lat' in var]
    
    if len(lon_c)==0:
        len_points = len(ds.coords[spatial_dim].values)
        dims_match = [dim for dim in ds.coords if len(ds[dim].values)==len_points]

        lon_c = [var for var in dims_match if 'lon' in var]
        lat_c = [var for var in dims_match if 'lat' in var]

    assert len(lon_c)==1, "no longitude variable was found"
    assert len(lat_c)==1, "no latitude variable was found"

    return {'lon':lon_c[0],'lat':lat_c[0], 'spatial_dim': spatial_dim}



def get_neighbours_of_distmat(distance_matrix, nh=5, dim=0):

    result = torch.topk(distance_matrix, nh, dim=dim, largest=False)
    return result


def get_neighbours_of_grids(ds1, ds2, variable=None, coord_dict1=None, coord_dict2=None, nh=None, avg_distance_km=None, dim=0):
    
    if variable is not None:
        coord_dict1 = get_coord_dict_from_var(ds1, variable)
        coord_dict2 = get_coord_dict_from_var(ds2, variable)

    dist_mat, coords = get_distance_matrix(ds1, ds2, coord_dict1, coord_dict2)

    if nh is not None:
        values, nh_indices = get_neighbours_of_distmat(dist_mat, nh=nh, dim=dim)

    elif avg_distance_km is not None:
        values, nh_indices = get_neighbours_of_distmat(dist_mat, nh=dist_mat.shape[dim], dim=dim)

        values *= radius_earth
        indices_in_dist = values.median(dim=1).values <= avg_distance_km 

        values = values[indices_in_dist]
        nh_indices = nh_indices[indices_in_dist] 
        
    return values, nh_indices


def prepare_coordinates_ds_dict(ds_dict, tags, variables, flatten=False, random_region=None):

    for tag in tags:
        ds = ds_dict[tag]['ds']

        ds_dict[tag]['spatial_dims'] = {}
        ds_dict[tag]['var_spatial_dims'] = {}

        spatial_dims = {}
        for variable in variables:
            
            coord_dict = get_coord_dict_from_var(ds, variable)

            lon = torch.tensor(ds[coord_dict['lon']].values) 
            lat = torch.tensor(ds[coord_dict['lat']].values) 
            
            lon = lon.deg2rad() if lon.max()>2*torch.pi else lon 
            lat = lat.deg2rad() if lat.max()>2*torch.pi else lat
            
            len_lon = len(lon) 
            len_lat = len(lat) 

            if flatten:
                lon = lon.flatten().repeat(len_lat) 
                lat = lat.view(-1,1).repeat(1,len_lon).flatten()
            
            spatial_coord_dict = {
                'coords':
                    {'lon': lon,
                    'lat': lat}
                        }
            
            ds_dict[tag]['spatial_dims'][coord_dict['spatial_dim']] = spatial_coord_dict
            spatial_dims[variable] = coord_dict['spatial_dim']

        ds_dict[tag]['var_spatial_dims'] = spatial_dims

    return ds_dict

def get_grid_relations(grids, coord_dict, regional=True, radius_regions_km=None, resolutions=None, min_overlap=3, radius_inc=100, save_file_path=None):
    #tbd: implement global

    if isinstance(grids[0], str):
        data = [xr.load_dataset(d) for d in grids]
    else:
        data = grids

    if radius_regions_km is None:
        resolutions = [d.mean_dual_edge_length for d in data]
        res_sort_idx = np.argsort(np.array(resolutions))[::-1]
        resolutions = [resolutions[ix] for ix in res_sort_idx]
        size_of_field = resolutions[0] * 1.5
        radius_regions_km = 2*size_of_field

    relations = {'indices':[],
                'rel_coords': {'children':[],'parents':[]}}

    region_coords = get_coords_as_tensor(data[0], lon=coord_dict['lon'], lat=coord_dict['lat'])

    for g_ix in range(1, len(data)):

        child_coords = get_coords_as_tensor(data[g_ix], lon=coord_dict['lon'], lat=coord_dict['lat'])

        relation_dict, radius_regions_km = get_parent_child_indices(region_coords, child_coords, radius_regions_km, radius_inc, min_overlap=min_overlap)

        relations['indices'].append(relation_dict)


    for g_ix in range(0, len(data)-1):

        relation = relations['indices'][g_ix]
        parent_coords = get_coords_as_tensor(data[g_ix], lon=coord_dict['lon'], lat=coord_dict['lat'])
        child_coords = get_coords_as_tensor(data[g_ix+1], lon=coord_dict['lon'], lat=coord_dict['lat'])

        rel_coords_children = get_relative_coordinates_grids(parent_coords, child_coords, relation, relative_to='parents')
        rel_coords_parents = get_relative_coordinates_grids(parent_coords, child_coords, relation, relative_to='children')

        relations['rel_coords']['children'].append(rel_coords_children)
        relations['rel_coords']['parents'].append(rel_coords_parents)
    
    grid_meta = [{'resolution:': resolutions[k]}.update(dict(data[k].attrs)) for k in range(len(resolutions))]
    grid_data = {
        'regional': True,
        'meta': grid_meta,
        'relations': relations}
    
    # maybe iobuffer here?
    if save_file_path is not None:
        torch.save(grid_data, save_file_path)
    
    return grid_data


def get_parent_child_indices(parent_coords: torch.tensor, child_coords: torch.tensor, radius_km: float, radius_inc_km: float, in_deg=False, min_overlap=1, max_overlap=5, batch_size=-1):

    n_iter = 0
    converged = False

    while not converged:

        pc_distance, p_c_indices, _, _ = get_regions(child_coords[0], child_coords[1], parent_coords[0], parent_coords[1], radius_region=radius_km, in_deg=in_deg, batch_size=batch_size)

        # relative to parents
        pc_distance = pc_distance.T
        p_c_indices = p_c_indices.T

        c_indices, counts = p_c_indices.unique(return_counts=True)
        
        if (len(c_indices) == len(child_coords[0])) and (counts.min() >= min_overlap):
            converged = True
            print(f'converged at iteration {n_iter} radius of {radius_km}km with min_overlap={counts.min()}, max_overlap={counts.max()}')
        else:
            radius_km = radius_km + radius_inc_km
            n_iter += 1

    max_overlap = int(counts.max()) if max_overlap is None else max_overlap

    # get parent indices for children
    c_p_distances = [[] for _ in range(len(child_coords[0]))]
    c_p_indices = [[] for _ in range(len(child_coords[0]))]
    c_p_indices_rel = [[] for _ in range(len(child_coords[0]))]
    p_indices_rel = [[] for _ in range(len(parent_coords[0]))]


    for p_index, c_indices in enumerate(p_c_indices):

        for c_index_rel, c_index in enumerate(list(c_indices)):
            p_indices_rel[p_index].append(len(c_p_indices[c_index]))
            c_p_indices[c_index].append(p_index)
            c_p_indices_rel[c_index].append(c_index_rel)
            c_p_distances[c_index].append(pc_distance[p_index, c_index_rel])
    
    p_indices_rel = torch.tensor(p_indices_rel)

    c_p_indices_pad = c_p_indices
    c_p_indices_rel_pad = c_p_indices_rel

    for c_index, p_indices in enumerate(c_p_indices):

        if len(p_indices) < max_overlap:
            p = 0
            m = len(p_indices)
        
            # sort parent regions by distance
            sort_ids = torch.tensor(c_p_distances[c_index]).sort().indices
            p_indices_repeat = torch.tensor(p_indices)[sort_ids]

            c_indices_rel = torch.tensor(c_p_indices_rel[c_index])[sort_ids]

            while len(c_p_indices_pad[c_index]) < max_overlap:
                c_p_indices_pad[c_index].append(int(p_indices_repeat[p]))
                c_p_indices_rel_pad[c_index].append(int(c_indices_rel[p]))
                p += 1
                p = p % m

        elif len(p_indices) > max_overlap:

            # sort parent regions by distance
            sort_ids = torch.tensor(c_p_distances[c_index]).sort().indices
            p_indices_repeat = torch.tensor(p_indices)[sort_ids]
            c_indices_rel = torch.tensor(c_p_indices_rel[c_index])[sort_ids]
            
            c_p_indices_pad[c_index] = p_indices_repeat[:max_overlap]
            c_p_indices_rel_pad[c_index] = c_indices_rel[:max_overlap]
        
        c_p_indices_rel_pad[c_index] = torch.tensor(c_p_indices_rel_pad[c_index]) if not torch.is_tensor(c_p_indices_rel_pad[c_index]) else c_p_indices_rel_pad[c_index]
        c_p_indices_pad[c_index] = torch.tensor(c_p_indices_pad[c_index]) if not torch.is_tensor(c_p_indices_pad[c_index]) else c_p_indices_pad[c_index]
    
    children_idx = torch.stack(c_p_indices_rel_pad)
    parents = torch.stack(c_p_indices_pad)

    indices = {'c_of_p': torch.stack((p_c_indices, p_indices_rel), dim=0),
               'p_of_c': torch.stack((parents, children_idx), dim=0)}
    
    return indices, radius_km


def get_relative_coordinates_grids(parent_coords, child_coords, relation_dict, relative_to="parents", batch_size=-1):

    if relative_to == 'children':
        coords_in_regions = [parent_coords[0][relation_dict['p_of_c'][0]], parent_coords[1][relation_dict['p_of_c'][0]]]
        region_center_coords = child_coords

    else:
        coords_in_regions = [child_coords[0][relation_dict['c_of_p'][0]], child_coords[1][relation_dict['c_of_p'][0]]]
        region_center_coords = parent_coords
    
    return get_relative_coordinates_regions(coords_in_regions, region_center_coords, batch_size=batch_size)


def get_relative_coordinates_regions(coords_in_regions, region_center_coords, batch_size=-1):

    PosCalc = PositionCalculator(batch_size)

    rel_coords = []

    for p in range(coords_in_regions[0].shape[0]):

        dist, phi, dlon, dlat = PosCalc(coords_in_regions[0][[p]].T, 
                                        coords_in_regions[1][[p]].T,
                                        (region_center_coords[0][p]).view(1,1),
                                        (region_center_coords[1][p]).view(1,1))
        
        rel_coords.append(torch.stack([dist, phi, dlon, dlat],axis=0))

    rel_coords = torch.concat(rel_coords, dim=1).permute(1,2,0)

    return rel_coords


def get_regions(lons, lats, seeds_lon, seeds_lat, radius_region=None, n_points=None ,in_deg=True, rect=False, batch_size=-1, return_rotated_coords=False):

    
    if in_deg:
        seeds_lon = torch.deg2rad(seeds_lon)
        seeds_lat = torch.deg2rad(seeds_lat)

    #if rotate_cs:
    #    lons, lats = rotate_coord_system(lons, lats, seeds_lon, seeds_lat)
    #    seeds_lon = seeds_lat = torch.tensor([0.]).view(-1,1)

    if rect:
        Pos_calc = PositionCalculator(batch_size=batch_size)
        d_mat, phi, dlon, dlat  = Pos_calc(lons, lats, seeds_lon, seeds_lat)
        region_indices_lon = dlon.median(dim=0).values.abs()*radius_earth <= radius_region 
        region_indices_lat = dlat.median(dim=0).values.abs()*radius_earth <= radius_region 

        region_indices = torch.logical_and(region_indices_lon, region_indices_lat).view(1,-1)

        distances_regions = d_mat[region_indices]
        indices_regions = region_indices.flatten().argwhere()

        idx_sort = distances_regions.sort().indices
        distances_regions = distances_regions[idx_sort]
        indices_regions = indices_regions[idx_sort]

    else:
        Pos_calc = PositionCalculator(batch_size=batch_size)
   
        d_mat, _, d_lons, d_lats = Pos_calc(lons, lats, seeds_lon, seeds_lat)

        d_mat = d_mat.T

        if return_rotated_coords:
            lons, lats = d_lons, d_lats

        if batch_size != -1 and batch_size <= d_mat.shape[-1]:
            n_chunks = torch.tensor(d_mat.shape).max() // batch_size 
        else:
            n_chunks = 1

        d_mat = d_mat.chunk(n_chunks, dim=-1)

        distances_regions = []
        indices_regions = []
        idx=0
        for d_mat_chunk in d_mat:
            if n_points is None:
                distances, indices = torch.topk(d_mat_chunk, k=d_mat_chunk.shape[0], dim=0, largest=False)
                distances *= radius_earth
                region_indices = distances.median(dim=1).values <= radius_region 

                distances = distances[region_indices]
                indices = indices[region_indices] 
                n_points = indices.shape[0]
            else:     
                distances, indices = torch.topk(d_mat_chunk, k=n_points, dim=0, largest=False)
                distances *= radius_earth

            distances_regions.append(distances)
            indices_regions.append(indices)
            idx+=0

        distances_regions = torch.concat(distances_regions, dim=-1)
        indices_regions = torch.concat(indices_regions, dim=-1)

    lons_regions = lons.view(-1)[indices_regions]
    lats_regions = lats.view(-1)[indices_regions]

    return distances_regions, indices_regions, lons_regions, lats_regions

def v_spherical_to_cart(u: torch.tensor, v: torch.tensor, lon: torch.tensor, lat: torch.tensor):


    vx = (torch.cos(lat) * torch.sin(lon)).view(1,-1)*u - (torch.sin(lat) * torch.cos(lon)).view(1,-1)*v
    vy = (torch.sin(lat) * torch.sin(lon)).view(1,-1)*v - (torch.cos(lat) * torch.cos(lon)).view(1,-1)*u
    vz = -(torch.cos(lat)).view(1,-1)*v

    return vx,vy,vz


def rotate_coord_system(lons: torch.tensor, lats: torch.tensor, rotation_lon: torch.tensor, rotation_lat: torch.tensor):

    theta = torch.tensor(rotation_lat) if not torch.is_tensor(rotation_lat) else rotation_lat
    phi = torch.tensor(rotation_lon) if not torch.is_tensor(rotation_lon) else rotation_lon

    theta = theta.view(-1,1)
    phi = phi.view(-1,1)

    x = (torch.cos(lons) * torch.cos(lats)).view(1,-1)
    y = (torch.sin(lons) * torch.cos(lats)).view(1,-1)
    z = (torch.sin(lats)).view(1,-1)

    rotated_x =  torch.cos(theta)*torch.cos(phi) * x + torch.cos(theta)*torch.sin(phi)*y + torch.sin(theta)*z
    rotated_y = -torch.sin(phi)*x + torch.cos(phi)*y
    rotated_z = -torch.sin(theta)*torch.cos(phi)*x - torch.sin(theta)*torch.sin(phi)*y + torch.cos(theta)*z

    rot_lon = torch.atan2(rotated_y, rotated_x)
    rot_lat = torch.arcsin(rotated_z)

    return rot_lon, rot_lat

class PositionCalculator(nn.Module):
    def __init__(self, batch_size=-1):
        super().__init__()
        self.batch_size = batch_size

    def forward(self, lons, lats, lons_t, lats_t):

        lons = lons.view(1,(lons).numel())
        lats = lats.view(1,(lats).numel())
       
        if self.batch_size != -1 and self.batch_size <= lons.shape[-1]:
            n_chunks = lons.shape[-1] // self.batch_size 
        else:
            n_chunks = 1
        
        lons_chunks = lons.chunk(n_chunks, dim=-1)
        lats_chunks = lats.chunk(n_chunks, dim=-1)

        lons_t = lons_t.view(len(lons_t),1)
        lats_t = lats_t.view(len(lats_t),1)

        d_lons = []
        d_lats = []
        d_mat = []
        phis = []

        lons_lats_rot = torch.zeros_like(lats_t)

        for chunk_idx in range(len(lons_chunks)):
            lons_chunk = lons_chunks[chunk_idx]
            lats_chunk = lats_chunks[chunk_idx]
            lons_chunk, lats_chunk = rotate_coord_system(lons_chunk, lats_chunk, lons_t, lats_t)

            d_lats_chunk = distance_on_sphere(lons_chunk, lats_chunk, lons_chunk, lons_lats_rot)
            d_lons_chunk = distance_on_sphere(lons_chunk, lats_chunk, lons_lats_rot, lats_chunk)

            sgn = torch.sign(lats_chunk)
            sgn[(lats_chunk ).abs()/torch.pi>1] = sgn[(lats_chunk).abs()/torch.pi>1]*-1
            d_lats_s = d_lats_chunk*sgn

            sgn = torch.sign(lons_chunk )
            sgn[(lons_chunk).abs()/torch.pi>1] = sgn[(lons_chunk).abs()/torch.pi>1]*-1
            d_lons_s = d_lons_chunk*sgn

            d_lons.append(d_lons_s)
            d_lats.append(d_lats_s)

            d_mat.append(torch.sqrt(d_lats_chunk**2+d_lons_chunk**2))
            phis.append(torch.atan2(d_lats_s,d_lons_s))

        d_lons = torch.concat(d_lons, dim=-1)
        d_lats = torch.concat(d_lats, dim=-1)
        d_mat = torch.concat(d_mat, dim=-1)
        phis = torch.concat(phis, dim=-1)

        return d_mat, phis, d_lons, d_lats
    
    


class random_region_generator():
    def __init__(self, lon_range, lat_range, lon_lr, lat_lr, lon_hr, lat_hr, radius_factor, radius_target=None, n_points_hr=None, batch_size=1) -> None:
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
        self.n_points_hr = n_points_hr
    
    def generate(self):
        seeds_lon = torch.randint(self.lon_range[0],self.lon_range[1], size=(self.batch_size,1))
        seeds_lat = torch.randint(self.lat_range[0],self.lat_range[1], size=(self.batch_size,1))

  
        distances_regions, indices_regions, lons_regions, lats_regions = get_regions(self.lon_hr, self.lat_hr, seeds_lon, seeds_lat, radius_region=self.radius_target, n_points=self.n_points_hr, in_deg=True)
        self.n_points_hr = len(distances_regions)

        out_dict_hr = {'distances': distances_regions,
                   'indices': indices_regions,
                   'lons': lons_regions,
                   'lats': lats_regions}
        
        if self.radius_target is None:
            radius_lr = (distances_regions.max()/2)*self.radius_factor
        else:
            radius_lr = self.radius_target*self.radius_factor

        distances_regions, indices_regions, lons_regions, lats_regions = get_regions(self.lon_lr, self.lat_lr, seeds_lon, seeds_lat, radius_region=radius_lr,n_points=self.n_points_lr, in_deg=True)
        self.n_points_lr = len(distances_regions)

        out_dict_lr = {'distances': distances_regions,
                   'indices': indices_regions,
                   'lons': lons_regions,
                   'lats': lats_regions}
    
        return {'hr':out_dict_hr,'lr': out_dict_lr, 'seeds': [torch.deg2rad(seeds_lon), torch.deg2rad(seeds_lat)]}
    

class random_region_generator_multi():
    def __init__(self, tags, lon_range, lat_range, lons, lats, n_points_global, factor=200, batch_size=1, init_radius=500) -> None:
        self.lon_range = lon_range
        self.lat_range = lat_range
        self.lons = lons
        self.lats = lats

        #self.tags = tags
        self.tag_dict = dict(zip(tags, np.arange(len(lons))))

        self.batch_size=batch_size
        
        self.n_points = [None] * len(lons)

        self.n_points_global = n_points_global

        self.radii = [init_radius] * len(lons)

        self.max_points = torch.tensor([len(lon) for lon in lons]).max()
        


    def generate(self, tags=None, indices=None, seeds=[]):

        if tags is not None:
            indices = [self.tag_dict[key] for key in tags]

        if len(seeds)==0:
            seeds_lon = torch.randint(self.lon_range[0],self.lon_range[1], size=(self.batch_size,1))
            seeds_lat = torch.randint(self.lat_range[0],self.lat_range[1], size=(self.batch_size,1))

        k= 0
        regions = []
        for k, index in enumerate(indices):
            
            n_points = self.n_points_global

            if n_points is not None:
                n_points = self.max_points if n_points > self.max_points else n_points

            distances_regions, indices_regions, lons_regions, lats_regions = get_regions(self.lons[index], self.lats[index], seeds_lon, seeds_lat, radius_region=self.radii[index], n_points=n_points, in_deg=True)
            self.n_points[index] = len(distances_regions)
            self.radii[index] = (distances_regions.max()/2)

            out_dict = {'distances': distances_regions,
                      'indices': indices_regions,
                      'lons': lons_regions,
                     'lats': lats_regions}

            regions.append(out_dict)

        self.grid_point_density = torch.tensor(self.n_points)/torch.tensor(self.radii)

        return {'regions':regions, 'seeds': [seeds_lon, seeds_lat]}
    
def rotate_ds(ds, angle_rad):
    
    for var_name in ds.variables:
        if 'lon' in var_name:
            ds[var_name].values = np.mod((ds[var_name].values + angle_rad)+np.pi, 2*np.pi)-np.pi

    return ds

def generate_region(coords, range_lon=None, range_lat=None, n_points=None, radius=None, locations=[], batch_size=1, rect=False, return_rotated_coords=False):

    if len(locations)==0:
        seeds_lon = (range_lon[1] - range_lon[0]) * torch.rand(size=(batch_size,1))+range_lon[0]
        seeds_lat = (range_lat[1] - range_lat[0]) * torch.rand(size=(batch_size,1))+range_lat[0]
    else:
        seeds_lon = locations[0]
        seeds_lat = locations[1]

    if radius > -1:
        distances_regions, indices_regions, lons_regions, lats_regions = get_regions(coords['lon'], coords['lat'], seeds_lon, seeds_lat, radius_region=radius, n_points=n_points, in_deg=True, rect=rect, return_rotated_coords=return_rotated_coords)
        n_points = len(distances_regions)
        radius = distances_regions.max()
    
    else:
        lons_regions, lats_regions = rotate_coord_system(coords['lon'], coords['lat'], seeds_lon, seeds_lat)
        lons_regions = lons_regions.T
        lats_regions = lats_regions.T
        n_points = lons_regions.shape[0]
        indices_regions = torch.arange((n_points), dtype=int)
        radius = -1
        distances_regions = 0

    out_dict = {'distances': distances_regions,
                'indices': indices_regions,
                'lon': lons_regions,
                'lat': lats_regions,
                "n_points": n_points,
                'radius': radius,
                'locations': [seeds_lon, seeds_lat]}


    return out_dict


def get_patches(grid_spacing_equator_km, pix_size, overlap):

    total_grid_size_lon_pix = (2 * math.pi * radius_earth)/grid_spacing_equator_km
    total_grid_size_lon_pix = 2**np.ceil(np.log2(total_grid_size_lon_pix))


    n_patches_lon = total_grid_size_lon_pix/pix_size

    border_patches_lon = np.linspace(-math.pi, math.pi,int(n_patches_lon)+1)
    border_patches_lat = np.linspace(-math.pi/2, math.pi/2,int(n_patches_lon/2)+1)

    centers_lon = (border_patches_lon[1:] + border_patches_lon[:-1])/2
    centers_lat = (border_patches_lat[1:] + border_patches_lat[:-1])/2

    border_patches_lon = np.stack((border_patches_lon[:-1],border_patches_lon[1:]), axis=1)
    border_patches_lat = np.stack((border_patches_lat[:-1],border_patches_lat[1:]), axis=1)

    overlap_grad = overlap * (border_patches_lon[0,1]- border_patches_lon[0,0])
    border_patches_lon = border_patches_lon + np.array([-overlap_grad, overlap_grad])

    overlap_grad = overlap * (border_patches_lat[0,1]- border_patches_lat[0,0])
    border_patches_lat = border_patches_lat + np.array([-overlap_grad, overlap_grad])

    patches = {
        'centers_lon':centers_lon, 
        'centers_lat':centers_lat,
        'borders_lon': border_patches_lon,
        'borders_lat': border_patches_lat
    }

    return patches


def get_ids_in_patches(patches, coords, return_torch=True):

    centers_lon = patches['centers_lon']
    centers_lat = patches['centers_lat']
    border_patches_lon = patches['borders_lon']
    border_patches_lat = patches['borders_lat']

    ids_in_patches = []
    patch_ids_lon = []
    patch_ids_lat = []
    for patch_lon in range(len(centers_lon)):
        for patch_lat in range(len(centers_lat)):
            border_patch_lon = border_patches_lon[patch_lon]
            border_patch_lat = border_patches_lat[patch_lat]
            patch_ids_lon.append(patch_lon)
            patch_ids_lat.append(patch_lat)

            in_patch_lon = np.logical_and(coords[0] >= border_patch_lon[0], coords[0] < border_patch_lon[1])

            if np.round(border_patch_lon[0],6) < np.round(math.pi,6):
                in_patch_lon = np.logical_or(in_patch_lon, (coords[0] >= (2*math.pi + border_patch_lon[0])))

            elif np.round(border_patch_lon[1],6) > np.round(math.pi,6):
                in_patch_lon = np.logical_or(in_patch_lon, (coords[0] <= (border_patch_lon[1] - 2*math.pi)))

            in_patch_lat = np.logical_and(coords[1] >= border_patch_lat[0], coords[1] < border_patch_lat[1])
            
            ids = np.where(np.logical_and(in_patch_lon, in_patch_lat))[0]

            if return_torch:
                ids = torch.tensor(ids)

            ids_in_patches.append(ids)

    patch_ids = {'lon': patch_ids_lon,
                 'lat': patch_ids_lat}

    
    return ids_in_patches, patch_ids

class grid_interpolator(nn.Module):
    def __init__(self, x_grid, y_grid, method='nearest', polar_shift=True):
        super().__init__()

        self.method = method

        if len(x_grid.shape)<2:
            self.x_grid, self.y_grid = np.meshgrid(x_grid, y_grid)
        else:
            self.x_grid, self.y_grid = x_grid, y_grid

    def forward(self, data, coords):
        device = data.device
        data = data.cpu()
        x,y = coords.cpu()

        if self.method=='nearest':
            grid_z = griddata((x, y), data, (self.x_grid, self.y_grid), method=self.method)
        else:
            grid_z = griddata((x, y), data, (self.x_grid, self.y_grid), method=self.method)
            grid_z_nn = griddata((x, y), data, (self.x_grid, self.y_grid), method='nearest')
            grid_z[np.isnan(grid_z)] = grid_z_nn[np.isnan(grid_z)]

        return torch.tensor(grid_z, device=device).float()
    


def get_distance_angle(lon1, lat1, lon2, lat2, base="polar", periodic_fov=None):


    d_lons =  2*torch.arcsin(torch.cos(lat1)*torch.sin(torch.abs(lon2-lon1)/2))

    d_lats = (lat2-lat1).abs() 


    #d_lons = distance_on_sphere(lon1, lat1, lon2, lat1)
    #d_lats = distance_on_sphere(lon1, lat1, lon1, lat2)
    

    sgn = torch.sign(lat2-lat1)
    sgn[(d_lats).abs()/torch.pi>1] = sgn[(d_lats).abs()/torch.pi>1]*-1
    d_lats = d_lats*sgn

    sgn = torch.sign(lon2-lon1)
    sgn[(d_lons).abs()/torch.pi>1] = sgn[(d_lons).abs()/torch.pi>1]*-1
    d_lons = d_lons*sgn


    if periodic_fov is not None:
        rng_lon = (periodic_fov[1] - periodic_fov[0])

        d_lons[d_lons > rng_lon] = d_lons[d_lons > rng_lon] - rng_lon
        d_lons[d_lons < -rng_lon] = d_lons[d_lons < -rng_lon] + rng_lon


    if base == "polar":
        distance = torch.sqrt(d_lats**2 + d_lons**2)
        phi = torch.atan2(d_lats, d_lons)

        return distance.float(), phi.float()

    else:
        return d_lons.float(), d_lats.float()
    


def get_adjacent_indices(acoe, eoc, nh=5, global_level=1):
    b = eoc.shape[-1]
    global_indices = torch.arange(b)

    nh1 = acoe.T[eoc.T].reshape(-1,4**global_level,6**1)
    self_indices = global_indices.view(-1,4**global_level)[:,0]
    self_indices = self_indices // 4**global_level

    adjc_indices = nh1.view(nh1.shape[0],-1) // 4**global_level

    adjc_unique = (adjc_indices).long().unique(dim=-1)

    is_self = adjc_unique - self_indices.view(-1,1) == 0

    adjc = adjc_unique[~is_self]

    adjc = adjc.reshape(self_indices.shape[0], -1)

    adjcs = [self_indices.view(-1,1), adjc]

    duplicates = [torch.zeros_like(adjcs[0], dtype=torch.bool), torch.zeros_like(adjcs[1], dtype=torch.bool)]
    
    b = adjc.shape[0]
    for k in range(1, nh):
        adjc_prev = adjcs[-1]

        adjc = adjcs[1][adjc_prev,:].view(b,-1)

        check_indices = torch.concat(adjcs, dim=-1).unsqueeze(dim=-2)

        # identify entities from previous nhs
        is_prev = adjc.unsqueeze(dim=-1) - check_indices == 0
        is_prev = is_prev.sum(dim=-1) > 0

        # remove duplicates from previous nhs
        
        is_removed = is_prev

        is_removed_count = is_removed.sum(dim=-1)
        
        unique, counts = is_removed_count.unique(return_counts=True)
        majority = unique[counts.argmax()]
        
        for minority in unique[unique!=majority]:

            where_minority = torch.where(is_removed_count==minority)[0]

            ind0, ind1 = torch.where(is_removed[where_minority])

            ind0 = ind0.reshape(len(where_minority),-1)[:,:minority-majority].reshape(-1)
            ind1 = ind1.reshape(len(where_minority),-1)[:,:minority-majority].reshape(-1)

            is_removed[where_minority[ind0], ind1] = False

        adjc = adjc[~is_removed]

        adjc = adjc.reshape(b, -1)
        
        if k > 1:
            counts = [] 
            uniques=[]
            for row in adjc:
                unique, count = row.unique(return_counts=True)
                uniques.append(unique)
                counts.append(len(unique))
        
            adjc = torch.nn.utils.rnn.pad_sequence(uniques, batch_first=True, padding_value=-1)
            duplicates_mask = adjc==-1
        else:
            duplicates_mask = torch.zeros_like(adjc)
            
        adjcs.append(adjc)
        duplicates.append(duplicates_mask)

    adjc = torch.concat(adjcs, dim=-1)
    duplicates = torch.concat(duplicates, dim=-1)

    return adjc, duplicates



def get_nearest_to_icon_rec(c_t_global, c_i, level=7, global_indices_i=None, nh=5, search_radius=5, reverse=False, periodic_fov=None):

    n_coords, n_sec_i, n_pts_i = c_i.shape
    n_target = c_t_global.shape[-1]

    id_t = torch.arange(n_target)

    n_level = n_target // 4**level

    if level > 0:
        mid_points_corners = id_t.reshape(-1, 4, 4**(level-1))[:,1:,0]
        mid_points = id_t.reshape(-1, 4, 4**(level-1))[:,[0],0]
    else:
        mid_points_corners = id_t.reshape(-1,4)[:,1:].repeat_interleave(4, dim=0)
        mid_points = id_t.unsqueeze(dim=-1)

    # get radius
    c_t_ = c_t_global[:,mid_points_corners]
    c_t_m = c_t_global[:,mid_points]

    dist_corners = get_distance_angle(c_t_[0].unsqueeze(dim=-1),c_t_[1].unsqueeze(dim=-1), c_t_m[0].unsqueeze(dim=-2),c_t_m[1].unsqueeze(dim=-2))[0]
    dist_corners_max = search_radius*dist_corners.max(dim=-1).values.max()

    c_i_ = c_i

    c_t_m = c_t_m.reshape(2, n_sec_i, -1)

    dist, phi = get_distance_angle(c_t_m[0].unsqueeze(dim=-1),c_t_m[1].unsqueeze(dim=-1), c_i_[0].unsqueeze(dim=-2), c_i_[1].unsqueeze(dim=-2), periodic_fov=periodic_fov)
    dist = dist.reshape(n_level, -1)
    phi = phi.reshape(n_level, -1)

    in_rad = dist <= dist_corners_max

    dist_values, indices_rel = dist.topk(in_rad.sum(dim=-1).max(), dim=-1, largest=False, sorted=True)
    

    if global_indices_i is None:
        global_indices = indices_rel
    else:
        global_indices = torch.gather(global_indices_i, index=indices_rel.reshape(global_indices_i.shape[0],-1), dim=-1)
        global_indices = global_indices.reshape(n_level,-1)
    
    if nh is not None:
        n_keep = nh
    else:
        n_keep = dist_values.shape[1]

    indices_keep = dist_values.topk(int(n_keep), dim=-1, largest=False, sorted=True)[1]

    dist_values = torch.gather(dist_values, index=indices_keep, dim=-1)
    in_rad = torch.gather(in_rad.reshape(n_level, -1), index=indices_keep, dim=-1)
    global_indices = torch.gather(global_indices, index=indices_keep, dim=-1)

    phi_values = torch.gather(phi, index=indices_keep, dim=-1)

    return global_indices, in_rad, (dist_values, phi_values)


def get_mapping_to_icon_grid(coords_icon, coords_input, search_raadius=3, max_nh=10, lowest_level=0, reverse_last=False, periodic_fov=None):

    level_start = int(math.log(coords_icon.shape[-1])/math.log(4))
    
    r = coords_icon.shape[-1]/4**level_start

    while math.floor(r)!=math.ceil(r):
        level_start -= 1
        r = coords_icon.shape[-1]/4**level_start

    grid_mapping = []
    for k in range(level_start + 1 - lowest_level):
        level = level_start - (k)

        nh = max_nh*4**level

        if level == lowest_level:
            nh = max_nh
        else:
            nh = None

        if k == 0:
            indices, in_rng, pos = get_nearest_to_icon_rec(coords_icon, coords_input.unsqueeze(dim=1), level=level, nh=nh, search_radius=search_raadius, periodic_fov=periodic_fov)
        else:
            indices, in_rng, pos = get_nearest_to_icon_rec(coords_icon, coords_input[:,indices], level=level, global_indices_i=indices, nh=nh, search_radius=search_raadius, periodic_fov=periodic_fov)

        if k == level_start - lowest_level and reverse_last:
            indices = np.array(indices.transpose(0,1).reshape(-1))
            uni, indices_rev = np.unique(indices, return_index=True)
            indices_rev = indices_rev % coords_icon.shape[-1]

            while len(uni) < coords_input.shape[-1]:
      
                indices_g = np.arange(coords_input.shape[-1])
                indices_g[uni]=-1
                indices_missing = torch.tensor(indices_g[indices_g!=-1])

                grid_mapping = get_mapping_to_icon_grid(coords_icon, coords_input[:,indices_missing], level_start=level_start, max_nh=max_nh, search_raadius=search_raadius, reverse_last=False, periodic_fov=periodic_fov)

                indices_new = np.array(grid_mapping[-1]['indices'].transpose(0,1).reshape(-1))
                uni_new, indices_rev_new = np.unique(indices_new, return_index=True)
                indices_rev_new = indices_rev_new % coords_icon.shape[-1]
               
                uni_new_g = indices_missing[uni_new]
                uni = np.concatenate((uni, uni_new_g), axis=0)
                indices_rev = np.concatenate((indices_rev, indices_rev_new), axis=0)
            
            indices = torch.tensor(indices_rev).unsqueeze(dim=-1)

        grid_mapping.append({'level': level, 'indices': indices, 'pos': pos, 'in_rng_mask': in_rng}) 

    return grid_mapping


def get_nh_variable_mapping_icon(grid_file_icon, grid_types_icon, grid_file, grid_types, search_raadius=3, max_nh=10, lowest_level = 0, return_last=True, reverse_last=False, coords_icon=None, scale_input=1., periodic_fov=None):
    
    grid_icon = xr.open_dataset(grid_file_icon)
    grid = xr.open_dataset(grid_file)

    lookup = {'edge': {'cell': 'adjacent_cell_of_edge',
                       'vertex': 'edge_vertices',
                       'edge': 'edge_index'},

              'cell': {'edge': 'edge_of_cell',
                       'vertex': 'vertex_of_cell',
                       'cell': 'cell_index'},

               'vertex': {'edge': 'edges_of_vertex',
                       'cell': 'cells_of_vertex',
                        'vertex': 'vertex_index'}}
    
    mapping_icon = {}
    in_range = {}
    for grid_type_icon in grid_types_icon:
        
        if coords_icon is None:
            coords_icon = get_coords_as_tensor(grid_icon, grid_type=grid_type_icon)

        mapping_grid_type = {}
        in_range_grid_type = {}

        for grid_type in grid_types:
            coords_input = get_coords_as_tensor(grid, grid_type=grid_type)
            if scale_input > 1:
                coords_input = scale_coordinates(coords_input, scale_input)

            if grid_file_icon == grid_file:
                indices = torch.arange(coords_icon.shape[1]).view(-1,1)
                in_rng_mask = torch.ones_like(indices, dtype=torch.bool)

            else:
                mapping = get_mapping_to_icon_grid(
                    coords_icon,
                    coords_input,
                    search_raadius=search_raadius,
                    max_nh=max_nh,
                    lowest_level=lowest_level,
                    reverse_last=reverse_last,
                    periodic_fov=periodic_fov)
                
                if return_last:
                    mapping = mapping[-1]
                
                indices = mapping['indices']
                in_rng_mask = mapping['in_rng_mask']

            mapping_grid_type[grid_type] = indices
            in_range_grid_type[grid_type] = in_rng_mask

        mapping_icon[grid_type_icon] = mapping_grid_type
        in_range[grid_type_icon] = in_range_grid_type

    return mapping_icon, in_range


def icon_grid_to_mgrid(grid, n_grid_levels, clon_fov=None, clat_fov=None, nh=0, extension=0.1):

    clon =  torch.tensor(grid.clon.values)
    clat =  torch.tensor(grid.clat.values)

    eoc = torch.tensor(grid.edge_of_cell.values - 1)
    acoe = torch.tensor(grid.adjacent_cell_of_edge.values - 1)

    global_indices = torch.arange(len(grid.clon))
    cell_coords_global = get_coords_as_tensor(grid, lon='clon', lat='clat').double()
    #coordinates per level


    if clon_fov is not None or clat_fov is not None:

        indices_max_lvl = torch.arange(len(grid.clon)).reshape(-1,4**int(n_grid_levels))
       
        clon_max_lvl = clon[indices_max_lvl[:,0]]
        clat_max_lvl = clat[indices_max_lvl[:,0]]

        keep_indices_clon = torch.ones_like(clon_max_lvl, dtype=bool)
        keep_indices_clat = torch.ones_like(clat_max_lvl, dtype=bool)

        if clon_fov is not None:
            fov_ext = (clon_fov[1] - clon_fov[0])*extension/2
            keep_indices_clon = torch.logical_and(clon_max_lvl >= clon_fov[0]-fov_ext, clon_max_lvl <= clon_fov[1]+fov_ext)

        if clat_fov is not None:   
            fov_ext = (clat_fov[1] - clat_fov[0])*extension/2
            keep_indices_clat = torch.logical_and(clat_max_lvl >= clat_fov[0]-fov_ext, clat_max_lvl <= clat_fov[1]+fov_ext)

        keep_grid_indices_max_lvl = torch.logical_and(keep_indices_clon, keep_indices_clat)

        keep_grid_indices = keep_grid_indices_max_lvl.view(-1,1).repeat_interleave(indices_max_lvl.shape[-1], dim=1)

    else:
        keep_grid_indices = torch.ones_like(global_indices, dtype=bool)


    global_levels = []
    grids = []
    for grid_level_idx in range(n_grid_levels):

        global_level = grid_level_idx
        global_levels.append(global_level)

        adjc, adjc_mask_duplicates = get_adjacent_indices(acoe, eoc, nh=nh, global_level=global_level)
        adjc_mask_duplicates = adjc_mask_duplicates==False

        keep_grid_indices_lvl = keep_grid_indices.reshape(-1,4**int(global_level))[:,0]

        indices_lvl = global_indices.reshape(-1,4**int(global_level))[:,0]

        indices_lvl = indices_lvl[keep_grid_indices_lvl]

        adjc_lvl = adjc[keep_grid_indices_lvl]
        adjc_mask_duplicates = adjc_mask_duplicates[keep_grid_indices_lvl]
        
        indices_in_fov = torch.where(keep_grid_indices_lvl)[0]
        index_shift = torch.concat((torch.tensor(0).view(1),indices_in_fov)).diff()
        wh = torch.where(index_shift>1)[0]

        for l, k in enumerate(wh):
            idx_shift = index_shift[k] if k==0 else index_shift[k]-1
            adjc_lvl[adjc_lvl >= adjc_lvl[k,0]] = adjc_lvl[adjc_lvl >= adjc_lvl[k,0]] - idx_shift

        cell_coords_lvl = cell_coords_global.reshape(2,-1,4**global_level)[:,:,0]

        if clon_fov is not None:
            fov_ext = (clon_fov[1] - clon_fov[0])*extension/2
            adjc_mask_lon = torch.logical_and(cell_coords_lvl[0, adjc[keep_grid_indices_lvl]] >= clon_fov[0]-fov_ext, cell_coords_lvl[0, adjc[keep_grid_indices_lvl]] <= clon_fov[1]+fov_ext)

            fov_ext = (clat_fov[1] - clat_fov[0])*extension/2
            adjc_mask_lat = torch.logical_and(cell_coords_lvl[1, adjc[keep_grid_indices_lvl]] >= clat_fov[0]-fov_ext, cell_coords_lvl[1, adjc[keep_grid_indices_lvl]] <= clat_fov[1]+fov_ext)

            adjc_mask = torch.logical_and(adjc_mask_lon, adjc_mask_lat)
            adjc_mask = torch.logical_and(adjc_mask, adjc_mask_duplicates)
        
        else:
            adjc_mask = adjc_mask_duplicates
        
        adjc_mask[adjc_lvl>(adjc_lvl.shape[0]-1)]=False

        r,c = torch.where(adjc_mask==False)
        adjc_lvl[r,c] = adjc_lvl[r,0] 

        grid_lvl = {
            'coords': cell_coords_global[:,indices_lvl],
            'adjc': adjc,
            'adjc_lvl': adjc_lvl,
            'adjc_mask': adjc_mask,
            "global_level": global_level
        }

        grids.append(grid_lvl)

    return grids

def scale_coordinates(coords, scale_factor):
    m = coords.mean(dim=1, keepdim=True)
    return (coords - m) * scale_factor + m