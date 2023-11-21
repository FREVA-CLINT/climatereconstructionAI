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

def get_coords_as_tensor(grid, lon='vlon', lat='vlat'):
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
        
        if (len(c_indices) == len(child_coords[0])) and (counts.min() > min_overlap):
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
    

    for p_index, c_indices in enumerate(p_c_indices):

        for c_index_rel, c_index in enumerate(list(c_indices)):
            c_p_indices[c_index].append(p_index)
            c_p_indices_rel[c_index].append(c_index_rel)
            c_p_distances[c_index].append(pc_distance[p_index, c_index_rel])
            

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
    
    indices = {'children': p_c_indices,
               'children_idx': torch.stack(c_p_indices_rel_pad),
              'parents':torch.stack(c_p_indices_pad)}

    return indices, radius_km


def get_relative_coordinates_grids(parent_coords, child_coords, relation_dict, relative_to="parents", batch_size=-1):

    if relative_to == 'children':
        coords_in_regions = [parent_coords[0][relation_dict['parents']], parent_coords[1][relation_dict['parents']]]
        region_center_coords = child_coords

    else:
        coords_in_regions = [child_coords[0][relation_dict['children']], child_coords[1][relation_dict['children']]]
        region_center_coords = parent_coords
    
    return get_relative_coordinates_regions(coords_in_regions, region_center_coords, batch_size=batch_size)


def get_relative_coordinates_regions(coords_in_regions, region_center_coords, batch_size=-1):

    PosCalc = PositionCalculator(batch_size)

    rel_coords = []

    for p in range(coords_in_regions[0].shape[0]):

        dist, phi, dlon, dlat = PosCalc(coords_in_regions[0][[p]].T, coords_in_regions[1][[p]].T,
                                        (region_center_coords[0][p]).view(1,1),
                                         (region_center_coords[1][p]).view(1,1),
                                         ((region_center_coords[0][p]).view(1,1),
                                         (region_center_coords[1][p]).view(1,1)))
        rel_coords.append(torch.stack([dist, phi, dlon, dlat],axis=0))

    rel_coords = torch.concat(rel_coords, dim=1).permute(1,2,0)

    return rel_coords


def get_regions(lons, lats, seeds_lon, seeds_lat, radius_region=None, n_points=None ,in_deg=True, rect=False, batch_size=-1):

    
    if in_deg:
        seeds_lon = torch.deg2rad(seeds_lon)
        seeds_lat = torch.deg2rad(seeds_lat)

    if rect:
        Pos_calc = PositionCalculator(batch_size=batch_size)
        d_mat, phi, dlon, dlat = Pos_calc(lons, lats, seeds_lon, seeds_lat)

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

        d_mat = Pos_calc(lons, lats, seeds_lon, seeds_lat)[0].T

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
                distances *= radius_earth*2

            distances_regions.append(distances)
            indices_regions.append(indices)
            idx+=0

        distances_regions = torch.concat(distances_regions, dim=-1)
        indices_regions = torch.concat(indices_regions, dim=-1)

    lons_regions = lons.view(-1)[indices_regions]
    lats_regions = lats.view(-1)[indices_regions]

    return distances_regions, indices_regions, lons_regions, lats_regions


def rotate_coord_system(lons: torch.tensor, lats: torch.tensor, rotation_lon: float, rotation_lat:float):

    theta = torch.tensor(rotation_lat) if not torch.is_tensor(rotation_lat) else rotation_lat
    phi = torch.tensor(rotation_lon) if not torch.is_tensor(rotation_lon) else rotation_lon

    x = torch.cos(lons)* torch.cos(lats)
    y = torch.sin(lons)* torch.cos(lats)
    z = torch.sin(lats)

    rotated_x = torch.cos(theta)*torch.cos(phi)*x + torch.cos(theta)*torch.sin(phi)*y + torch.sin(theta)*z
    rotated_y = -torch.sin(phi)*x + torch.cos(phi)*y
    rotated_z = -torch.sin(theta)*torch.cos(phi)*x - torch.sin(theta)*torch.sin(phi)*y+ torch.cos(theta)*z

    rot_lon = torch.atan2(rotated_y, rotated_x)
    rot_lat = torch.arcsin(rotated_z)

    return rot_lon, rot_lat


class PositionCalculator(nn.Module):
    def __init__(self, batch_size=-1):
        super().__init__()
        self.batch_size = batch_size

    def forward(self, lons, lats, lons_t=None, lats_t=None, rotation_center=None):

        if rotation_center is not None:
            lons, lats = rotate_coord_system(lons, lats, float(rotation_center[0]), float(rotation_center[1]))
            lons_t, lats_t = rotate_coord_system(lons_t, lats_t, float(rotation_center[0]), float(rotation_center[1]))
            
        lons = lons.view(1,(lons).numel())
        lats = lats.view(1,(lats).numel())

        if self.batch_size != -1 and self.batch_size <= lons.shape[-1]:
            n_chunks = lons.shape[-1] // self.batch_size 
        else:
            n_chunks = 1
        
        lons_chunks = lons.chunk(n_chunks, dim=-1)
        lats_chunks = lats.chunk(n_chunks, dim=-1)

        d_lons = []
        d_lats = []
        d_mat = []
        phis = []
        for chunk_idx in range(len(lons_chunks)):
            lons_chunk = lons_chunks[chunk_idx]
            lats_chunk = lats_chunks[chunk_idx]

            if lons_t is None:
                lons_t=lons_chunk.transpose(1,0)
                lats_t=lats_chunk.transpose(1,0)

                d_lons_chunk = distance_on_sphere(lons_chunk,lats_chunk,lons_t,lats_t.transpose(0,1))
                d_lats_chunk = distance_on_sphere(lons_chunk,lats_chunk,lons_t.transpose(0,1),lats_t)
            else:
                lons_t = lons_t.view(len(lons_t),1)
                lats_t = lats_t.view(len(lats_t),1)

                d_lats_chunk = distance_on_sphere(lons_chunk, lats_chunk, lons_chunk, lats_t)
                d_lons_chunk = distance_on_sphere(lons_chunk, lats_chunk, lons_t, lats_chunk)

            sgn = torch.sign(lats_chunk - lats_t)
            sgn[(lats_chunk - lats_t).abs()/torch.pi>1] = sgn[(lats_chunk - lats_t).abs()/torch.pi>1]*-1
            d_lats_s = d_lats_chunk*sgn

            sgn = torch.sign(lons_chunk - lons_t)
            sgn[(lons_chunk - lons_t).abs()/torch.pi>1] = sgn[(lons_chunk - lons_t).abs()/torch.pi>1]*-1
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
    


def generate_region(coords, range_lon=None, range_lat=None, n_points=None, radius=None, locations=[], batch_size=1, rect=False):

    if len(locations)==0:
        seeds_lon = torch.randint(range_lon[0],range_lon[1], size=(batch_size,1))
        seeds_lat = torch.randint(range_lat[0],range_lat[1], size=(batch_size,1))
    else:
        seeds_lon = locations[0]
        seeds_lat = locations[1]


    distances_regions, indices_regions, lons_regions, lats_regions = get_regions(coords['lon'], coords['lat'], seeds_lon, seeds_lat, radius_region=radius, n_points=n_points, in_deg=True, rect=rect)
    n_points = len(distances_regions)
    radius = (distances_regions.max()/2)

    out_dict = {'distances': distances_regions,
                'indices': indices_regions,
                'lon': lons_regions,
                'lat': lats_regions,
                "n_points": n_points,
                'radius': radius,
                'locations': [seeds_lon, seeds_lat]}


    return out_dict