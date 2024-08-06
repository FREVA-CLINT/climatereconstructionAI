import torch
import torch.nn as nn
import netCDF4 as netcdf
import xarray as xr
import numpy as np
import climatereconstructionai.model.transformer_helpers as helpers

def arclen(p1,p2):
  length = 2*torch.arcsin(torch.linalg.norm(p2-p1,axis=-1)/2)
  return length

class physics_calculator():
    def __init__(self, grid_file_path, device='cpu') -> None:
        
        self.rad = 6371e3
        self.device=device
        dset = netcdf.Dataset(grid_file_path)

        self.zonal_normal_primal_edge = torch.from_numpy(dset['zonal_normal_primal_edge'][:].data).to(device)
        self.meridional_normal_primal_edge = torch.from_numpy(dset['meridional_normal_primal_edge'][:].data).to(device)
        self.edges_of_vertex = torch.from_numpy(dset['edges_of_vertex'][:].data.T-1).clamp(min=0).long().to(device)
        self.dual_edge_length = torch.from_numpy(dset['dual_edge_length'][:].data).to(device)
        self.edge_length = torch.from_numpy(dset['edge_length'][:].data).to(device)
        self.edge_vertices = torch.from_numpy(dset['edge_vertices'][:].data).long().to(device)-1

        self.cell_area = torch.from_numpy(dset['cell_area'][:].data).to(device)
        self.edge_of_cell = torch.from_numpy(dset['edge_of_cell'][:].data.T-1).to(device)
        self.adjacent_cell_of_edge = torch.from_numpy(dset['adjacent_cell_of_edge'][:].data.T-1).long().to(device)

        self.dual_area = torch.tensor(dset['dual_area'][:].data, device=device)

        self.orientation_of_normal = torch.from_numpy(dset['orientation_of_normal'][:].data).to(device)
        
        
        self.edge_primal_normal_cartesian = torch.tensor([dset['edge_primal_normal_cartesian_x'][:].data,dset['edge_primal_normal_cartesian_y'][:].data,
                                            dset['edge_primal_normal_cartesian_z'][:].data],device=device).T
        
        self.cartesian_cell_circumcenter = torch.tensor([dset['cell_circumcenter_cartesian_x'][:].data,dset['cell_circumcenter_cartesian_y'][:].data,
                                            dset['cell_circumcenter_cartesian_z'][:].data],device=device).T
        

        edge_dual_normal_cartesian_x = dset['edge_dual_normal_cartesian_x'][:].data
        edge_dual_normal_cartesian_y = dset['edge_dual_normal_cartesian_y'][:].data
        edge_dual_normal_cartesian_z = dset['edge_dual_normal_cartesian_z'][:].data

        edge_middle_cartesian_x = dset['edge_middle_cartesian_x'][:].data
        edge_middle_cartesian_y = dset['edge_middle_cartesian_y'][:].data
        edge_middle_cartesian_z = dset['edge_middle_cartesian_z'][:].data

        cartesian_x_vertices = dset['cartesian_x_vertices'][:].data
        cartesian_y_vertices = dset['cartesian_y_vertices'][:].data
        cartesian_z_vertices = dset['cartesian_z_vertices'][:].data


        cartesian_vertices = torch.tensor([cartesian_x_vertices,cartesian_y_vertices,
                                            cartesian_z_vertices],device=device).T

        edge_dual_normal_cartesian = torch.tensor([edge_dual_normal_cartesian_x,edge_dual_normal_cartesian_y,
                                            edge_dual_normal_cartesian_z],device=device).T

        edge_middle_cartesian = torch.tensor([edge_middle_cartesian_x,edge_middle_cartesian_y,
                                        edge_middle_cartesian_z],device=device).T

        
        self.nout = edge_middle_cartesian[self.edges_of_vertex]-cartesian_vertices.unsqueeze(dim=1)
        self.signorient = torch.sign((self.nout*edge_dual_normal_cartesian[self.edges_of_vertex]).sum(axis=-1))

        self.edge_middle_cartesian = edge_middle_cartesian.to(device)

        self.nout = self.nout.to(device)
        self.signorient = self.signorient.to(device)
        self.coords_v = torch.stack([torch.tensor(dset['vlon'][:].data, device=device), torch.tensor(dset['vlat'][:].data, device=device)])
        dset.close()

    def get_vorticity(self, normalVelocity):
        vort = ((normalVelocity*self.dual_edge_length)[self.edges_of_vertex]*self.signorient).sum(axis=-1)/self.dual_area
        return vort
    

    def get_vorticity_from_uv(self, u, v):
        normalVectorx = self.zonal_normal_primal_edge
        normalVectory = self.meridional_normal_primal_edge

        normalVelocity = u * normalVectorx + v * normalVectory

        return self.get_vorticity(normalVelocity)
    

    def get_vorticity_from_ds(self, ds, ts):
        u = torch.from_numpy(ds['u'][:].data[ts,0].astype(np.float32))
        v = torch.from_numpy(ds['v'][:].data[ts,0].astype(np.float32))

        return self.get_vorticity_from_uv(u,v)
    

    def get_normal_velocity_from_indices(self, global_edge_indices, u, v):
        b,n = global_edge_indices.shape
        n_edges_global = self.zonal_normal_primal_edge.shape[-1]

        global_edge_indices_b = global_edge_indices + (torch.arange(b, device=u.device)*n_edges_global).view(b,1)

        u_global = torch.zeros((b*n_edges_global),device=u.device)
        v_global = torch.zeros((b*n_edges_global),device=u.device)
        valid_mask = torch.zeros((b*n_edges_global),device=u.device, dtype=bool)

        u_global[global_edge_indices_b.view(-1)] = u.contiguous().view(-1)
        v_global[global_edge_indices_b.view(-1)] = v.contiguous().view(-1)
        valid_mask[global_edge_indices_b.view(-1)] = True

        u_global = u_global.view(b,n_edges_global)
        v_global = v_global.view(b,n_edges_global)
        valid_mask = valid_mask.view(b,n_edges_global)

        normalVelocity = u_global*self.zonal_normal_primal_edge.to(u.device) + v_global*self.meridional_normal_primal_edge.to(u.device)

        return normalVelocity, valid_mask, u_global, v_global

    def get_normal_velocity_var_from_indices(self, global_edge_indices, var_u, var_v):

        b,n = global_edge_indices.shape
        n_edges_global = self.zonal_normal_primal_edge.shape[-1]

        global_edge_indices_b = global_edge_indices + (torch.arange(b, device=var_u.device)*n_edges_global).view(b,1)

        var_u_global = torch.zeros((b*n_edges_global),device=var_u.device)
        var_v_global = torch.zeros((b*n_edges_global),device=var_u.device)
        valid_mask = torch.zeros((b*n_edges_global),device=var_u.device, dtype=bool)

        var_u_global[global_edge_indices_b.view(-1)] = var_u.contiguous().view(-1)
        var_v_global[global_edge_indices_b.view(-1)] = var_v.contiguous().view(-1)
        valid_mask[global_edge_indices_b.view(-1)] = True

        var_u_global = var_u_global.view(b,n_edges_global)
        var_v_global = var_v_global.view(b,n_edges_global)
        valid_mask = valid_mask.view(b,n_edges_global)

        var_normalVelocity = var_u_global*(self.zonal_normal_primal_edge**2).to(var_u_global.device) + var_v_global*(self.meridional_normal_primal_edge**2).to(var_v_global.device)

        return var_normalVelocity, valid_mask, var_u_global, var_v_global


    def get_vorticity_from_edge_indices(self, global_edge_indices, u, v, n_vert=6, normalv=None, normalvmask=None):
        b,n = global_edge_indices.shape

        if normalv is None:
            normalv, mask,_,_ = self.get_normal_velocity_from_indices(global_edge_indices, u, v)

        vort_mask = ((mask)[:,self.edges_of_vertex]).sum(axis=-1) >= n_vert
        vort = ((normalv*self.dual_edge_length)[:,self.edges_of_vertex]*self.signorient).sum(axis=-1)/self.dual_area.unsqueeze(dim=0)

        return vort.view(b,1,1,-1), ~vort_mask


    def get_filtered_div(self, normalVelocity):
        dek = arclen((self.cartesian_cell_circumcenter/torch.linalg.norm(self.cartesian_cell_circumcenter,dim=-1).unsqueeze(dim=-1)).unsqueeze(dim=1),
                (self.edge_middle_cartesian[self.edge_of_cell]/torch.linalg.norm(self.edge_middle_cartesian[self.edge_of_cell],dim=-1).unsqueeze(dim=-1)))*self.rad

        Pu = ((((normalVelocity*self.edge_length)[:,self.edge_of_cell]*dek).unsqueeze(dim=-1)*self.edge_primal_normal_cartesian[self.edge_of_cell].unsqueeze(dim=0)).sum(dim=2)/self.cell_area.unsqueeze(dim=-1).unsqueeze(dim=0))

        dek = torch.stack([arclen(self.cartesian_cell_circumcenter[self.adjacent_cell_of_edge[:,0]],self.edge_middle_cartesian),
            arclen(self.cartesian_cell_circumcenter[self.adjacent_cell_of_edge[:,1]],self.edge_middle_cartesian)]).T*self.rad

        hPu = Pu
        PThPu = ((hPu[:,self.adjacent_cell_of_edge]*dek.unsqueeze(dim=-1).unsqueeze(dim=0)).sum(dim=2)/self.dual_edge_length.unsqueeze(dim=-1).unsqueeze(dim=0)*self.edge_primal_normal_cartesian.unsqueeze(dim=0)).sum(dim=-1)
        div = ((PThPu*self.edge_length.unsqueeze(dim=0))[:,self.edge_of_cell]*self.orientation_of_normal.T.unsqueeze(dim=0)).sum(dim=-1)/self.cell_area.unsqueeze(dim=0) 

        return div

    def get_divergence_from_edge_indices(self, global_edge_indices, u, v, filtered=True, normalv=None):

        if normalv is None:
            normalv, mask,_,_ = self.get_normal_velocity_from_indices(global_edge_indices, u, v)
        
        if filtered:
            # to be implemented
            div = self.get_filtered_div(normalv)
            div_mask = mask[:,self.edge_of_cell].sum(dim=-1) == 3

        else:
            div_mask = mask[:,self.edge_of_cell].sum(dim=-1) == 3
            div = ((normalv*self.edge_length.unsqueeze(dim=0))[:,self.edge_of_cell]*self.orientation_of_normal.T.unsqueeze(dim=0)).sum(dim=-1)/self.cell_area.unsqueeze(dim=0)

        div_sum = (self.cell_area.unsqueeze(dim=0)*div/self.rad).sum(dim=1)
       
        return div, ~div_mask, div_sum
    

    def get_kinetic_energy_from_edge_indices(self, global_edge_indices, u, v):

        _, mask, u_global, v_global = self.get_normal_velocity_from_indices(global_edge_indices, u, v)

        kin_energy = (((u_global**2+v_global**2)/2)[:,self.edge_of_cell]).sum(dim=-1)/self.cell_area.unsqueeze(dim=0)
        kin_energy_mask = mask[:,self.edge_of_cell].sum(dim=-1) == 3
               
        return kin_energy, ~kin_energy_mask
    
    
    def get_adjacent_cells(self):
        adjc_indices = self.adjacent_cell_of_edge[self.edge_of_cell].reshape(-1,1,6)
        self_indices = torch.arange(self.cell_area.shape[0], device=adjc_indices.device).view(-1,1)

        adjc_indices = adjc_indices.view(self_indices.shape[0],-1) 
        self_indices = self_indices 

        adjc_unique = (adjc_indices).long().unique(dim=-1)
        
        is_self = adjc_unique - self_indices.view(-1,1) == 0

        adjc = adjc_unique[~is_self]

        adjc = adjc.reshape(self_indices.shape[0], -1)

        self.adjc = adjc


    def get_nh(self, x, sample_dict):
        indices_nh, mask = helpers.get_nh_of_batch_indices(sample_dict['global_cell'], self.adjc)

        x_nh = {}
        for key, data in x.items():
            data = data.view(data.shape[0], data.shape[1],-1)

            b,n,e = data.shape
            nh = indices_nh.shape[-1]

            local_cell_indices_nh_batch = indices_nh - (sample_dict['sample']*4**(sample_dict['sample_level'] - 0)).view(-1,1,1)

            x_nh[key] =torch.gather(data.reshape(b,-1,e),1, index=local_cell_indices_nh_batch.reshape(b,-1,1).repeat(1,1,e)).reshape(b,n,nh,e)

        return x_nh

    

def get_vorticity(phys_calc, tensor_dict, global_edge_indices, n_vert=6, normalv=None):
    u = tensor_dict['u']
    v = tensor_dict['v']

    u = u[:,0,0] if u.dim()==4 else u
    v = v[:,0,0] if v.dim()==4 else v
    vort, non_valid_mask = phys_calc.get_vorticity_from_edge_indices(global_edge_indices, u, v, n_vert=n_vert, normalv=normalv)
    return vort, non_valid_mask


def get_normalv(phys_calc, tensor_dict, global_edge_indices):
    u = tensor_dict['u']
    v = tensor_dict['v']

    u = u[:,0,0] if u.dim()==4 else u
    v = v[:,0,0] if v.dim()==4 else v
    normalv, valid_mask, _, _ = phys_calc.get_normal_velocity_from_indices(global_edge_indices, u, v)
    return normalv.unsqueeze(dim=1).unsqueeze(dim=2), ~valid_mask

def get_normalv_var(phys_calc, tensor_dict, global_edge_indices):
    u = tensor_dict['u']
    v = tensor_dict['v']

    var_u = u[:,0,1]
    var_v = v[:,0,1] 
    normalv, valid_mask, _, _ = phys_calc.get_normal_velocity_var_from_indices(global_edge_indices, var_u, var_v)
    return normalv.unsqueeze(dim=1).unsqueeze(dim=2), ~valid_mask

class NHTVLoss(nn.Module):
    def __init__(self, phys_calc):
        super().__init__()
        self.phys_calc = phys_calc
        self.phys_calc.get_adjacent_cells()

    def forward(self, output, target, sample_dict):
        output = self.phys_calc.get_nh(output, sample_dict)

        loss = 0
        for key, data in output.items():
            loss += ((data - data[:,:,[0],:])**2).mean()
        return loss

class GaussLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.Gauss = torch.nn.GaussianNLLLoss()

    def forward(self, output, target):
        loss =  self.Gauss(output[:,:,[0]], target, output[:,:,[1]])
        return loss


def get_sum(output_levels, lowest_level, gauss=False):

    if not isinstance(output_levels["x"], list):
        return output_levels["x"].unsqueeze(dim=-2)
    
    output = output_levels['x'][lowest_level:]
    if gauss:
        output_var = output_levels['x_var'][lowest_level:]

        x = torch.sum(torch.stack(output, dim=-1), dim=-1)
        x_var = torch.sum(torch.stack(output_var, dim=-1), dim=-1)
        x = torch.stack([x, x_var],dim=-2)
    
    else:
        x = torch.sum(torch.stack(output, dim=-1), dim=-1).unsqueeze(dim=-2)

    return x


class HierLoss(nn.Module):
    def __init__(self, loss_fcn, lambdas_levels):
        super().__init__()
        self.loss_fcn = loss_fcn
        self.lambdas_levels = lambdas_levels

        self.gauss=False
        if isinstance(loss_fcn, GaussLoss):
            self.gauss=True


    def forward(self, output_levels, target, in_range_mask=None):
        loss_dict = {}
        total_loss = 0
        for level, lambda_ in self.lambdas_levels.items():
            if lambda_ > 0:
                output = get_sum(output_levels, int(level), gauss=self.gauss)

                if in_range_mask is not None:
                    loss = lambda_ * self.loss_fcn(output[in_range_mask[:,:,0]==True,:,:], target['cell'][in_range_mask==True,:])
                else:
                    loss = lambda_ * self.loss_fcn(output, target['cell'])
                loss_dict[f'level_{level}'] = loss.item()
                total_loss += loss

        return total_loss, loss_dict

class TVLoss(nn.Module):
    def __init__(self, lambdas_levels):
        super().__init__()
        self.lambdas_levels = lambdas_levels

    def forward(self, model, output_levels, source_indices, in_range_mask, multi_gpu=False):

        loss_dict = {}
        total_loss = 0
        for level, lambda_ in self.lambdas_levels.items():
            if lambda_ > 0:
                
                if multi_gpu:
                    nh_values, _ ,nh_mask  = model.module.decomp_layer.grid_layers[str(0)].get_nh(output_levels["x"][int(level)], source_indices["global_cell"], source_indices)
                else:
                    nh_values, _ ,nh_mask  = model.decomp_layer.grid_layers[str(0)].get_nh(output_levels["x"][int(level)], source_indices["global_cell"], source_indices)

                nh_values_error = ((nh_values[:,:,[0]] - nh_values[:,:,1:4])**4).sum(dim=[-2])

                loss = lambda_ * nh_values_error.mean()

                loss_dict[f'level_{level}'] = loss.item()
                total_loss += loss

        return total_loss, loss_dict

class loss_calculator(nn.Module):
    def __init__(self, training_settings, grid_variables_dict, model_settings):
        super().__init__()

        #self.lambdas_var = training_settings['lambdas_var']
        self.lambdas_levels = training_settings['lambdas_levels']
        self.lambdas_static = training_settings['lambdas']
        self.mask_out_of_range = training_settings['mask_out_of_range'] if 'mask_out_of_range' in training_settings.keys() else False
        self.multi_gpus = training_settings['multi_gpus']
        self.grid_variables_dict = grid_variables_dict

        self.loss_fcn_dict = {} 

        for loss_type, value in self.lambdas_static.items():
            if value > 0:
                if loss_type == 'gauss':
                    loss_fcn = GaussLoss()
                    self.loss_fcn_dict[loss_type] = HierLoss(loss_fcn, {0:self.lambdas_levels[0]})
                elif loss_type == 'l1':
                    loss_fcn = torch.nn.L1Loss() 
                    self.loss_fcn_dict[loss_type] = HierLoss(loss_fcn, self.lambdas_levels)
                elif loss_type == 'l2':
                    loss_fcn = torch.nn.MSELoss()
                    self.loss_fcn_dict[loss_type] = HierLoss(loss_fcn, self.lambdas_levels)
                elif loss_type == 'tv':
                    lambdas_tv = dict(zip(self.lambdas_levels.keys(), [lambda_*self.lambdas_static['tv'] for lambda_ in self.lambdas_levels.values()]))
                    self.loss_fcn_dict[loss_type] = TVLoss(lambdas_tv)


    def forward(self, lambdas_optim, target, model, source, source_indices=None, val=False, k=None):
        
        if val:
            with torch.no_grad():
                output_levels, debug_dict = model(source, source_indices, debug=True)
        else:
            output_levels = model(source, source_indices)

        loss_dict = {}
        total_loss = 0


        if self.mask_out_of_range:
            if self.multi_gpus:
                in_range_mask = model.module.output_in_range[source_indices['global_cell']]
            elif 'global_cell' in model.__dict__['_buffers'].keys():
                in_range_mask = model.output_in_range[source_indices['global_cell']]
            else:
                in_range_mask = None
        else:
            in_range_mask = None

        for loss_type, loss_fcn in self.loss_fcn_dict.items():
            
            if loss_type == 'tv':
                loss, loss_levels = loss_fcn(model, output_levels, source_indices, in_range_mask, multi_gpu=self.multi_gpus)
            else:
                loss, loss_levels = loss_fcn(output_levels, target, in_range_mask)
            total_loss += self.lambdas_static[loss_type] * lambdas_optim[loss_type] * loss

            loss_levels_keys = [f'{loss_type}_{key}' for key in loss_levels.keys()]
            loss_dict.update(dict(zip(loss_levels_keys, list(loss_levels.values()))))

        loss_dict['total_loss'] = total_loss.item()
        
        if val:
            return total_loss, loss_dict, output_levels, target, debug_dict
        else:
            return total_loss, loss_dict
        