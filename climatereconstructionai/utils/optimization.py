import torch
import torch.nn as nn
import netCDF4 as netcdf
import xarray as xr
import numpy as np

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

        global_edge_indices_b = global_edge_indices + (torch.arange(b, device=self.device)*n_edges_global).view(b,1)

        u_global = torch.zeros((b*n_edges_global),device=self.device)
        v_global = torch.zeros((b*n_edges_global),device=self.device)
        valid_mask = torch.zeros((b*n_edges_global),device=self.device, dtype=bool)

        u_global[global_edge_indices_b.view(-1)] = u.contiguous().view(-1)
        v_global[global_edge_indices_b.view(-1)] = v.contiguous().view(-1)
        valid_mask[global_edge_indices_b.view(-1)] = True

        u_global = u_global.view(b,n_edges_global)
        v_global = v_global.view(b,n_edges_global)
        valid_mask = valid_mask.view(b,n_edges_global)

        normalVelocity = u_global*self.zonal_normal_primal_edge + v_global*self.meridional_normal_primal_edge

        return normalVelocity, valid_mask


    def get_vorticity_from_edge_indices(self, global_edge_indices, u, v):
        b,n = global_edge_indices.shape
        normalVelocity, mask = self.get_normal_velocity_from_indices(global_edge_indices, u, v)
        vort_mask = ((mask)[:,self.edges_of_vertex]).sum(axis=-1) == 6
        vort = ((normalVelocity*self.dual_edge_length)[:,self.edges_of_vertex]*self.signorient).sum(axis=-1)/self.dual_area.unsqueeze(dim=0)
        vort[~vort_mask] = 0

        return vort.view(b,1,1,-1), ~vort_mask


    def get_divergence_from_edge_indices(self, global_edge_indices, u, v, zos=None):

        normalVelocity, mask = self.get_normal_velocity_from_indices(global_edge_indices, u, v)
        
        if zos is not None:
            # to be implemented
            dek = arclen((self.cartesian_cell_circumcenter/torch.linalg.norm(self.cartesian_cell_circumcenter,dim=-1).unsqueeze(dim=-1)).unsqueeze(dim=1),
                    (self.edge_middle_cartesian[self.edge_of_cell]/torch.linalg.norm(self.edge_middle_cartesian[self.edge_of_cell],dim=-1).unsqueeze(dim=-1)))*self.rad
            
            Pu = (((normalVelocity*self.edge_length)[self.edge_of_cell]*dek).unsqueeze(dim=-1)*self.edge_primal_normal_cartesian[self.edge_of_cell]).sum(dim=1)/self.cell_area.unsqueeze(dim=-1)
            dek = torch.stack([arclen(self.cartesian_cell_circumcenter[self.adjacent_cell_of_edge[:,0]],self.edge_middle_cartesian),
                        arclen(self.cartesian_cell_circumcenter[self.adjacent_cell_of_edge[:,1]],self.edge_middle_cartesian)]).T*self.rad
            
            hPu = zos.unsqueeze(dim=-1)*Pu
            PThPu = ((hPu[self.adjacent_cell_of_edge]*dek.unsqueeze(dim=-1)).sum(dim=1)/self.dual_edge_length.unsqueeze(dim=-1)*self.edge_primal_normal_cartesian).sum(dim=-1)
            div = ((PThPu*self.edge_length)[self.edge_of_cell]*self.orientation_of_normal.T).sum(dim=-1)/self.cell_area 

        else:
            div_mask = mask[:,self.edge_of_cell].sum(dim=-1) == 3
            div = ((normalVelocity*self.edge_length.unsqueeze(dim=0))[:,self.edge_of_cell]*self.orientation_of_normal.T.unsqueeze(dim=0)).sum(dim=-1)/self.cell_area.unsqueeze(dim=0)
            div[~div_mask] = 0
            
        return div, ~div_mask
    

def get_vorticity(phys_calc, tensor_dict, global_edge_indices):
    u = tensor_dict['u']
    v = tensor_dict['v']

    u = u[:,0,0] if u.dim()==4 else u
    v = v[:,0,0] if v.dim()==4 else v
    vort, non_valid_mask = phys_calc.get_vorticity_from_edge_indices(global_edge_indices, u, v)
    return vort, non_valid_mask


class GaussLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.Gauss = torch.nn.GaussianNLLLoss()

    def forward(self, output, target, non_valid_mask):
        output_valid = output[:,0].transpose(-2,-1)[~non_valid_mask]
        target_valid = target[~non_valid_mask].squeeze()
        loss =  self.Gauss(output_valid[:,0],target_valid,output_valid[:,1])
        return loss

class L1Loss(nn.Module):
    def __init__(self, loss='l1'):
        super().__init__()
        if loss=='l1':
            self.loss = torch.nn.L1Loss()
        else:
            self.loss = torch.nn.MSELoss()

    def forward(self, output, target, non_valid_mask):
        output_valid = output[:,0,0][~non_valid_mask]
        target_valid = target.squeeze()[~non_valid_mask].squeeze()
        loss = self.loss(output_valid,target_valid)
        return loss

class L1Loss_rel(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.MSELoss()

    def forward(self, output, target, non_valid_mask):
        output_valid = output[:,0,0][~non_valid_mask]
        target_valid = target[~non_valid_mask].squeeze()
        abs_loss = ((output_valid - target_valid)/(target_valid+1e-10)).abs()
        loss = abs_loss.clamp(max=1)
        loss = loss.mean()
        return loss
    
class TVLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output_hr):
        
        loss = (output_hr[:,:,1:] - output_hr[:,:,:-1]).abs().mean() + (output_hr[:,:,:,1:] - output_hr[:,:,:,:-1]).abs().mean()

        return loss
    
class TVLoss_log(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output_hr):
        output_hr = (output_hr.abs()+1e-10).log10()

        loss = (output_hr[:,:,1:] - output_hr[:,:,:-1]).abs().mean() + (output_hr[:,:,:,1:] - output_hr[:,:,:,:-1]).abs().mean()

        return loss


class LogLoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.loss = torch.nn.MSELoss()

    def forward(self, output, target, non_valid_mask):
        output_valid = output[:,0,0][~non_valid_mask]
        target_valid = target.squeeze()[~non_valid_mask].squeeze()

        output_sgn = output_valid.sign()
        target_sgn = target_valid.sign()

        loss_sgn = self.loss(output_sgn,target_sgn)
        output_mag = (output_valid.abs()+1e-10).log10()
        target_mag = (target_valid.abs()+1e-10).log10()

        loss_mag = self.loss(output_mag,target_mag)
        return loss_sgn + loss_mag

class TVLoss_rel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output_hr):
        
        rel_diff1 = ((output_hr[:,:,1:] - output_hr[:,:,:-1])/(output_hr[:,:,1:]+1e-10)).abs()
        rel_diff1 = rel_diff1.clamp(max=1)

        rel_diff2 = ((output_hr[:,:,:,1:] - output_hr[:,:,:,:-1])/(output_hr[:,:,:,1:]+1e-10)).abs()
        rel_diff2 = rel_diff2.clamp(max=1)

        loss = (rel_diff1.mean() + rel_diff2.mean())

        return loss

class DictLoss(nn.Module):
    def __init__(self, loss_fcn):
        super().__init__()
        self.loss_fcn = loss_fcn

    def forward(self, output, target, non_valid_mask):
        loss_dict = {}
        total_loss = 0

        for var in output.keys():
            loss = self.loss_fcn(output[var], target[var], non_valid_mask[var])
            total_loss+=loss
            loss_dict[f'{var}_{str(self.loss_fcn._get_name())}'] = loss.item()

        return total_loss

class VortLoss(nn.Module):
    def __init__(self, phys_calc):
        super().__init__()
        self.phys_calc = phys_calc
        self.loss_fcn = L1Loss()

    def forward(self, output, target, target_indices, spatial_dim_var_target):

        spatial_dim_uv = [k for k,v in spatial_dim_var_target.items() if 'u' in v][0]
        uv_dim_indices = target_indices[spatial_dim_uv]
        output_vort, non_valid_mask_vort = get_vorticity(self.phys_calc, output, uv_dim_indices)
        target_vort = get_vorticity(self.phys_calc, target, uv_dim_indices)[0]

        vort_loss = self.loss_fcn(output_vort, target_vort, non_valid_mask_vort)  

        return vort_loss

class DivLoss(nn.Module):
    def __init__(self, phys_calc):
        super().__init__()
        self.phys_calc = phys_calc
        self.loss_fcn = L1Loss()

    def forward(self, output, target_indices, spatial_dim_var_target):
       
        u = output['u']
        v = output['v']

        u = u[:,0,0] if u.dim()==4 else u
        v = v[:,0,0] if v.dim()==4 else v

        spatial_dim_uv = [k for k,v in spatial_dim_var_target.items() if 'u' in v][0]
        uv_dim_indices = target_indices[spatial_dim_uv]

        div, non_valid_mask = self.phys_calc.get_divergence_from_edge_indices(uv_dim_indices, u, v)

        loss = (div[~non_valid_mask]**2).mean()     

        return loss

class Trivial_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.MSELoss()
        self.registered = False
    
    def register_samples(self, source, coords_target, target):
        self.source_0 = torch.zeros_like(source, device=source.device)
        self.target_0 = {}
        for var, values in target.items():
            if var == 'u' or var == 'v': 
                self.target_0[var] = torch.zeros_like(values, device=source.device)
            elif var == 'zos':
                self.target_0[var] = torch.ones_like(values, device=source.device)*0.5
        self.coords_target = coords_target
        self.registered = True

    def forward(self, model):
        output = model(self.source_0, self.coords_target)[0]

        total_loss = 0 
        for var in output.keys():
            loss = self.loss(output[var][:,0,0], self.target_0[var][:,:,0])
            total_loss+=loss

        return total_loss

class loss_calculator(nn.Module):
    def __init__(self, training_settings, spatial_dim_var_target):
        super().__init__()

        self.spatial_dim_var_target = spatial_dim_var_target
        self.lambdas_var = training_settings['lambdas_var']
        self.lambdas_static = training_settings['lambdas']

        self.loss_fcn_dict = {} 

        phys_calc = None
        for loss_type, value in self.lambdas_static.items():
        
            if loss_type == 'tv' and value > 0:
                self.loss_fcn_dict['tv'] = TVLoss()

            elif loss_type == 'tv_log' and value > 0:
                self.loss_fcn_dict['tv_log'] = TVLoss_log()

            elif loss_type == 'tv_rel' and value > 0:
                self.loss_fcn_dict['tv_rel'] = TVLoss_rel()

            elif loss_type == 'l2' and value > 0:
                self.loss_fcn_dict['l2'] = DictLoss(L1Loss(loss='l2'))

            elif loss_type == 'l1' and value > 0:
                self.loss_fcn_dict['l1'] = DictLoss(L1Loss(loss='l1'))
            
            elif loss_type == 'log' and value > 0:
                self.loss_fcn_dict['log'] = DictLoss(LogLoss())

            elif loss_type == 'rel' and value > 0:
                self.loss_fcn_dict['rel'] = DictLoss(L1Loss_rel())

            elif loss_type == 'trivial' and value > 0:
                self.loss_fcn_dict['trivial'] = Trivial_loss()
            
            elif loss_type == 'gauss' and value > 0:
                self.loss_fcn_dict['gauss'] =DictLoss(GaussLoss())

            elif loss_type == 'vort' and value > 0:
                if phys_calc is None:
                    phys_calc = physics_calculator(training_settings['grid_file'], device=training_settings['device'])
                self.loss_fcn_dict['vort'] = VortLoss(phys_calc)

            elif loss_type == 'div' and value > 0:
                if phys_calc is None:
                    phys_calc = physics_calculator(training_settings['grid_file'], device=training_settings['device'])
                self.loss_fcn_dict['div'] = DivLoss(phys_calc)


    def forward(self, lambdas_optim, target, model, source, coords_target, target_indices, coords_source=None, val=False):
        
        if val:
            with torch.no_grad():
                output, _, output_reg_hr, non_valid_mask = model(source, coords_target, coords_source=coords_source)
        else:
            output, _, output_reg_hr, non_valid_mask = model(source, coords_target, coords_source=coords_source)

        if 'trivial' in self.loss_fcn_dict.keys() and not self.loss_fcn_dict['trivial'].registered:
            self.loss_fcn_dict['trivial'].register_samples(source, coords_target, target)

        loss_dict = {}
        total_loss = 0

        for loss_type, loss_fcn in self.loss_fcn_dict.items():
            if loss_type == 'trivial':
                loss =  loss_fcn(model)

            elif loss_type == 'tv' or loss_type == 'tv_rel' or loss_type == 'tv_log':
                loss =  loss_fcn(output_reg_hr)

            elif loss_type == 'l2' or loss_type == 'l1' or loss_type == 'gauss' or loss_type == 'log':
                loss =  loss_fcn(output, target, non_valid_mask)

            elif loss_type == 'rel':
                loss =  loss_fcn(output, target, non_valid_mask)

            elif loss_type == 'vort':
                loss =  loss_fcn(output, target, target_indices, self.spatial_dim_var_target)
            
            elif loss_type == 'div':
                loss =  loss_fcn(output, target_indices, self.spatial_dim_var_target)
            
            total_loss += self.lambdas_static[loss_type]*lambdas_optim[loss_type] * loss
            loss_dict[loss_type] = loss.item()

        loss_dict['total_loss'] = total_loss.item()
        
        if val:
            return total_loss, loss_dict, output, output_reg_hr, non_valid_mask
        else:
            return total_loss, loss_dict