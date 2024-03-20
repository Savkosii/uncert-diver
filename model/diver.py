import torch
import torch.nn as nn
import torch.nn.functional as NF
import torch_scatter

import sys
sys.path.append('..')

from model.mlps import PositionalEncoding, mlp, ImplicitMLP

from utils.ray_voxel_intersection import ray_voxel_intersect, masked_intersect
from utils.integrator import integrate, integrate_mlp

    
class DIVeR(nn.Module):
    def __init__(self, hparams):
        super(DIVeR, self).__init__()
        # setup voxel grid parameters
        self.voxel_num = hparams.voxel_num
        self.voxel_dim = hparams.voxel_dim
        self.grid_size = hparams.grid_size
        self.voxel_size = self.grid_size/self.voxel_num
        self.mask_scale = hparams.mask_scale # coarse model occupancy mask is different from the fine model
        self.white_back = hparams.white_back
        
        # assume zero centered voxel grid
        N = self.voxel_num*self.voxel_size
        self.xyzmin = -N*0.5 # voxel grid boundary
        self.xyzmax = N*0.5 # voxel grid boundary

        mask_voxel_num = int(self.voxel_num * self.mask_scale)
        # TODO: maybe not generalization?
        # coarse model use normal ray intersection without mask
        # occupancy mask && !uncertainty_coarse_mask
        self.register_parameter('coarse_mask',nn.Parameter(
            torch.ones(mask_voxel_num,mask_voxel_num,mask_voxel_num,dtype=torch.bool),requires_grad=False))

        if hparams.fine:
            # repeat_interleave(occpancy mask) || uncertainty_fine_mask
            self.register_parameter('fine_mask',nn.Parameter(
                torch.zeros(mask_voxel_num,mask_voxel_num,mask_voxel_num,dtype=torch.bool),requires_grad=False))

        
        # whether to use implicit model (or implicit2explicit)
        if hparams.implicit:
            self.mlp_voxel = ImplicitMLP(hparams.implicit_network_depth, hparams.implicit_channels, hparams.implicit_skips,\
                             self.voxel_dim, hparams.implicit_point_encode)
        else:
            vertex_coords = torch.stack(
                    [*torch.meshgrid(*[torch.arange(0,self.voxel_num+1)]*3)],dim=-1)
            vertex_coords = vertex_coords.reshape(-1, 3)
            # coord2index
            x, y, z = vertex_coords[:,0], vertex_coords[:,1], vertex_coords[:,2]
            vertex_coords_indices = z + y * (self.voxel_num+1) + x * (self.voxel_num+1)**2

            vertex_keys = vertex_coords_indices.reshape(
                    self.voxel_num+1, self.voxel_num+1, self.voxel_num+1)
            self.register_parameter('coarse_vertex_keys', 
                                    nn.Parameter(vertex_keys, requires_grad=False))

            vertex_embedding_weight = torch.randn((self.voxel_num+1)**3, self.voxel_dim)*1e-2
            self.vertex_embedding_weight = vertex_embedding_weight


        ### Input dim: self.voxel_dim (default: 64 or 32)
        # feature -> (density, view_feature) 
        # mlp_out: f (64) + sigma (1) + beta (1)
        mlp_dim, mlp_depth, mlp_out = hparams.mlp_point
        self.mlp1 = mlp(self.voxel_dim, [mlp_dim]*mlp_depth, mlp_out)
        self.beta_min = 0.01
        
        # (view_feature, viewing dir) -> rgb
        self.view_enc = PositionalEncoding(hparams.dir_encode)
        mlp_dim, mlp_depth = hparams.mlp_view
        view_dim = hparams.dir_encode*2*3+3
        self.mlp2 = mlp(view_dim+mlp_out-2,[mlp_dim]*mlp_depth,3)

        if hasattr(self, 'vertex_embedding_weight'):
            self.vertex_embedding = nn.Embedding(*self.vertex_embedding_weight.shape)
            self.vertex_embedding.weight.data = self.vertex_embedding_weight

    def init_voxels(self, evaluate, alpha_thresh, alpha_map_path, beta_thresh, beta_map_path):
        # TODO: decouple voxel_mask and voxel parameters for weight loading
        alpha_map = torch.load(alpha_map_path, map_location='cpu')
        beta_map = torch.load(beta_map_path, map_location='cpu')
        occupancy_mask = alpha_map > alpha_thresh
        uncert_mask = beta_map > beta_thresh
        self.coarse_mask.data = occupancy_mask & ~uncert_mask
        self.fine_mask.data = occupancy_mask & uncert_mask
        print("coarse_mask: {}".format(self.coarse_mask.float().mean()))
        print("fine_mask: {}".format(self.fine_mask.float().mean()))

        def compute_masked_vertex_coords(voxel_mask, mask_scale, N):
            if mask_scale < 1.0:
                voxel_mask = torch.repeat_interleave(voxel_mask, int(1 / mask_scale), dim=0) 
                voxel_mask = torch.repeat_interleave(voxel_mask, int(1 / mask_scale), dim=1) 
                voxel_mask = torch.repeat_interleave(voxel_mask, int(1 / mask_scale), dim=2)

            # TODO: maybe mask here?
            voxel_coords = torch.stack([*torch.meshgrid(*[torch.arange(0,N)]*3)],dim=-1)
            xmin, ymin, zmin = voxel_coords.split(1, dim=-1)
            xmax, ymax, zmax = (voxel_coords+1).split(1, dim=-1)
            del voxel_coords

            vertex_coords = torch.stack([
                xmin,ymin,zmin, xmin,ymin,zmax,
                xmin,ymax,zmin, xmin,ymax,zmax,
                xmax,ymin,zmin, xmax,ymin,zmax,
                xmax,ymax,zmin, xmax,ymax,zmax
            ],dim=-1).reshape(N, N, N, 8, 3)
            del xmin, ymin, zmin, xmax, ymax, zmax

            # (M*8, 3), where M is the number of masked voxels
            masked_vertex_coords = vertex_coords[voxel_mask].reshape(-1, 3)
            return masked_vertex_coords

        masked_vertex_coords = compute_masked_vertex_coords(
                self.coarse_mask, self.mask_scale, self.voxel_num)
        extra_masked_vertex_coords = compute_masked_vertex_coords(
                self.fine_mask, self.mask_scale / 2.0, self.voxel_num * 2)
        masked_vertex_coords = torch.cat(
                [masked_vertex_coords * 2, extra_masked_vertex_coords], dim=0)
        # (K, 3), (M*8)
        masked_vertex_unique, masked_vertex_rev = masked_vertex_coords.unique(
                                                dim=0, return_inverse=True)
        padding_idx = masked_vertex_unique.shape[0]
        padding_vertex = torch.zeros(1, 3)
        # (K+1, 3)
        masked_vertex_unique = torch.cat([masked_vertex_unique, padding_vertex], dim=0)


        if evaluate:
            vertex_embedding_weight = self.mlp_voxel(masked_vertex_unique.float()/((2*self.voxel_num)*0.25)-2.0)
        else:
            vertex_embedding_weight = torch.randn(masked_vertex_unique.shape[0], self.voxel_dim)*1e-2

        self.vertex_embedding = nn.Embedding(masked_vertex_unique.shape[0], self.voxel_dim)
        self.vertex_embedding.weight.data = vertex_embedding_weight

        def construct_vertex_keys(vertex_coords, vertex_rev, padding_idx, N):
            # coord2index
            x, y, z = vertex_coords[:,0], vertex_coords[:,1], vertex_coords[:,2]
            vertex_coords_indices = z + y * (N+1) + x * (N+1)**2

            vertex_keys = torch.zeros((N+1)**3, dtype=torch.long) + padding_idx
            vertex_keys = torch_scatter.scatter(
                        src=vertex_rev, index=vertex_coords_indices, 
                        out=vertex_keys, reduce='min')
            vertex_keys = vertex_keys.reshape(N+1, N+1, N+1)
            return vertex_keys

        fine_vertex_keys = construct_vertex_keys(masked_vertex_coords, 
                                                 masked_vertex_rev, padding_idx, 2*self.voxel_num)
        self.register_parameter('fine_vertex_keys',
                                nn.Parameter(fine_vertex_keys,requires_grad=False))

        coarse_keys_mask = (masked_vertex_coords % 2 == 0).all(-1)
        coarse_masked_vertex_coords = masked_vertex_coords[coarse_keys_mask] // 2
        coarse_masked_vertex_rev = masked_vertex_rev[coarse_keys_mask]

        coarse_vertex_keys = construct_vertex_keys(coarse_masked_vertex_coords, 
                                                   coarse_masked_vertex_rev, padding_idx, self.voxel_num)
        self.register_parameter('coarse_vertex_keys',
                                nn.Parameter(coarse_vertex_keys,requires_grad=False))


    
    def extract_features(self, os, ds, extract_fine=False):
        voxel_num = self.voxel_num
        voxel_size = self.voxel_size
        voxel_mask = self.coarse_mask
        mask_scale = self.mask_scale
        if extract_fine:
            voxel_mask = self.fine_mask
            voxel_num *= 2
            voxel_size /= 2.0
            mask_scale /= 2.0

        # (B, K, .)
        coord, mask, ts_unique = masked_intersect(
            os.contiguous(), ds.contiguous(),
            self.xyzmin, self.xyzmax, int(voxel_num), voxel_size,
            voxel_mask.contiguous(), mask_scale)
            
        if not mask.any(): # not hit, mask = []
            return mask, None, None

        coord_in = coord[..., :3] 
        coord_out = coord[..., 3:]

        # fix the bug of masked ray intersection
        pmin = torch.min((coord_in+1e-4).long(),(coord_out+1e-4).long())
        mask = mask & (pmin < voxel_num).all(-1)
        if not mask.any():
            return mask, None, None

            
        if hasattr(self,'coarse_vertex_keys'): # check whether use explicit or implicit query
            if not extract_fine:
                features_ = integrate(self.coarse_vertex_keys, self.vertex_embedding,
                                      coord_in[mask], coord_out[mask]) 
            else:
                features_ = integrate(self.fine_vertex_keys, self.vertex_embedding, 
                                      coord_in[mask], coord_out[mask]) 

        else:
            # The vertex coordinates used in mlp feature integration are
            # the coordinates under the voxel basis and divided by voxel_nums
            # i.e., a relative coordinates. Thus use the same voxels_mlp is fine!
            features_ = integrate_mlp(self.mlp_voxel, voxel_num+1, 
                                  self.voxel_dim, coord_in[mask], coord_out[mask])

        features = torch.zeros(*mask.shape, features_.shape[-1], device=mask.device)
        features[mask] = features_


        ts_in = (coord_in - (os[:, None, :] - self.xyzmin) / voxel_size) \
                * voxel_size / ds[:, None, :] 
        ts_in = torch.mean(ts_in, dim=-1) # (B, K)
        ts_out = (coord_out - (os[:, None, :] - self.xyzmin) / voxel_size) \
                * voxel_size / ds[:, None, :] 
        ts_out = torch.mean(ts_out, dim=-1) 
        # (B, K, 3)
        ts = torch.stack([ts_in, ts_out, ts_unique], dim=-1)

        # (B, K, .)
        return mask, features, ts

    def forward(self, os, ds):
        """ find the accumulated densities and colors on the voxel grid given corresponding rays
        Args:
            os: Bx3 float tensor of ray origin
            ds: Bx3 float tensor of ray direction
        Return:
            color: BxNx3 float tensor of accumulated colors
            sigma: BxN float tensor of accumulated densities (zero for not hit voxels)
            mask: BxN bool tensor of hit indicator (used for rendering and reshaping)
            ts: BxN float tensor of distance to the ray origin
        """
        # (B, K, .)
        coarse_mask, coarse_features, coarse_ts = self.extract_features(os, ds)

        ts = []
        features = []
        mask = []
        if coarse_features is not None:
            mask.append(coarse_mask)
            features.append(coarse_features)
            ts.append(coarse_ts)

        if hasattr(self, 'fine_mask'):
            # (B, K', .)
            fine_mask, fine_features, fine_ts = self.extract_features(os, ds, extract_fine=True)
            if fine_features is not None:
                mask.append(fine_mask)
                features.append(fine_features)
                ts.append(fine_ts)

        if not features:
            return None,None,None,coarse_mask,None

        # (B, K+K')
        mask = torch.cat(mask, dim=1)

        # (B, K+K', .)
        features = torch.cat(features, dim=1)
        ts = torch.cat(ts, dim=1)

        # We will sort the features segments by the entry depth of intersections
        ts_in = ts[..., 0]

        # Batch indexing to sort the intersection features and their corresponding mask 
        # according to the depth ts.
        # The location of padding is trivial since they did not influence rendering.
        # (B, K+K', C)
        _, indices = ts_in.sort(dim=1)
        features = torch.gather(features, 1, 
                    indices.unsqueeze(-1).expand(-1, -1, features.shape[-1]))
        mask = torch.gather(mask, 1, indices) # (B, K+K')
        ts = torch.gather(ts, 1, 
                    indices.unsqueeze(-1).expand(-1, -1, ts.shape[-1]))

        # (N_valid, C)
        features_ = features[mask]
        
        # feature --> (density, feature)
        x = self.mlp1(features_)
        sigma_, beta_, x = x[:,0], x[:,1], x[:, 2:] # (N_valid, .)
        # TODO: https://github.com/bmild/nerf/issues/29
        beta_ = NF.softplus(beta_)
        

        # feature --> (feature, pose_enc(direction))
        x = torch.cat([
            x,
            self.view_enc(ds[torch.where(mask)[0]])
        ],dim=-1)
        
        # feature --> color
        color_ = self.mlp2(x)
        color_ = torch.sigmoid(color_)
        
        
        # set density and color to be zero for the locations corresponde to miss hits
        # (B, K+K', .)
        sigma = torch.zeros(*mask.shape, device=mask.device)
        sigma[mask] = sigma_
        color = torch.zeros(*mask.shape, 3, device=mask.device)
        color[mask] = color_
        beta = torch.zeros(*mask.shape, device=mask.device)
        beta[mask] = beta_
        return color, sigma, beta, mask, ts # (B, K+K', .)

    
    def render(self, color, sigma, beta, mask):
        """ alpha blending
        Args:
            color: (B, K, 3) float tensor of accumulated colors
            sigma: (B, K) float tensor of accumulated densities
            mask: (B, K) bool tensor of hit indicator
        Return:
            rgb: (B, 3) rendered pixels
            weight: (B, K) accumulated alphas
        """
        # TODO: why act here? maybe due to the sigma loss in train.py
        sigma = torch.relu(sigma)
        # fiter out the zero sigma while keep the shape
        alpha = 1-torch.exp(-sigma * mask) # (B, K)

        # 1, 1-alpha1, 1-alpha2, ...
        alpha_shifted = NF.pad(1-alpha[None,:,:-1], (1,0), value=1)[0]
        
        # color = ac + (1-a)ac + .... 
        weight = alpha * torch.cumprod(alpha_shifted,-1)
        color = (weight[:,:,None]*color).sum(1)

        # rendered beta
        uncert = (weight*beta).sum(1)
        uncert = uncert + self.beta_min
        
        if self.white_back: # whether to use white background
            color = color + (1 - weight.sum(1, keepdim=True))
        return color, weight, uncert


    """TODO: prune.py of fine model"""
    def decode(self, coord_in, coord_out, ds, mask):
        """ get rgb, density given ray entry, exit point
        Args:
          coord_in: (N_valid, 3)
          coord_out: (N_valid, 3)
          mask: (B, K)
        Return:
          (B, K, .)
        """
        if hasattr(self,'coarse_vertex_keys'):
            feature = integrate(self.coarse_vertex_keys, self.vertex_embedding, coord_in, coord_out)
        else:
            feature = integrate_mlp(self.mlp_voxel, self.voxel_num+1, self.voxel_dim, coord_in, coord_out)
            
        x = self.mlp1(feature)
        sigma_, beta_, x = x[:,0], x[:,1], x[:,2:] # (N_valid, .)
        beta_ = NF.softplus(beta_)

        x = torch.cat([
            x,
            self.view_enc(ds[torch.where(mask)[0]])
        ],dim=-1)

        color_ = self.mlp2(x)
        color_ = torch.sigmoid(color_)

        # (B, K)
        sigma = torch.zeros(*mask.shape, device=mask.device)
        sigma[mask] = sigma_ 
        color = torch.zeros(*mask.shape, 3, device=mask.device)
        color[mask] = color_
        beta = torch.zeros(*mask.shape, device=mask.device)
        beta[mask] = beta_
        return color, sigma, beta # (B, K, .)
