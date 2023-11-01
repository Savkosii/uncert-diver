import torch
import torch.nn as nn
import torch.nn.functional as NF

from tqdm import tqdm

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
        
        # whether to use implicit model (or implicit2explicit)
        if hparams.implicit:
            self.mlp_voxel = ImplicitMLP(hparams.implicit_network_depth, hparams.implicit_channels, hparams.implicit_skips,\
                             self.voxel_dim, hparams.implicit_point_encode)
        else:
            # random initialized explicit grid
            voxels = torch.randn(*[self.voxel_num+1]*3, self.voxel_dim)*1e-2 
            voxels = nn.Parameter(voxels, requires_grad=True)
            self.register_parameter('voxels',voxels)
            
        if not hparams.fine == 0: # coarse to fine model
            mask_voxel_num = int(self.voxel_num*self.mask_scale)
            # the default voxel mask (replaced according to the alpha map if 
            # it has been extracted by prune.py, see main() in train.py)
            self.register_parameter('voxel_mask',nn.Parameter(
                torch.zeros(mask_voxel_num,mask_voxel_num,mask_voxel_num,dtype=torch.bool),requires_grad=False))

            self.register_parameter('uncert_mask',nn.Parameter(
                torch.zeros(mask_voxel_num,mask_voxel_num,mask_voxel_num,dtype=torch.bool),requires_grad=False))

        ### Input dim: self.voxel_dim (default: 64 or 32)
        # feature -> (density, view_feature) 
        # mlp_out: f (64) + sigma (1) + beta (1)
        mlp_dim, mlp_depth, mlp_out = hparams.mlp_point
        self.mlp1 = mlp(self.voxel_dim, [mlp_dim]*mlp_depth, mlp_out)
        self.beta_min = 0.1
        
        # (view_feature, viewing dir) -> rgb
        self.view_enc = PositionalEncoding(hparams.dir_encode)
        mlp_dim, mlp_depth = hparams.mlp_view
        view_dim = hparams.dir_encode*2*3+3
        self.mlp2 = mlp(view_dim+mlp_out-2,[mlp_dim]*mlp_depth,3)
    
    def init_voxels(self, evaluate=True):
        """ initialize explicit voxel grid 
        Args:
            evaluate: whether to initialize the grid from implicit MLP
        """
        device = self.voxel_mask.device
        N = self.voxel_num+1
        
        if evaluate:
            Z,Y,X = torch.meshgrid(*[torch.arange(0,N)]*3) # (N, N, N) each
            P = torch.stack([Z,Y,X],dim=-1).float()/(N*0.5)-1.0 # (N, N, N, 3)

            voxels = []
            for i in tqdm(range(len(P))):
                voxel = self.mlp_voxel(P[i].to(device).reshape(-1,3)).reshape(N,N,self.voxel_dim)
                voxels.append(voxel)
            self.register_parameter('voxels',nn.Parameter(torch.stack(voxels,dim=0),requires_grad=True))
        else:
            self.register_parameter('voxels',nn.Parameter(torch.zeros(N,N,N,self.voxel_dim),requires_grad=True))
    
    def extract_features(self, os, ds):
        """ extract features given rays
        Args:
            os: Bx3 ray origin
            ds: Bx3 ray direction
        Return:
            mask: BxN bool tensor of intersection indicator
            features: BxNxC float tensor of integrated features
            ts: BxNxC float tensor of intersection distance (for depth map)
        """
        if hasattr(self, 'voxel_mask'): # with occupancy mask
            """
            masked_intersect (no grad)
            Args:
                - mask: (Nxmask_scale)**3  occupancy mask
                - mask_scale: relative scale of the occupancy mask in respect to the voxel grid
                  1/mask_scale: the number of voxels that share one mask
            Return:
                - coord: BxKx6 intersected entry + exit point
                - mask: BxK hit indicator (used for rendering)
                - ts: BxK distance from the ray origin
            """
            coord, mask, ts = masked_intersect(
                os.contiguous(), ds.contiguous(),
                self.xyzmin, self.xyzmax, int(self.voxel_num), self.voxel_size,
                self.voxel_mask.contiguous(), self.mask_scale)
            coord=coord[mask]
            coord_in = coord[:,:3]
            coord_out = coord[:,3:]
            # TODO: hack masked_intersect()
        else:
            """
            ray_voxel_intersect (no grad)
            Return:
                - coord: BxKx3 intersected points
                - mask: BxK hit indicator (used for rendering)
                - ts: BxK distance from the ray origin
            """
            coord, mask, ts = ray_voxel_intersect(
                os.contiguous(), ds.contiguous(), 
                self.xyzmin, self.xyzmax, int(self.voxel_num), self.voxel_size)
            
            ts = ts[:,:-1]
            coord = coord.clamp_min(0)
            mask = mask[:,:-1]&mask[:,1:]
            coord_in = coord[:,:-1][mask]
            coord_out = coord[:,1:][mask]

        if not mask.any(): # not hit
            return mask, None, None

        """
        voxels_to_resolve = [0]
        coord, mask, ts = masked_intersect(
            os.contiguous(), ds.contiguous(),
            self.xyzmin, self.xyzmax, int(self.voxel_num), self.voxel_size,
            mask, self.mask_scale)
        coord=coord[mask]
        coord_in = coord[:,:3]
        coord_out = coord[:,3:]
        """
            
            
        if hasattr(self,'voxels'): # check whether use explicit or implicit query
            features = integrate(self.voxels, coord_in, coord_out)
        else:
            features = integrate_mlp(self.mlp_voxel, self.voxel_num+1, self.voxel_dim, coord_in, coord_out)

        return mask, features, ts
    

    def decode(self, coord_in, coord_out, ds, mask):
        """ get rgb, density given ray entry, exit point
        Args:
          coord_in: (B*K, 3)
          coord_out: (B*K, 3)
          mask: (B, K), where K is the number of hit voxels
        Return:
          sigma: (B, K)
          color: (B, K, 3)
          beta: (B, K)
        """
        if hasattr(self,'voxels'):
            feature = integrate(self.voxels, coord_in, coord_out)
        else:
            feature = integrate_mlp(self.mlp_voxel, self.voxel_num+1, self.voxel_dim, coord_in, coord_out)
            
        B,M = mask.shape
        x = self.mlp1(feature)
        sigma_, beta_, x = x[:,0],x[:,1], x[:,2:]
        sigma_ = NF.softplus(sigma_)
        beta_ = NF.softplus(beta_)

        x = torch.cat([
            x,
            self.view_enc(ds[torch.where(mask)[0]])
        ],dim=-1)

        color_ = self.mlp2(x)
        color_ = torch.sigmoid(color_)

        sigma = torch.zeros(B,M,device=mask.device)
        sigma[mask] = sigma_
        color = torch.zeros(B,M,3,device=mask.device)
        color[mask] = color_
        beta = torch.zeros(B,M,device=mask.device)
        beta[mask] = beta_
        return color, sigma, beta
    
    def forward(self, os, ds):
        """ find the accumulated densities and colors on the voxel grid given corresponding rays
        Args:
            os: Bx3 float tensor of ray origin
            ds: Bx3 float tensor of ray direction
        Return:
            color: BxNx3 float tensor of accumulated colors
            sigma: BxN float tensor of accumulated densities
            mask: BxN bool tensor of hit indicator (used for rendering)
            ts: BxN float tensor of distance to the ray origin
        """
        mask, feature, ts = self.extract_features(os, ds)
        
        B,M = mask.shape
        if feature is None: # all the rays do not hit the volume
            return None,None,None,mask,None
        
        # feature --> (density, feature)
        x = self.mlp1(feature)
        sigma_, beta_, x= x[:,0],x[:,1],x[:, 2:]
        # TODO: https://github.com/bmild/nerf/issues/29
        sigma_ = NF.softplus(sigma_)
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
        sigma = torch.zeros(B,M,device=mask.device)
        sigma[mask] = sigma_
        color = torch.zeros(B,M,3,device=mask.device)
        color[mask] = color_

        beta = torch.zeros(B,M,device=mask.device)
        beta[mask] = beta_
        
        return color, sigma, beta, mask, ts
        
    
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
        
        # alpha = 1-exp(-sigma)
        alpha = 1-torch.exp(-sigma*mask)

        # 1, 1-alpha1, 1-alpha2, ...
        alpha_shifted = NF.pad(1-alpha[None,:,:-1], (1,0), value=1)[0]
        
        # color = ac + (1-a)ac + .... 
        weight = alpha * torch.cumprod(alpha_shifted,-1)
        rgb = (weight[:,:,None]*color).sum(1)

        # rendered beta
        uncert = (weight*beta).sum(1)
        uncert = uncert + self.beta_min
        
        if self.white_back: # whether to use white background
            rgb = rgb + (1-weight.sum(1,keepdim=True))
        return rgb, weight, uncert
