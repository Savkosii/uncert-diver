import torch
import torch_scatter
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt

import numpy as np
import math

from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

import sys
sys.path.append('..')
from configs.config import default_options
from model.diver import DIVeR

from utils.dataset import BlenderDataset,TanksDataset
from utils.ray_voxel_intersection import ray_voxel_intersect, masked_intersect

def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        for name, args in default_options.items():
            if(args['type'] == bool):
                parser.add_argument('--{}'.format(name), type=eval, choices=[True, False], default=str(args.get('default')))
            else:
                parser.add_argument('--{}'.format(name), **args)
        return parser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = add_model_specific_args(parser)
    hparams, _ = parser.parse_known_args()
    
    # add PROGRAM level args
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--thresh', type=float, default=1e-2) # voxel culling threshold
    parser.add_argument('--pad' , type=int, default=0) # whether to padd the boundary
    parser.add_argument('--device', type=int, required=False,default=None)
    parser.add_argument('--batch', type=int, default=4000)
    parser.add_argument('--bias_sampling', type=int, default=0) # whether bias sampling the fine_rays

    parser.set_defaults(resume=False)
    args = parser.parse_args()

    
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint_path)
    
    # load model
    state_dict = torch.load(checkpoint_path/'last.ckpt', map_location='cpu')['state_dict']
    weight = {}
    for k,v in state_dict.items():
        if 'model.' in k:
            weight[k.replace('model.', '')] = v

    model = DIVeR(hparams)
    if hparams.implicit:
        alpha_map_path = Path(args.coarse_path) / 'alpha_map.pt'
        beta_map_path = Path(args.coarse_path) / 'beta_map.pt'
        with torch.no_grad():
            model.init_voxels(False, hparams.thresh_a, alpha_map_path, hparams.thresh_beta, beta_map_path)
    model.load_state_dict(weight, strict=True)

    model.to(device)
    for p in model.parameters():
        p.requires_grad = False

    
    # load dataset
    dataset_name,dataset_path = hparams.dataset
    batch_size = args.batch
    if dataset_name == 'blender':
        dataset_fn = BlenderDataset
        dataset = dataset_fn(dataset_path,img_wh=hparams.im_shape[::-1], split='train')
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    elif dataset_name == 'tanks':
        dataset_fn = TanksDataset
        dataset = dataset_fn(dataset_path,img_wh=hparams.im_shape[::-1], split='extra')
        dataloader = dataset
    


    alpha_map_path = checkpoint_path / 'alpha_map.pt'
    beta_map_path = checkpoint_path / 'beta_map.pt'

    # extracting alpha map
    print('extracting alpha map')
    alpha_map = torch.zeros((model.voxel_num)**3,device=device)
    beta_map = torch.zeros((model.voxel_num)**3,device=device)

    for batch in tqdm(dataloader):
        rays = batch['rays'].to(device)
        for b_id in range(math.ceil(len(rays)*1.0/batch_size)):
            b_min = b_id*batch_size
            b_max = min((b_id+1)*batch_size,len(rays))
            xs, ds = rays[b_min:b_max,:3],rays[b_min:b_max,3:6]

            # TODO: uncertainty map is coarse currently, which should not be a desired behavior
            # use forward() instead of decode() to extract both coarse and fine features
            # and think about how to integrate the two into one map

            # coord: (B, K, 6), under the voxel basis; 
            # mask: (B, K), 
            # where K is the max number of hitted voxel in batch
            coord,mask,_ = masked_intersect(xs.contiguous(), ds.contiguous(),\
                                           model.xyzmin, model.xyzmax, int(model.voxel_num), model.voxel_size,\
                                           model.coarse_mask.contiguous(), model.mask_scale)
            if not mask.any():
                continue

            # note: bool array indexing only reserve the indexed dimension
            coord=coord[mask] # (N_valid, 6)  
            coord_in = coord[:,:3] 
            coord_out = coord[:,3:]

            # get accumulated alphas
            # (B, K, .)
            color, sigma, beta = model.decode(coord_in, coord_out, ds,mask)
            # weight: (B, K) accumulated alpha
            _, weight, _ = model.render(color, sigma, beta, mask)
            # beta = beta * weight

            weight = weight[mask] # (N_valid)
            beta = beta[mask]

            # accurate voxel corner calculation
            coord = torch.min((coord_in+1e-4).long(),(coord_out+1e-4).long()) # (N_valid, 3)

            # check if out of boundary
            bound_mask = ((coord>=args.pad) & (coord<=model.voxel_num-1-args.pad)).all(-1)
            coord = coord[bound_mask] 
            weight = weight[bound_mask]
            beta = beta[bound_mask]

            beta = beta * weight

            # flattened occupancy mask index
            # coord: the coordinate of voxel under the voxel basis (z, y, x)
            # index = z + N * y + N^2 * x
            coord = coord[:,0] + coord[:,1]*model.voxel_num + coord[:,2]*(model.voxel_num)**2
            # (B*M)

            # scatter-max to the occupancy map
            # alpha_map (N*N*N), where N is the number of voxels
            alpha_map = torch_scatter.scatter(
                    weight, coord, dim=0, out=alpha_map,reduce='max') 

            beta_map = torch_scatter.scatter(
                    beta, coord, dim=0, out=beta_map, reduce='max') 


    # alpha_map (N, N, N), where N is the number of voxels
    alpha_map = alpha_map.reshape(model.voxel_num,model.voxel_num, model.voxel_num)
    beta_map = beta_map.reshape(model.voxel_num,model.voxel_num, model.voxel_num)
    print(alpha_map.shape)
    torch.save(alpha_map.cpu(), alpha_map_path) # save in the model weight folder
    torch.save(beta_map.cpu(), beta_map_path) # save in the model weight folder

    occupancy_mask = alpha_map.to(device) > args.thresh
    print('alpha map masks {} voxels'.format(occupancy_mask.float().mean().item()))
    uncert_mask = beta_map.to(device) > 0.0135
    print('beta map mask {} voxels'.format(uncert_mask.float().mean().item()))

    coarse_mask = occupancy_mask & ~uncert_mask
    fine_mask = occupancy_mask & uncert_mask
    # Plot the voxels using matplotlib
    ax = plt.figure().add_subplot(projection='3d')
    ax.voxels(coarse_mask.cpu().numpy(), facecolors='red', edgecolor='k')
    ax.voxels(fine_mask.cpu().numpy(), facecolors='blue', edgecolor='k')
    plt.savefig('voxel_mask.png', format='png')

    if args.bias_sampling==0: # exit if not bias sampling
        exit(0)
    
          
    print('extract fine rays')
    fine_rays = []
    fine_rgbs = []
    
    batch_size *= 10
    # bias sampling the fine rays 
    if dataset_name == 'blender': # for nerf-synthetic, we directly store the sampled rays as .npz file in the weight folder
        dataset_fn = BlenderDataset
        fine_size = [800,800]
        dataset = dataset_fn(dataset_path,img_wh=fine_size[::-1], split='train')
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
        for batch in tqdm(dataloader):
            rays = batch['rays'].to(device)
            rgbs = batch['rgbs'].to(device)

            xs,ds = rays[:,:3],rays[:,3:6]
            _,mask,_ = masked_intersect(xs.contiguous(), ds.contiguous(),\
                                        model.xyzmin, model.xyzmax, int(model.voxel_num), model.voxel_size,\
                                        occupancy_mask.contiguous(), 1.0)
            valid_rays = mask.any(1)

            if valid_rays.any():
                fine_rays.append(rays[valid_rays].cpu())
                fine_rgbs.append(rgbs[valid_rays].cpu())

        fine_rays = torch.cat(fine_rays,0)
        fine_rgbs = torch.cat(fine_rgbs,0)

        print('{} rays perserved'.format(len(fine_rays)*1.0/len(dataset)))

        fine_path = checkpoint_path / 'fine_rays.npz'
        np.savez(fine_path, rays=fine_rays, rgbs=fine_rgbs)
    
    elif dataset_name == 'tanks': # for tnt, blendedmvs, we store the pixel(ray) mask
        if 'BlendedMVS' in dataset_path:
            fine_size = [576,768]
        elif 'TanksAndTemple' in dataset_path:
            fine_size = [1080,1920]
        
        dataset_fn = TanksDataset
        dataset = dataset_fn(dataset_path,img_wh=fine_size[::-1], split='extra')
        dataloader = dataset
        
        mask_path = checkpoint_path / 'masks'
        mask_path.mkdir(parents=True, exist_ok=True)
        
        idx = 0
        for batch in tqdm(dataloader):
            masks = []
            rays = batch['rays'].to(device)
            rgbs = batch['rgbs'].to(device)
            
            for b_id in range(math.ceil(len(rays)*1.0/batch_size)):
                b_min = b_id*batch_size
                b_max = min((b_id+1)*batch_size,len(rays))
                xs, ds = rays[b_min:b_max,:3],rays[b_min:b_max,3:6]
                
                _,mask,_ = masked_intersect(xs.contiguous(), ds.contiguous(),\
                                        model.xyzmin, model.xyzmax, int(model.voxel_num), model.voxel_size,\
                                        occupancy_mask.contiguous(), 1.0)
                mask = mask.any(1)
                masks.append(mask.cpu())
            
            masks = torch.cat(masks,-1).reshape(fine_size).float()
            save_image(masks, mask_path/'mask_{}.png'.format(idx)) # save in the mask folder inside model weight folder
            idx += 1
