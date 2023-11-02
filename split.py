import torch
import torch_scatter
from torch.utils.data import DataLoader
from torchvision.utils import save_image

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
        with torch.no_grad():
            model.init_voxels(False)
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
    

    uncert_map_path = checkpoint_path / 'uncert_map.pt'

    if True: #not alpha_map_path.exists():
        # extracting alpha map
        print('extracting uncert map')
        uncert_map = torch.zeros((model.voxel_num)**3,device=device)

        for batch in tqdm(dataloader):
            rays = batch['rays'].to(device)
            for b_id in range(math.ceil(len(rays)*1.0/batch_size)):
                b_min = b_id*batch_size
                b_max = min((b_id+1)*batch_size,len(rays))
                xs, ds = rays[b_min:b_max,:3],rays[b_min:b_max,3:6]

                # perform ray-voxel intersection
                if hasattr(model, 'voxel_mask'):
                    # coord: (B, K, 6), under the voxel basis; 
                    # mask: (B, K), 
                    # where K is the max number of hitted voxel in batch
                    coord,mask,_ = masked_intersect(xs.contiguous(), ds.contiguous(),\
                                                   model.xyzmin, model.xyzmax, int(model.voxel_num), model.voxel_size,\
                                                   model.voxel_mask.contiguous(), model.mask_scale)
                    if not mask.any():
                        continue
                    # note: bool array indexing only reserve the indexed dimension
                    coord=coord[mask] # (B*K, 6)  
                    coord_in = coord[:,:3] # (B*K, 3)
                    coord_out = coord[:,3:] # (B*K, 3)
                else:
                    # coord: (B, K, 3); mask: (B, K), 
                    # where K is the max number of hit points in batch
                    coord, mask, _ = ray_voxel_intersect(xs.contiguous(), ds.contiguous(), model.xyzmin, model.xyzmax, int(model.voxel_num),model.voxel_size)
                    if not mask.any():
                        continue
                    # only hit both side of the voxel is regarded as hit
                    mask = mask[:,:-1]&mask[:,1:] # (B, K-1)
                    # where K-1 is the number of hit voxels
                    coord_in = coord[:,:-1][mask] # (B*(K-1), 3)
                    coord_out = coord[:,1:][mask] # (B*(K-1), 3)

                # get accumulated alphas
                # color: (B, K, 3)
                color, sigma, beta = model.decode(coord_in, coord_out, ds,mask)
                # weight: (B, K) accumulated alpha
                _, _, uncert = model.render(color, sigma, beta, mask)
                uncert = uncert[mask]

                # accurate voxel corner calculation
                coord = torch.min((coord_in+1e-4).long(),(coord_out+1e-4).long()) # (B*K, 3)

                # check if out of boundary
                # (B*M)
                bound_mask = ((coord>=args.pad) & (coord<=model.voxel_num-1-args.pad)).all(-1)
                coord = coord[bound_mask] # (B*M)
                uncert = uncert[bound_mask]

                # flattened occupancy mask index
                # coord: the coordinate of voxel under the voxel basis
                # coord = (x, y, z), n = x + N * y + N^2 * z
                coord = coord[:,0] + coord[:,1]*model.voxel_num + coord[:,2]*(model.voxel_num)**2
                # (B*M)

                uncert_map = torch_scatter.scatter(
                        uncert, coord, dim=0, out=uncert_map,reduce='max') 

        # (N, N, N), where N is the number of voxels
        uncert_map = uncert_map.reshape(model.voxel_num,model.voxel_num, model.voxel_num)
        torch.save(uncert_map.cpu(), uncert_map_path)

        
    uncert_voxel_mask = uncert_map.to(device) > 0.5
    print('{} voxel to split'.format(uncert_voxel_mask.float().mean().item()))
