default_options = {
    # dataset config
    'im_shape':{
        'type': int,
        'nargs': 2,
        'default': [800,800]
    },
    'batch_size':{
        'type': int,
        'default': 1024*8
    },
    'dataset': {
        'type': str,
        'nargs': 2,
        'default': ['blender','nerf_synthetic/drums']
    },
    'num_workers': {
        'type': int,
        'default': 6
    },


    # coarse to fine config
    'fine':{
        'type': int,
        'default': 1
    },
    'coarse_path':{
        'type': str,
        'default': 'checkpoints/drums_coarse'
    },
    'mask_scale':{
        'type': float,
        'default': 0.25
    },
    
    
    # training strategy config
    'implicit':{
        'type': bool,
        'default': True
    },
    'thresh_a':{
        'type': float,
        'default': 1e-2
    },
    

    # optimizer config
    'optimizer': {
        'type': str,
        'choices': ['SGD', 'Ranger', 'Adam'],
        'default': 'Adam'
    },
    'learning_rate': {
        'type': float,
        'default': 5e-4
    },
    'weight_decay': {
        'type': float,
        'default': 0
    },

    'scheduler_rate':{
        'type': float,
        'default': 0.5
    },
    'milestones':{
        'type': int,
        'nargs': '*',
        'default': [1000] # never used
    },
    
    
    # voxel grid config
    'grid_size':{
        'type': float,
        'default': 2.8
    },
    'voxel_num':{
        'type': int,
        'default': 256
    },
    'voxel_dim':{
        'type': int,
        'default': 32
    },
    
    
    # regularization loss config
    'l_s':{
        'type': float,
        'default': 1e-5
    },
    
    
    # implicit model config
    'implicit_network_depth':{
        'type': int,
        'default': 8
    },
    'implicit_channels':{
        'type': int,
        'default': 512
    },
    'implicit_skips':{
        'type': int,
        'nargs':'*',
        'default': [4]
    },
    'implicit_point_encode':{
        'type': int,
        'default': 10
    },
    
    
    # decoer mlp config
    'mlp_point': {
        'type': int,
        'nargs': 3,
        'default': [32,2,34] # dim f + sigma + beta = 34
    },
    'mlp_view': {
        'type': int,
        'nargs': 2,
        'default': [32,1]
    },
    'dir_encode':{
        'type': int,
        'default': 4
    },
    
    
    # rendering config
    'white_back': {
        'type': bool,
        'default': True
    }
}
