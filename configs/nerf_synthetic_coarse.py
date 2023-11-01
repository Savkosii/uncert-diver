default_options = {
    # dataset config
    'im_shape':{
        'type': int,
        'nargs': 2,
        'default': [200,200]
    },
    'batch_size':{
        'type': int,
        'default': 1024
    },
    'dataset': {
        'type': str,
        'nargs': 2,
        'default': ['blender','path_to_scene']
    },
    'num_workers': {
        'type': int,
        'default': 6
    },


    # coarse to fine config
    'fine':{
        'type': int,
        'default': 0
    },
    'coarse_path':{
        'type': str,
        'default': None
    },
    # The number of voxels that share one mask are 1/mask_scale
    'mask_scale':{
        'type': float,
        'default': 1.0
    },
    
    
    # training strategy config
    'implicit':{
        'type': bool,
        'default': False
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
        'default': 1e-3
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
    # canvas size at one dimension, same as the fine model
    'grid_size':{
        'type': float,
        'default': 2.8
    },
    # voxel_size = grid_size / voxel_num, 4x as the fine model
    'voxel_num':{
        'type': int,
        'default': 64
    },
    # voxel feature dimension
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
        'default': [64,1,66] # f (64) + sigma (1) + beta (1)
    },
    'mlp_view': {
        'type': int,
        'nargs': 2,
        'default': [64,1]
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
