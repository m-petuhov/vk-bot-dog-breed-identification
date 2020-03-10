class Config:

    routes = {
        'data': './dataset',
        'weights': './weights',
        'log': './logs',
        'tensorboard': './tensorboard'
    }

    transform_params = {
        'rescale_size': 224,
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225]
    }

    train_params = {
        'arch': 'DogNet',           # [Required]
        'num_workers': 1,           # [Required]
        'batch_size': 32,           # [Required]
        'start_epoch': 1,           # [Required]
        'epochs': 70,               # [Required]
        'resume_net': None,         # [Optional - None]
    }

    env_params = {
        'random_seed': 1  # [Optional - None]
    }


cfg = Config()
