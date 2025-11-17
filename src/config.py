class FLoV2TConfig:
    MAX_PACKETS = 196
    PATCH_SIZE = 16
    IMAGE_SIZE = 224
    
    NUM_CLASSES = 8
    
    LORA_RANK = 4
    LORA_ALPHA = 8
    
    LAMBDA_REG = 0.1
    
    NUM_ROUNDS = 20
    LOCAL_EPOCHS = 5
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 0.01
    
    DEVICE = 'cuda'
    
    ATTACK_CATEGORIES = {
        0: 'Botnet',
        1: 'DoS-Slowloris',
        2: 'DoS-Goldeneye',
        3: 'DoS-Hulk',
        4: 'SSH-BruteForce',
        5: 'Web-SQL',
        6: 'Web-XSS',
        7: 'Web-Bruteforce'
    }
    
    NON_IID_DISTRIBUTIONS = {
        'CICIDS2017': {
            3: {
                0: [0, 1, 5],
                1: [2, 4],
                2: [3, 6, 7]
            },
            5: {
                0: [0, 7],
                1: [2, 3],
                2: [5, 6],
                3: [1],
                4: [4]
            }
        },
        'CICIDS2018': {
            3: {
                0: [0, 1, 5],
                1: [2, 4],
                2: [3, 6, 7]
            },
            5: {
                0: [0, 7],
                1: [2, 3],
                2: [5, 6],
                3: [1],
                4: [4]
            }
        }
    }
    
    IID_SAMPLE_DISTRIBUTION_3_CLIENTS = {
        0: 246,
        1: 684,
        2: 2494,
        3: 4660,
        4: 993,
        5: 5,
        6: 6,
        7: 30
    }
    
    IID_SAMPLE_DISTRIBUTION_5_CLIENTS = {
        0: 148,
        1: 410,
        2: 1496,
        3: 2796,
        4: 596,
        5: 3,
        6: 4,
        7: 18
    }
    
    @classmethod
    def get_config_dict(cls):
        return {
            'max_packets': cls.MAX_PACKETS,
            'patch_size': cls.PATCH_SIZE,
            'image_size': cls.IMAGE_SIZE,
            'num_classes': cls.NUM_CLASSES,
            'lora_rank': cls.LORA_RANK,
            'lora_alpha': cls.LORA_ALPHA,
            'lambda_reg': cls.LAMBDA_REG,
            'num_rounds': cls.NUM_ROUNDS,
            'local_epochs': cls.LOCAL_EPOCHS,
            'batch_size': cls.BATCH_SIZE,
            'learning_rate': cls.LEARNING_RATE,
            'weight_decay': cls.WEIGHT_DECAY,
            'device': cls.DEVICE
        }
    
    @classmethod
    def get_non_iid_distribution(cls, dataset: str, num_clients: int):
        if dataset in cls.NON_IID_DISTRIBUTIONS and num_clients in cls.NON_IID_DISTRIBUTIONS[dataset]:
            return cls.NON_IID_DISTRIBUTIONS[dataset][num_clients]
        else:
            raise ValueError(f"No non-IID distribution defined for dataset '{dataset}' with {num_clients} clients")
    
    @classmethod
    def get_iid_sample_distribution(cls, num_clients: int):
        if num_clients == 3:
            return cls.IID_SAMPLE_DISTRIBUTION_3_CLIENTS
        elif num_clients == 5:
            return cls.IID_SAMPLE_DISTRIBUTION_5_CLIENTS
        else:
            return None
