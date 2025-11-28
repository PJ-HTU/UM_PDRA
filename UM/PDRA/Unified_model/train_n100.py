##########################################################################################
# Machine Environment Config

DEBUG_MODE = True
USE_CUDA = False
CUDA_DEVICE_NUM = 0

##########################################################################################
# Path Config

import os
import sys
__file__ = r"C:\Users\gongh\UM\PDRA\Unified_model\train_n100.py"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from PDRATrainer import PDRATrainer as Trainer


##########################################################################################
# parameters
# problem_type:
# If problem_type = 'unified': trained on 100% drone_l, 50% drone_ltw, 50% drone_lo
# problem_type can be drone_l, drone_ltw, drone_lo and their any combinations, e.g., drone_ltwo
# Where drone_l is for PDRA-Basic, drone_o is for PDRA-OR, drone_tw is for PDRA-TW in this paper

env_params = {
    'problem_type': 'unified', 
    'problem_size': 97,         # Total problem size: 97 customer nodes + 1 depot = 98 nodes
    'pomo_size': 47,            # pomo_size: nodes 1 to 47
    'original_node_count': 48,  # Original node count (0-47, reusable)
    'link_count': 50,           # Road connection count, new node count (48-97, non-reusable)   
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'softmax',
}

optimizer_params = {'optimizer': {'lr': 1e-4,
                                  'weight_decay': 1e-6},
                    'scheduler': {'milestones': [8001, 8051],
                                  'gamma': 0.1}}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 8100,
    'train_episodes': 100 * 1000,
    'train_batch_size': 64,
    
    # Dynamic vehicle number configuration
    'use_dynamic_vehicles': True,  # Set to False to disable dynamic vehicle numbers
    'vehicle_config_range': {'num_vehicles': {'min': 2, 'max': 4},           # Vehicle number range: 2-5 vehicles
                             'vehicle_capacity': {'min': 1, 'max': 3}   # Capacity range: 0.8-1.2
                            },
    'reward_alpha': 0.25,
    'normalization_method': 2,
    
    'logging': {'model_save_interval': 10,
                'img_save_interval': 10,
                'log_image_params_1': {'json_foldername': 'log_image_style',
                                       'filename': 'style_PDRA_20.json'},
                'log_image_params_2': {'json_foldername': 'log_image_style',
                                       'filename': 'style_loss_1.json'},},
    'model_load': {'enable': False,  # enable loading pre-trained model
                    # 'path': './result/saved_PDRA20_model',  # directory path of pre-trained model and log files saved.
                    # 'epoch': 2000,  # epoch version of pre-trained model to laod.
                    }}

logger_params = {'log_file': {'desc': 'train_PDRA_n100_dynamic_vehicles',  
                              'filename': 'run_log'}}


##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)

    # copy_all_src(trainer.result_folder)

    trainer.run()


def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 8
    trainer_params['train_batch_size'] = 4


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]

##########################################################################################

if __name__ == "__main__":
    main()