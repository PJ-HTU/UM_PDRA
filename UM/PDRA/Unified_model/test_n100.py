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
from PDRATester import PDRATester as Tester
##########################################################################################
# parameters
# problem_type can be drone_l, drone_ltw, drone_lo and their any combinations, e.g., drone_ltwo
# Where drone_l is for PDRA-Basic, drone_o is for PDRA-OR, drone_tw is for PDRA-TW in this paper

env_params = {
    'problem_type': "ltwo", # test problem type
    'problem_size': 197,         # Total problem size: 197 customer nodes + 1 depot = 198 nodes
    'pomo_size': 97,            # pomo_size: nodes 1 to 97
    'original_node_count': 98,  # Original node count (0-97, reusable)
    'link_count': 100,           # Road connection count, new node count (98-197, non-reusable)
    'num_vehicles': 4, # Test instance parameter
    'vehicle_capacity': 3,
}
    # 'problem_size': 97,         # Total problem size: 97 customer nodes + 1 depot = 98 nodes
    # 'pomo_size': 47,            # pomo_size: nodes 1 to 47
    # 'original_node_count': 48,  # Original node count (0-47, reusable)
    # 'link_count': 50,           # Road connection count, new node count (48-97, non-reusable)
# }
model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}
tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': r"C:\Users\gongh\UM\PDRA\Unified_model\result\saved_PDRA100_model",  # directory path of pre-trained model and log files saved.
        'epoch': 200,  # epoch version of pre-trained model to laod.
    },
    'test_episodes': 10,
    'test_batch_size': 16,
    'augmentation_enable': True,
    'aug_factor': 8,
    'aug_batch_size': 1,
    'test_data_load': {
        'enable': False,
        'filename': '../vrp100_test_seed1234.pt'
    },
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']
logger_params = {
    'log_file': {
        'desc': 'test_PDRA100',
        'filename': 'log.txt'
    }
}
##########################################################################################
# main
def main():
    if DEBUG_MODE:
        _set_debug_mode()
    create_logger(**logger_params)
    _print_config()
    tester = Tester(env_params=env_params,
                      model_params=model_params,
                      tester_params=tester_params)
    # copy_all_src(tester.result_folder)
    tester.run()
def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 2
def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]
##########################################################################################
if __name__ == "__main__":
    main()