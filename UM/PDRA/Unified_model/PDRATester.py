import torch
import numpy as np
import random

import os
from logging import getLogger

from PDRAEnv import PDRAEnv as Env
from PDRAModel import PDRAModel as Model

from utils.utils import *


class PDRATester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # Vehicle configuration parameters
        self.num_vehicles = env_params['num_vehicles']
        self.vehicle_capacity = env_params['vehicle_capacity']

        # result folder, logger
        self.logger = getLogger(name='test')
        self.result_folder = get_result_folder()

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print('Model loaded')
        
        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset()

        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()

        if self.tester_params['test_data_load']['enable']:
            self.env.use_saved_problems(self.tester_params['test_data_load']['filename'], self.device)

        test_num_episode = self.tester_params['test_episodes']
        episode = 0

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            # Vehicle configuration
            vehicle_config = {'num_vehicles': self.num_vehicles,'vehicle_capacity': self.vehicle_capacity} 
            
            score, aug_score = self._test_one_batch(batch_size, vehicle_config=vehicle_config)

            score_AM.update(score, batch_size)
            aug_score_AM.update(aug_score, batch_size)

            episode += batch_size

            # Logs
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score, aug_score))

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM.avg))

    def _test_one_batch(self, batch_size, vehicle_config=None):

        # Augmentation
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        # Vehicle configuration
        self.vehicle_config = vehicle_config
        
        # Ready
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(batch_size, vehicle_config = self.vehicle_config, aug_factor = aug_factor)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state, vehicle_config=vehicle_config)

        
            
        selection_node_list = torch.zeros(size=(batch_size*aug_factor, self.env.pomo_size, 0))
        
        # POMO Rollout
        state, reward, done = self.env.pre_step()
        while not done:
            selected, _ = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = self.env.step(selected)
            selection_node_list = torch.cat((selection_node_list, selected[:, :, None]), dim=2)

        global_max_reward, global_max_idx = reward.view(-1).max(dim=0)
        batch_idx = global_max_idx // self.env.pomo_size
        pomo_idx = global_max_idx % self.env.pomo_size
    
        # Print vehicle configuration and optimal path
        num_veh = vehicle_config['num_vehicles']
        cap = vehicle_config['vehicle_capacity']
        attr_tw = 't' if reset_state.attribute_tw else 'f'
        attr_o = 't' if reset_state.attribute_o else 'f'
        config_key = f"v{num_veh}_c{cap:.2f}_tw{attr_tw}_o{attr_o}"
  
        print(f"Drone configuration: {config_key}")
        print(f"Optimal path (batch_idx={batch_idx}, pomo_idx={pomo_idx}): Reward = {global_max_reward.item()}")
        print(f"Path node sequence: {selection_node_list[batch_idx][pomo_idx]}")

        # Return
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)

        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        no_aug_score = max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_score = max_aug_pomo_reward.float().mean()  # negative sign to make positive value

        return no_aug_score.item(), aug_score.item()