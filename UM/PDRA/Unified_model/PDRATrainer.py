import torch
import random
import numpy as np

from logging import getLogger

from PDRAEnv import PDRAEnv as Env
from PDRAModel import PDRAModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils.utils import *


class PDRATrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params):
        
        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # Dynamic vehicle configuration parameters
        self.use_dynamic_vehicles = trainer_params['use_dynamic_vehicles']
        self.vehicle_config_range = trainer_params['vehicle_config_range']

        # Reward normalization parameters and state variables
        self.reward_alpha = trainer_params.get('reward_alpha', 0.25)
        
        # EMA storage for options 2 and 3
        self.config_ema = {}
        self.config_ema_var = {}

        # Select normalization method (1, 2, 3)
        self.normalization_method = trainer_params.get('normalization_method', 3)
        
        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Main Components
        self.model = Model(**self.model_params)
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch']-1
            self.logger.info('Saved Model Loaded !!')

        # utility
        self.time_estimator = TimeEstimator()

    def sample_vehicle_config(self):
        
        num_vehicles_range = self.vehicle_config_range['num_vehicles']
        capacity_range = self.vehicle_config_range['vehicle_capacity']
        
        # Uniformly sample number of vehicles
        num_min = num_vehicles_range['min']
        num_max = num_vehicles_range['max']
        num_options = num_max - num_min + 1
        seed_num = np.random.rand()
        interval_num = 1.0 / num_options
        index_num = int(seed_num // interval_num)
        index_num = min(index_num, num_options - 1)
        sampled_num = num_min + index_num

        # Uniformly sample vehicle capacity
        cap_min = capacity_range['min']
        cap_max = capacity_range['max']
        cap_options = cap_max - cap_min + 1
        seed_cap = np.random.rand()
        interval_cap = 1.0 / cap_options
        index_cap = int(seed_cap // interval_cap)
        index_cap = min(index_cap, cap_options - 1)
        sampled_cap = cap_min + index_cap
        
        config = {'num_vehicles': sampled_num,'vehicle_capacity': sampled_cap}
        return config
        
    def run(self):
        self.time_estimator.reset(self.start_epoch)
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')

            # LR Decay
            self.scheduler.step()

            # Train
            train_score, train_loss = self._train_one_epoch(epoch)
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            # Logs & Checkpoint
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}: Time Est.: Elapsed[{}], Remain[{}]".format(
                              epoch, self.trainer_params['epochs'], elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']
            
            # Save latest images, every epoch
            if epoch > 1:
                self.logger.info("Saving log_image")
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                               self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                               self.result_log, labels=['train_loss'])

            # Save Model
            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {'epoch': epoch,
                                   'model_state_dict': self.model.state_dict(),
                                   'optimizer_state_dict': self.optimizer.state_dict(),
                                   'scheduler_state_dict': self.scheduler.state_dict(),
                                   'result_log': self.result_log.get_raw_data()}
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            # Save Image
            if all_done or (epoch % img_save_interval) == 0:
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                               self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                               self.result_log, labels=['train_loss'])

            # All-done announcement
            if all_done:
                self.logger.info(" *** Training Done *** ")
                self.logger.info("Now, printing log array...")
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):

        score_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_num_episode:

            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            # Sample vehicle configuration
            vehicle_config = self.sample_vehicle_config() 
            
            avg_score, avg_loss = self._train_one_batch(batch_size, vehicle_config)
            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)

            episode += batch_size

            # Log First 10 Batch, only at the first epoch
            if epoch == self.start_epoch:
                loop_cnt += 1
                if loop_cnt <= 10:
                    self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}'
                                     .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                             score_AM.avg, loss_AM.avg))

        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, loss_AM.avg))

        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, batch_size, vehicle_config=None):
        """Route to the corresponding training function based on selected normalization method"""
        if self.normalization_method == 1:
            return self._train_one_batch_option1(batch_size, vehicle_config)
        elif self.normalization_method == 2:
            return self._train_one_batch_option2(batch_size, vehicle_config)
        elif self.normalization_method == 3:
            return self._train_one_batch_option3(batch_size, vehicle_config)
        else:
            raise ValueError(f"Unsupported normalization method: {self.normalization_method}")


    def _train_one_batch_option1(self, batch_size, vehicle_config=None):
        """Option 1: Batch-wise standardization"""
        self.model.train()
        
        # Run POMO to get trajectory rewards
        self.env.load_problems(batch_size, vehicle_config=vehicle_config)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state, vehicle_config=vehicle_config)
        
        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        state, reward, done = self.env.pre_step()
        
        while not done:
            selected, prob = self.model(state)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
        
        # Batch-wise normalization
        reward_mean = reward.float().mean()
        reward_std = reward.float().std()
        epsilon = 1e-8
        
        # Z-score standardization (within batch)
        reward_normalized = (reward - reward_mean) / (reward_std + epsilon)
        
        # Calculate advantage
        advantage = reward_normalized - reward_normalized.float().mean(dim=1, keepdim=True)
        
        # Calculate policy gradient loss
        log_prob = prob_list.log().sum(dim=2)
        loss = - advantage * log_prob
        loss_mean = loss.mean()
        
        # Calculate score for evaluation (using original rewards)
        max_pomo_reward, _ = reward.max(dim=1)
        score_mean = max_pomo_reward.float().mean()
        
        # Backward pass
        self.model.zero_grad()
        loss_mean.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return score_mean.item(), loss_mean.item()
    
    def _train_one_batch_option2(self, batch_size, vehicle_config=None):
        """Option 2: Improved EMA normalization with variance tracking"""
        self.model.train()
        
        # Run POMO to get trajectory rewards
        self.env.load_problems(batch_size, vehicle_config=vehicle_config)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state, vehicle_config=vehicle_config)
        
        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        state, reward, done = self.env.pre_step()
        
        while not done:
            selected, prob = self.model(state)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
        
        # EMA mean + variance normalization
        # Generate configuration key
        num_veh = vehicle_config['num_vehicles']
        cap = vehicle_config['vehicle_capacity']
        attr_tw = 't' if reset_state.attribute_tw else 'f'
        attr_o = 't' if reset_state.attribute_o else 'f'
        config_key = f"v{num_veh}_c{cap:.2f}_tw{attr_tw}_o{attr_o}"
        
        # Calculate current batch mean and variance
        max_pomo_reward, _ = reward.max(dim=1)
        current_batch_mean = max_pomo_reward.float().mean().item()
        current_batch_var = max_pomo_reward.float().var().item()
        
        # Update EMA mean and variance
        if config_key not in self.config_ema:
            self.config_ema[config_key] = current_batch_mean
            self.config_ema_var[config_key] = max(current_batch_var, 1.0)
        else:
            self.config_ema[config_key] = (1 - self.reward_alpha) * self.config_ema[config_key] + self.reward_alpha * current_batch_mean
            self.config_ema_var[config_key] = (1 - self.reward_alpha) * self.config_ema_var[config_key] + self.reward_alpha * current_batch_var
        
        ema_mean = self.config_ema[config_key]
        ema_var = self.config_ema_var[config_key]
        epsilon = 1e-8
        
        # Z-score standardization using EMA mean and std
        ema_mean_tensor = torch.tensor(ema_mean, device=reward.device)
        ema_std_tensor = torch.sqrt(torch.tensor(ema_var, device=reward.device))
        reward_normalized = (reward - ema_mean_tensor) / (ema_std_tensor + epsilon)
        
        # Calculate advantage
        advantage = reward_normalized - reward_normalized.float().mean(dim=1, keepdim=True)
        
        # Calculate policy gradient loss
        log_prob = prob_list.log().sum(dim=2)
        loss = - advantage * log_prob
        loss_mean = loss.mean()
        
        # Calculate score for evaluation
        score_mean = max_pomo_reward.float().mean()
        
        # Backward pass
        self.model.zero_grad()
        loss_mean.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return score_mean.item(), loss_mean.item()
    
    def _train_one_batch_option3(self, batch_size, vehicle_config=None):
        """Option 3: EMA division normalization"""
        self.model.train()
        
        # Run POMO to get trajectory rewards
        self.env.load_problems(batch_size, vehicle_config=vehicle_config)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state, vehicle_config=vehicle_config)
        
        prob_list = torch.zeros(size=(batch_size, self.env.pomo_size, 0))
        state, reward, done = self.env.pre_step()
        
        while not done:
            selected, prob = self.model(state)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
        
        # EMA division normalization
        # Generate configuration key
        num_veh = vehicle_config['num_vehicles']
        cap = vehicle_config['vehicle_capacity']
        attr_tw = 't' if reset_state.attribute_tw else 'f'
        attr_o = 't' if reset_state.attribute_o else 'f'
        config_key = f"v{num_veh}_c{cap:.2f}_tw{attr_tw}_o{attr_o}"
        
        # Calculate current batch reward mean
        max_pomo_reward, _ = reward.max(dim=1)
        current_batch_mean = max_pomo_reward.float().mean().item()
        
        # Update EMA mean for this configuration
        if config_key not in self.config_ema:
            self.config_ema[config_key] = current_batch_mean
        else:
            self.config_ema[config_key] = (1 - self.reward_alpha) * self.config_ema[config_key] + self.reward_alpha * current_batch_mean
        ema_mean = self.config_ema[config_key]
        
        # Division normalization
        epsilon = 1e-8
        ema_tensor = torch.tensor(ema_mean, device=reward.device)
        reward_normalized = reward / (torch.abs(ema_tensor) + epsilon)
        
        # Calculate advantage
        advantage = reward_normalized - reward_normalized.float().mean(dim=1, keepdim=True)
        
        # Calculate policy gradient loss
        log_prob = prob_list.log().sum(dim=2)
        loss = - advantage * log_prob
        loss_mean = loss.mean()
        
        # Calculate score for evaluation
        score_mean = max_pomo_reward.float().mean()
        
        # Backward pass
        self.model.zero_grad()
        loss_mean.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return score_mean.item(), loss_mean.item()