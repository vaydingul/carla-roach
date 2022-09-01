import logging
from omegaconf import OmegaConf
import wandb
import copy
from collections import deque
import numpy as np
from carla_gym.utils.config_utils import load_entry_point
import math

class CilrsAgent():
    def __init__(self, path_to_conf_file='config_agent.yaml'):
        self._logger = logging.getLogger(__name__)
        self._render_dict = None
        self._steer_deque = deque(maxlen=10)

        self.setup(path_to_conf_file)

    def setup(self, path_to_conf_file):
        cfg = OmegaConf.load(path_to_conf_file)

        # load checkpoint from wandb
        rl_ckpt = None
        if cfg.wb_run_path is not None:
            api = wandb.Api()
            run = api.run(cfg.wb_run_path)
            all_ckpts = [f for f in run.files() if 'ckpt' in f.name]

            if cfg.wb_ckpt_step is None:
                f = max(all_ckpts, key=lambda x: int(x.name.split('_')[1].split('.')[0]))
                self._logger.info(f'Resume checkpoint latest {f.name}')
            else:
                wb_ckpt_step = int(cfg.wb_ckpt_step)
                f = min(all_ckpts, key=lambda x: abs(int(x.name.split('_')[1].split('.')[0]) - wb_ckpt_step))
                self._logger.info(f'Resume checkpoint closest to step {wb_ckpt_step}: {f.name}')

            f.download(replace=True)
            self._logger.info(f'Downloading {f.name}')
            self._ckpt = f.name

            run.file('config_agent.yaml').download(replace=True)
            self._logger.info(f'Downloading config_agent.yaml')
            cfg = OmegaConf.load('config_agent.yaml')
        else:
            self._ckpt = None
            # load rl state dict from wandb
            if cfg.rl_run_path is not None:
                api = wandb.Api()
                run = api.run(cfg.rl_run_path)
                all_ckpts = [f for f in run.files() if 'ckpt' in f.name]
                if cfg.rl_ckpt_step is None:
                    f = max(all_ckpts, key=lambda x: int(x.name.split('_')[1].split('.')[0]))
                    self._logger.info(f'Download rl checkpoint latest {f.name}')
                else:
                    rl_ckpt_step = int(cfg.rl_ckpt_step)
                    f = min(all_ckpts, key=lambda x: abs(int(x.name.split('_')[1].split('.')[0]) - rl_ckpt_step))
                    self._logger.info(f'Download rl checkpoint closest to step {rl_ckpt_step}: {f.name}')
                f.download(replace=True)
                self._logger.info(f'Downloading {f.name}')
                rl_ckpt = f.name

        cfg = OmegaConf.to_container(cfg)
        self._obs_configs = cfg['obs_configs']
        # for debug view
        self._obs_configs['route_plan'] = {'module': 'navigation.waypoint_plan', 'steps': 20}
        wrapper_class = load_entry_point(cfg['env_wrapper']['entry_point'])
        wrapper_kwargs = cfg['env_wrapper']['kwargs']
        self._env_wrapper = wrapper_class(**wrapper_kwargs)

        self._train_cfg = cfg['training']

        # prepare policy
        policy_class = load_entry_point(cfg['policy']['entry_point'])
        if self._ckpt is None:
            n_stack = len(self._env_wrapper.im_stack_idx)
            if self._env_wrapper.view_augmentation:
                im_shape = (3, n_stack, self._obs_configs['central_rgb']['height'],
                            self._obs_configs['central_rgb']['width']*3)
            else:
                im_shape = (3, n_stack, self._obs_configs['central_rgb']['height'],
                            self._obs_configs['central_rgb']['width'])
            self._policy = policy_class(im_shape, wrapper_kwargs['input_states'],
                                        wrapper_kwargs['acc_as_action'],
                                        wrapper_kwargs['value_as_supervision'],
                                        wrapper_kwargs['action_distribution'],
                                        wrapper_kwargs['dim_features_supervision'],
                                        rl_ckpt,
                                        **cfg['policy']['kwargs'])
        else:
            self._logger.info(f'Loading wandb checkpoint: {self._ckpt}')
            self._policy = policy_class.load(self._ckpt)
        self._policy = self._policy.eval()

        if self._env_wrapper.view_augmentation:
            self._im_queue = {
                'left_rgb': deque(maxlen=abs(min(self._env_wrapper.im_stack_idx))),
                'central_rgb': deque(maxlen=abs(min(self._env_wrapper.im_stack_idx))),
                'right_rgb': deque(maxlen=abs(min(self._env_wrapper.im_stack_idx)))
            }
        else:
            self._im_queue = {
                'central_rgb': deque(maxlen=abs(min(self._env_wrapper.im_stack_idx)))
            }

    def run_step(self, input_data, timestamp):
        input_data = copy.deepcopy(input_data)

        for im_key in self._im_queue.keys():
            if len(self._im_queue[im_key]) == 0:
                for _ in range(self._im_queue[im_key].maxlen):
                    self._im_queue[im_key].append(input_data[im_key])
            else:
                self._im_queue[im_key].append(input_data[im_key])

            input_data[im_key] = [copy.deepcopy(self._im_queue[im_key][i]) for i in self._env_wrapper.im_stack_idx]

        policy_input, command = self._env_wrapper.process_obs(input_data)

        actions_control, actions_trajectory, pred_speed, pred_waypoint, outputs = self._policy.forward_branch(command, im = policy_input['im'], state = policy_input['state'])
        
        control_action, control_action_array = self._env_wrapper.process_act_control(actions_control)
        control_trajectory, control_trajectory_array = self._env_wrapper.process_act_trajectory(actions_trajectory)
        
        if len(self._steer_deque) < 10:
            trajectory_specialized = True
            self._steer_deque.append(abs(control_action_array[1]))
        else:
            if np.sum(np.array(self._steer_deque) > 0.1) > 5:
                trajectory_specialized = False
                self._steer_deque.append(abs(control_trajectory_array[1]))
            else:
                trajectory_specialized = True
                self._steer_deque.append(abs(control_action_array[1]))
        
        control_fusion, control_fusion_array = self._env_wrapper.process_act_fusion(0.3, trajectory_specialized, control_action_array, control_trajectory_array)
        

        self._render_dict = {
            'policy_input': policy_input,
            'command': command,
            'action_control': control_action_array,
            'action_trajectory': control_trajectory_array,
            'action_fusion': control_fusion_array,
            'control_output': np.array(actions_control),
            'trajectory_output': np.array(actions_trajectory),
            'trajectory_specialized': trajectory_specialized,
            'pred_speed': pred_speed,
            'pred_waypoint': pred_waypoint,
            'pred_attention_map': outputs['pred_attention_map'][0, :, 0, :, :],
            'world_2_camera': policy_input['world_2_camera'],
            'camera_2_world': policy_input['camera_2_world'],
            'obs_configs': self._obs_configs,
            'birdview': input_data['birdview']['rendered'],
            'route_plan': input_data['route_plan'],
            'central_rgb': input_data['central_rgb'][-1]['data']
        }
        self._render_dict = copy.deepcopy(self._render_dict)
        self.supervision_dict = {}

        return control_fusion

    def reset(self, log_file_path):
        # logger
        self._logger.handlers = []
        self._logger.propagate = False
        self._logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(log_file_path, mode='w')
        fh.setLevel(logging.DEBUG)
        self._logger.addHandler(fh)

        for _, v in self._im_queue.items():
            v.clear()

    def learn(self, dataset_dir, train_epochs, reset_step=False):

        trainer_class = load_entry_point(self._train_cfg['entry_point'])
        if self._ckpt is None:
            trainer = trainer_class(self._policy, **self._train_cfg['kwargs'])
        else:
            trainer = trainer_class.load(self._policy, self._ckpt)

        trainer.learn(dataset_dir, int(train_epochs), self._env_wrapper, reset_step)

    def render(self, reward_debug, terminal_debug):
        '''
        test render, used in benchmark.py
        '''
        self._render_dict['reward_debug'] = reward_debug
        self._render_dict['terminal_debug'] = terminal_debug
        im_render = self._env_wrapper.im_render(self._render_dict)
        self._render_dict = None
        return im_render

    @property
    def obs_configs(self):
        return self._obs_configs
