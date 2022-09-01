import logging
import torch.nn as nn
import torch as th
import numpy as np
import copy
from collections import OrderedDict

from .networks.branching import Branching
from .networks.fc import FC
from .networks.join import Join
from .networks import resnet
from .networks.multistep import MultiStepControl, MultiStepTrajectory, MultiStepControlCell, MultiStepTrajectoryCell, MultiStepTrajectoryGuidedControl


log = logging.getLogger(__name__)


class CoILICRA(nn.Module):

    def __init__(self, im_shape, input_states, acc_as_action,
                 value_as_supervision, action_distribution, dim_features_supervision, rl_ckpt=None,
                 freeze_value_head=False, freeze_action_head=False,
                 resnet_pretrain=True,
                 use_multi_step_control=True,
                 use_multi_step_waypoint=True,
                 use_trajectory_guided_control=True,
                 initial_hidden_zeros_control=True,
                 initial_hidden_zeros_trajectory=False,
                 perception_output_neurons=512,
                 measurements_neurons=[128, 128],
                 measurements_dropouts=[0.0, 0.0],
                 join_neurons=[512],
                 join_dropouts=[0.0],
                 speed_branch_neurons=[256, 256],
                 speed_branch_dropouts=[0.0, 0.5],
                 value_branch_neurons=[256, 256],
                 value_branch_dropouts=[0.0, 0.5],
                 number_of_branches=6,
                 branches_neurons=[256, 256],
                 branches_dropouts=[0.0, 0.5],
                 multi_step_neurons=[256, 256],
                 multi_step_dropouts=[0.0, 0.5],
                 number_of_steps_control=4,
                 number_of_steps_waypoint=4,
                 squash_outputs=True,
                 perception_net='resnet34',
                 ):
        super(CoILICRA, self).__init__()

        self._init_kwargs = copy.deepcopy(locals())
        del self._init_kwargs['self']
        del self._init_kwargs['__class__']

        if th.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.number_of_branches = number_of_branches
        self.use_multi_step_control = use_multi_step_control
        self.use_multi_step_waypoint = use_multi_step_waypoint
        self.use_trajectory_guided_control = use_trajectory_guided_control
        self.initial_hidden_zeros_control = initial_hidden_zeros_control
        self.initial_hidden_zeros_trajectory = initial_hidden_zeros_trajectory



        rl_state_dict = None
        if rl_ckpt is not None:
            try:
                rl_state_dict = th.load(rl_ckpt, map_location='cpu')['policy_state_dict']
                log.info(f'Load rl_ckpt: {rl_ckpt}')
            except:
                log.info(f'Unable to load rl_ckpt: {rl_ckpt}')

        # preception block
        self.dim_features_supervision = dim_features_supervision
        if dim_features_supervision > 0:
            join_neurons[-1] = dim_features_supervision

        self._video_input = 'video' in perception_net
        self.perception = resnet.get_model(perception_net, im_shape,
                                           num_classes=perception_output_neurons, pretrained=resnet_pretrain)

        # measurement block
        input_states_len = 0
        if 'speed' in input_states:
            input_states_len += 1
        if 'vec' in input_states:
            input_states_len += 2
        if 'cmd' in input_states:
            input_states_len += 6
        self.measurements = FC(params={'neurons': [input_states_len] + measurements_neurons,
                                       'dropouts': measurements_dropouts,
                                       'end_layer': None})

        self.attention_encoder_1 = FC(params={'neurons': [measurements_neurons[-1] + dim_features_supervision, int(np.prod(self.perception.attention_dims[1:]))],
                                            'dropouts': [0.0], 
                                            'end_layer': None})

        self.attention_encoder_2 = FC(params={'neurons': [self.perception.attention_dims[0] + measurements_neurons[-1], dim_features_supervision],
                                            'dropouts': [0.0], 
                                            'end_layer': None})


        # concat/join block
        self.join = Join(params={'after_process':
                                 FC(params={'neurons':
                                            [measurements_neurons[-1] + perception_output_neurons] + join_neurons,
                                            'dropouts': join_dropouts,
                                            'end_layer': None}),
                                 'mode': 'cat'})

        if squash_outputs:
            end_layer_speed = nn.Sigmoid
            end_layer_action = nn.Tanh
        else:
            end_layer_speed = None
            end_layer_action = None

        # speed branch
        self.speed_branch = FC(params={'neurons': [perception_output_neurons] + speed_branch_neurons + [1],
                                       'dropouts': speed_branch_dropouts + [0.0],
                                       'end_layer': end_layer_speed})



        # value branch
        self.value_as_supervision = value_as_supervision
        if value_as_supervision:
            self.value_branch = FC(params={'neurons': [join_neurons[-1]] + value_branch_neurons + [1],
                                           'dropouts': value_branch_dropouts + [0.0],
                                           'end_layer': None})
            if rl_state_dict is not None:
                self._load_state_dict(self.value_branch, rl_state_dict, 'value_head')
                log.info(f'Load rl_ckpt state dict for value_head.')
            if freeze_value_head:
                for param in self.value_branch.parameters():
                    param.requires_grad = False
                log.info('Freeze value head weights.')

        # action branches
        self.action_distribution = action_distribution
        assert action_distribution in ['beta', 'beta_shared', None]
        if action_distribution == 'beta':
            dim_out = 2
            mu_branch_vector = []
            sigma_branch_vector = []
            for i in range(number_of_branches):
                mu_branch_vector.append(FC(params={'neurons': [join_neurons[-1]] + branches_neurons + [dim_out],
                                                   'dropouts': branches_dropouts + [0.0],
                                                   'end_layer': nn.Softplus}))
                sigma_branch_vector.append(FC(params={'neurons': [join_neurons[-1]] + branches_neurons + [dim_out],
                                                      'dropouts': branches_dropouts + [0.0],
                                                      'end_layer': nn.Softplus}))
            self.mu_branches = Branching(mu_branch_vector)
            self.sigma_branches = Branching(sigma_branch_vector)
        elif action_distribution == 'beta_shared':
            # shared branches_neurons
            dim_out = 2
            mu_branch_vector = []
            sigma_branch_vector = []

            for i in range(number_of_branches):
                policy_head = FC(params={'neurons': [join_neurons[-1]] + branches_neurons,
                                         'dropouts': branches_dropouts,
                                         'end_layer': nn.ReLU})
                dist_mu = nn.Sequential(nn.Linear(branches_neurons[-1], dim_out), nn.Softplus())
                dist_sigma = nn.Sequential(nn.Linear(branches_neurons[-1], dim_out), nn.Softplus())

                if rl_state_dict is not None:
                    self._load_state_dict(policy_head, rl_state_dict, 'policy_head')
                    self._load_state_dict(dist_mu, rl_state_dict, 'dist_mu')
                    self._load_state_dict(dist_sigma, rl_state_dict, 'dist_sigma')
                    log.info(f'Load rl_ckpt state dict for policy_head, dist_mu, dist_sigma.')

                mu_branch_vector.append(nn.Sequential(policy_head, dist_mu))
                sigma_branch_vector.append(nn.Sequential(policy_head, dist_sigma))

            self.mu_branches = Branching(mu_branch_vector)
            self.sigma_branches = Branching(sigma_branch_vector)
            if freeze_action_head:
                for param in self.mu_branches.parameters():
                    param.requires_grad = False
                for param in self.sigma_branches.parameters():
                    param.requires_grad = False
                log.info('Freeze action head weights.')

        else:
            if acc_as_action:
                dim_out = 2
            else:
                dim_out = 3
            branch_fc_vector = []
            for i in range(number_of_branches):
                branch_fc_vector.append(FC(params={'neurons': [join_neurons[-1]] + branches_neurons + [dim_out],
                                                   'dropouts': branches_dropouts + [0.0],
                                                   'end_layer': end_layer_action}))
            self.branches = Branching(branch_fc_vector)

        if use_multi_step_waypoint:

            dim_out = 2 # Number of dimensions in a cartesian coordinate (x, y)
            waypoint_branch_vector = []
            for i in range(number_of_branches):
                waypoint_branch_vector.append(FC(params={'neurons': [join_neurons[-1]] + branches_neurons + [dim_out],
                                                   'dropouts': branches_dropouts + [0.0],
                                                   'end_layer': end_layer_action}))
            self.waypoint_branches = Branching(waypoint_branch_vector)

        # Multi-step action prediction
        self.number_of_steps_control = number_of_steps_control
        self.number_of_steps_waypoint = number_of_steps_waypoint


        if use_multi_step_control:

            multi_step_control = MultiStepControl(params={
            'recurrent_cell': nn.GRUCell,
            'input_size' : 4 + join_neurons[-1], # 4 comes from alpha-acc, alpha-steer, beta-acc, beta-steer
            'hidden_size' : join_neurons[-1],
            'encoder' : FC(params = {
                'neurons' : [join_neurons[-1]] + multi_step_neurons + [join_neurons[-1]],
                'dropouts' : multi_step_dropouts + [0.0],
                'end_layer' : None
            }
            ),
            'policy_head_mu' : self.mu_branches,
            'policy_head_sigma' : self.sigma_branches,
            'number_of_steps' : self.number_of_steps_control,
            'initial_hidden_zeros' : initial_hidden_zeros_control
            }
            )

        if use_multi_step_waypoint:

            multi_step_waypoint = MultiStepTrajectory(params={
            'recurrent_cell': nn.GRUCell,
            'input_size' : 2 + 2,#join_neurons[-1], # 4 comes from alpha-acc, alpha-steer, beta-acc, beta-steer
            'hidden_size' : join_neurons[-1],
            'policy_head_waypoint' : self.waypoint_branches,
            'number_of_steps' : self.number_of_steps_waypoint,
            'initial_hidden_zeros' : initial_hidden_zeros_trajectory
            }
            )
        
        if use_trajectory_guided_control:

            self.multi_step_control_cell = MultiStepControlCell(multi_step_control=multi_step_control)
            self.multi_step_waypoint_cell = MultiStepTrajectoryCell(multi_step_trajectory=multi_step_waypoint)

            self.multi_step_trajectory_guided_control = MultiStepTrajectoryGuidedControl(params={
                "multi_step_control_cell" : self.multi_step_control_cell,
                "multi_step_trajectory_cell" : self.multi_step_waypoint_cell,
                "attention_encoder_1": self.attention_encoder_1,
                "attention_encoder_2": self.attention_encoder_2
            }
            )

        else:

            self.multi_step_control = multi_step_control
            self.multi_step_waypoint = multi_step_waypoint


        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, im, state):
        '''
        im: (b, c, t, h, w) np.uint8
        state: (n,)
        '''
        if not self._video_input:
            b, c, t, h, w = im.shape
            im = im.view(b, c*t, h, w)

        """ ###### APPLY THE PERCEPTION MODULE """
        #all_layers = self.perception.get_layers_features(im)
        #for layer, layer_name in zip(all_layers, ["x0", "x1", "x2", "x3", "x4", "x5", "x"]):
        #    
        #    log.info(f"{layer_name} shape: {layer.shape}")
        
        # Feed to ResNet34, take the output of the avgpool and FC
        x, F = self.perception(im) # Image feature
        # log.info(f"x shape: {x.shape}")
        # log.info(f"F shape: {F.shape}")


        """ ###### APPLY THE MEASUREMENT MODULE """
        m = self.measurements(state)
        # log.info(f"m shape: {m.shape}")

        """ Join measurements and perception"""
        j_traj = self.join(x, m)
        # log.info(f"j_traj shape: {j_traj.shape}")

        # build outputs dict
        outputs = {'pred_speed': self.speed_branch(x)}

        if self.value_as_supervision:
            outputs['pred_value'] = self.value_branch(j_traj)

        waypoint = th.zeros((j_traj.shape[0], 2), dtype = j_traj.dtype, device=j_traj.device)
        # log.info(f"waypoint shape: {waypoint.shape}")

        if self.use_trajectory_guided_control:
            
            assert self.number_of_steps_control == self.number_of_steps_waypoint, "Number of steps for control and trajectory branch must be equal!"
            
            # log.info(f"Attention Dims: {self.perception.attention_dims}")
            initial_attention_map = th.softmax(self.attention_encoder_1(th.cat([j_traj, m], dim=1)),dim=1).view(-1, 1, *(self.perception.attention_dims[1:]))
            # log.info(f"initial_attention_map shape: {initial_attention_map.shape}")
            attended_features = th.sum(initial_attention_map * F, dim = (2, 3))
            # log.info(f"attended_features shape: {attended_features.shape}")
            j_control = self.attention_encoder_2(th.cat([attended_features, m], dim=1))
            # log.info(f"j_control shape: {j_control.shape}")

            mu = self.mu_branches(j_control)[0]
            # log.info(f"mu shape: {mu.shape}")
            sigma = self.sigma_branches(j_control)[0]
            # log.info(f"sigma shape: {sigma.shape}")

            pred_mu, pred_sigma, pred_waypoint, pred_j_control, pred_attention_map = self.multi_step_trajectory_guided_control(j_control, mu, sigma, j_traj, waypoint, F, state[:, 1:3], self.perception.attention_dims)
        


        else:

            if self.use_multi_step_control:

                pred_mu, pred_sigma, pred_j_control = self.multi_step_control(j_traj)

            if self.use_multi_step_waypoint:

                pred_waypoint = self.multi_step_waypoint(j_traj, state[:, 1:3])

        
        outputs['pred_j_control'] = pred_j_control
        outputs['pred_waypoint'] = pred_waypoint
        outputs['pred_attention_map'] = pred_attention_map


        if self.action_distribution is None:
            #outputs['action_branches'] = self.branches(j)
            raise NotImplementedError
        else:
            outputs['pred_mu'] = pred_mu
            outputs['pred_sigma'] = pred_sigma


        if self.dim_features_supervision > 0:
            outputs['pred_features_control'] = pred_j_control
            outputs['pred_features_trajectory'] = j_traj
        
        
        return outputs

    def forward_branch(self, command, im, state, waypoints = None):

        with th.no_grad():
            
            #log.info(f"Image Size: {im.shape}")
            #log.info(f"State Size: {state.shape}")

            im_tensor = im.unsqueeze(0).to(self.device)
            state_tensor = state.unsqueeze(0).to(self.device)

            #log.info(f"Image Size as Tensor: {im_tensor.shape}")
            #log.info(f"State Size as Tensor: {state_tensor.shape}")

            outputs = self.forward(im_tensor, state_tensor)
            #log.info(f"Outputs Keys: {outputs.keys()}")


            if self.action_distribution == 'beta' or self.action_distribution=='beta_shared':
                

                #log.info(f"Predicted Mu Shape: {outputs['pred_mu'].shape}")
                #log.info(f"Predicted Sigma Shape: {outputs['pred_sigma'].shape}")
                #log.info(f"Predicted Waypoint Shape: {outputs['pred_waypoint'].shape}")

                if waypoints is None:

                    action_control = self._get_action_beta(outputs['pred_mu'][:, 0, :], outputs['pred_sigma'][:, 0, :])
                    speed_error, angle_error = self._get_action_trajectory(outputs['pred_waypoint'], state_tensor[:, 0])

                else:

                    action_control = self._get_action_beta(outputs['pred_mu'][:, 0, :], outputs['pred_sigma'][:, 0, :])
                    speed_error, angle_error = self._get_action_trajectory(waypoints, state_tensor[:, 0])
                
                #action = self.extract_branch(action)

            else:

                action = self.extract_branch(outputs['action_branches'])

            action_control = action_control[0].cpu().numpy()
            
            action_trajectory = speed_error.item(), angle_error.item()


            #log.info(f"Action Control Shape: {action_control.shape}")
            #log.info(f"Action Speed Error Shape: {action_trajectory[0].shape}")
            #log.info(f"Action Angle Error Shape: {action_trajectory[1].shape}")

        return action_control, action_trajectory, outputs['pred_speed'].item(), outputs['pred_waypoint'][0].cpu().numpy(), outputs

    @staticmethod
    def extract_branch(action):
        '''
        action_branches: list, len=num_branches, (batch_size, action_dim)
        '''

        return action[:, 0], action[:, 1]

    @property
    def init_kwargs(self):
        return self._init_kwargs

    @classmethod
    def load(cls, path):
        if th.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        saved_variables = th.load(path, map_location=device)
        # Create policy object
        saved_variables['policy_init_kwargs']['resnet_pretrain'] = False
        saved_variables['policy_init_kwargs']['rl_ckpt'] = None
        model = cls(**saved_variables['policy_init_kwargs'])
        # Load weights
        log.info(f'load state dict : {path}')
        model.load_state_dict(saved_variables['policy_state_dict'])
        model.to(device)
        return model

    @staticmethod
    def _get_action_beta(alpha, beta):
        action_branches = []
        
        x = th.zeros_like(alpha)
        x[:, 1] += 0.5
        mask1 = (alpha > 1) & (beta > 1)
        x[mask1] = (alpha[mask1]-1)/(alpha[mask1]+beta[mask1]-2)

        mask2 = (alpha <= 1) & (beta > 1)
        x[mask2] = 0.0

        mask3 = (alpha > 1) & (beta <= 1)
        x[mask3] = 1.0

        # mean
        mask4 = (alpha <= 1) & (beta <= 1)
        x[mask4] = alpha[mask4]/(alpha[mask4]+beta[mask4])

        x = x * 2 - 1

        action = x
        return action

    @staticmethod
    def _get_action_trajectory(waypoints, speed):
        
        waypoints = waypoints.squeeze()

        delta_waypoints = th.stack([waypoints[k+1,:] - waypoints[k,:] for k in range(waypoints.shape[0]-1)], dim=0)

        #log.info(f"Delta Waypoints Shape: {delta_waypoints.shape}")

        delta_waypoints_norm = delta_waypoints.norm(dim = -1)
        
        #log.info(f"Delta Waypoints Norm Shape: {delta_waypoints_norm.shape}")
        delta_waypoints_norm_mean = delta_waypoints_norm.mean(dim = -1)

        #log.info(f"Delta Waypoints Norm Mean Shape: {delta_waypoints_norm_mean.shape}")

        delta_waypoints_angle = th.atan2(delta_waypoints[:,1], delta_waypoints[:,0])

        #log.info(f"Delta Waypoints Angle Shape: {delta_waypoints_angle.shape}")

        delta_waypoints_angle_mean = delta_waypoints_angle.mean(dim = -1)

        #log.info(f"Delta Waypoints Angle Mean Shape: {delta_waypoints_angle_mean.shape}")

        speed_error = delta_waypoints_norm_mean  - speed

        #log.info(f"Speed Error Shape: {speed_error.shape}")

        angle_error = delta_waypoints_angle_mean
        #log.info(f"Angle Error Shape: {angle_error.shape}")

        return speed_error, angle_error



    @staticmethod
    def _load_state_dict(il_net, rl_state_dict, key_word):
        rl_keys = [k for k in rl_state_dict.keys() if key_word in k]
        il_keys = il_net.state_dict().keys()
        assert len(rl_keys) == len(il_net.state_dict().keys()), f'mismatch number of layers loading {key_word}'
        new_state_dict = OrderedDict()
        for k_il, k_rl in zip(il_keys, rl_keys):
            new_state_dict[k_il] = rl_state_dict[k_rl]
        il_net.load_state_dict(new_state_dict)
