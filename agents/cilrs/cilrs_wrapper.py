import logging
import numpy as np
import carla
import cv2
import carla_gym.utils.transforms as trans_utils
import carla_gym.core.task_actor.common.navigation.route_manipulation as gps_util
from torchvision import transforms as T
import torch as th
from agents.cilrs.models.utils import controller
from agents.cilrs.models.utils import waypoint

log = logging.getLogger(__name__)

COLOR_BLACK = (0, 0, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_BLUE = (0, 0, 255)
COLOR_YELLOW = (255, 255, 0)
COLOR_ORANGE_0 = (252, 175, 62)


class CilrsWrapper():
    def __init__(self, acc_as_action, input_states, view_augmentation, im_mean=None, im_std=None, im_stack_idx=[-1],
                 value_as_supervision=False, action_distribution=None, dim_features_supervision=0, value_factor=1.0):
        self.acc_as_action = acc_as_action
        self.input_states = input_states
        self.view_augmentation = view_augmentation
        self.value_as_supervision = value_as_supervision
        self.action_distribution = action_distribution
        self.dim_features_supervision = dim_features_supervision

        # PID controllers for the vehicle control
        self.longitudinal_pid_controller = controller.PIDController([5, 0.025, 0.01])
        self.lateral_pid_controller = controller.PIDController([2.5, 0.05, 0.01])
        

        self.speed_factor = 12.0
        self.value_factor = value_factor
        if im_mean is None:
            im_mean = [0, 0, 0]
        if im_std is None:
            im_std = [1, 1, 1]
        self._im_transform = T.Compose([T.ToTensor(), T.Normalize(mean=im_mean, std=im_std)])
        self.im_stack_idx = im_stack_idx

    def process_obs(self, obs):
        ev_gps = obs['gnss']['gnss']
        # imu nan bug
        compass = 0.0 if np.isnan(obs['gnss']['imu'][-1]) else obs['gnss']['imu'][-1]

        gps_point = obs['gnss']['target_gps']
        target_vec_in_global = gps_util.gps_to_location(gps_point) - gps_util.gps_to_location(ev_gps)
        ref_rot_in_global = carla.Rotation(yaw=np.rad2deg(compass)-90.0)
        loc_in_ev = trans_utils.vec_global_to_ref(target_vec_in_global, ref_rot_in_global)

        # VOID = -1
        # LEFT = 1
        # RIGHT = 2
        # STRAIGHT = 3
        # LANEFOLLOW = 4
        # CHANGELANELEFT = 5
        # CHANGELANERIGHT = 6
        command = obs['gnss']['command'][0]
        if command < 0:
            command = 4
        command -= 1

        assert command in [0, 1, 2, 3, 4, 5]

        state_list = []
        if 'speed' in self.input_states:
            state_list.append(obs['speed']['forward_speed'][0]/self.speed_factor)
        if 'vec' in self.input_states:
            state_list.append(loc_in_ev.x)
            state_list.append(loc_in_ev.y)
        if 'cmd' in self.input_states:
            cmd_one_hot = [0] * 6
            cmd_one_hot[command] = 1
            state_list += cmd_one_hot

        im_list = []
        for i in range(len(obs['central_rgb'])):
            if self.view_augmentation:
                im = th.cat([
                    self._im_transform(obs['left_rgb'][i]['data']),
                    self._im_transform(obs['central_rgb'][i]['data']),
                    self._im_transform(obs['right_rgb'][i]['data'])
                ], dim=2)
            else:
                im = self._im_transform(obs['central_rgb'][i]['data'])

            im_list.append(im)

        # Waypoint ground-truth generation w.r.t. the ego-vehicle coordinate frame
        #waypoint_locations = obs['ego_vehicle_route']['route_locs']
        #waypoint_locations_world_frame = obs['ego_vehicle_route']['wp_locs']
        world_2_camera = obs['central_rgb'][0]['world_2_camera']
        camera_2_world = obs['central_rgb'][0]['camera_2_world']

        #log.info(f"Ego Vehicle Route Keys: {obs['ego_vehicle_route'].keys()}")
        #for key in obs['ego_vehicle_route'].keys():
        #    log.info(f"{key} Size: {obs['ego_vehicle_route'][key].shape}")
        

        ev_transform = obs['ego_vehicle_route']['ev_transform']
        ev_transform_inverse = obs['ego_vehicle_route']['ev_transform_inverse']
        waypoint_world_location = obs['ego_vehicle_route']['ev_wp']

        # 'waypoint_locations': th.tensor(waypoint_locations, dtype=th.float32),
        # 'waypoint_locations_world_frame': th.tensor(waypoint_locations_world_frame, dtype=th.float32),
        
        policy_input = {
            'im': th.stack(im_list, dim=1),
            'state': th.tensor(state_list, dtype=th.float32),
            'world_2_camera': th.tensor(world_2_camera, dtype=th.float32),
            'camera_2_world': th.tensor(camera_2_world, dtype=th.float32),
            'ev_transform': th.tensor(ev_transform, dtype=th.float32),
            'ev_transform_inverse': th.tensor(ev_transform_inverse, dtype=th.float64),
            'waypoint_location': th.tensor(waypoint_world_location, dtype=th.float64)
        }
        return policy_input, th.tensor([command], dtype=th.int8)

    def process_act_control(self, action):
        if self.acc_as_action:
            acc, steer = action.astype(np.float64)
            if acc >= 0.0:
                throttle = acc
                brake = 0.0
            else:
                throttle = 0.0
                brake = np.abs(acc)
        else:
            throttle, steer, brake = action.astype(np.float64)

        throttle = np.clip(throttle, 0, 0.75)
        steer = np.clip(steer, -1, 1)
        brake = np.clip(brake, 0, 1)

        #log.info(f"Throttle: {throttle}, Steer: {steer}, Brake: {brake}")
        
        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        return control, np.array([throttle, steer, brake])

    def process_act_trajectory(self, action):

        acc = self.longitudinal_pid_controller.step(action[0])
        steer = self.lateral_pid_controller.step(action[1])



        if acc >= 0.0:

            throttle = acc
            brake = 0.0

        else:

            throttle = 0.0
            brake = np.abs(acc)

        throttle = np.clip(throttle, 0, 0.75)        
        brake = np.clip(brake, 0, 1)
        steer = np.clip(steer, -1, 1)

        #log.info(f"Throttle: {throttle}, Steer: {steer}, Brake: {brake}")

        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)

        return control, np.array([throttle, steer, brake])

    def process_act_fusion(self, alpha, trajectory_specialized, control_action_array, control_trajectory_array):
        
        

        if trajectory_specialized:

            control = alpha * control_action_array + (1 - alpha) * control_trajectory_array

        else:

            control = alpha * control_trajectory_array + (1 - alpha) * control_action_array

        
        throttle = np.clip(control[0], 0, 0.75)
        brake = np.clip(control[2], 0, 1)
        steer = np.clip(control[1], -1, 1)

        control_fused = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)

        return control_fused, np.array([throttle, steer, brake])

        




    def process_supervision(self, supervision):
        '''
        supervision['speed']: in m/s
        supervision['action']: throttle, steer, brake in [0,1]
        ppo:
            supervision['action_mu']
            supervision['action_sigma']
            supervision['value']
            supervision['features']
        '''
        processed_supervision = {}
        # action distribution
        if self.action_distribution is not None:
            processed_supervision['action_mu'] = th.tensor(supervision['action_mu'])
            processed_supervision['action_sigma'] = th.tensor(supervision['action_sigma'])
        # deterministic action
        if self.acc_as_action:
            throttle, steer, brake = supervision['action']
            if brake > 0:
                acc = -1.0 * brake
            else:
                acc = throttle
            action = np.array([acc, steer], dtype=np.float32)
        else:
            action = supervision['action']
        processed_supervision['action'] = th.tensor(action)
        # if dataset includes advantage
        if 'advantage' in supervision:
            processed_supervision['advantage'] = th.tensor(supervision['advantage'])

        processed_supervision['speed'] = th.tensor(supervision['speed'])/self.speed_factor

        if self.value_as_supervision:
            value = th.tensor(supervision['value'])
            if len(value.shape) == 0:
                value = value.unsqueeze(0)
            processed_supervision['value'] = value / self.value_factor

        if self.dim_features_supervision > 0:
            processed_supervision['features'] = th.tensor(supervision['features'])

        return processed_supervision

    @staticmethod
    def im_render(render_dict):
        #im_birdview = CilrsWrapper.draw_route(render_dict)
        im_birdview = CilrsWrapper.draw_waypoint(render_dict, 'pred_waypoint', COLOR_BLUE)
        im_birdview = CilrsWrapper.draw_gnss(im_birdview, render_dict)
        im_rgb = render_dict['central_rgb']
        #im_rgb = waypoint.draw_waypoints(im_rgb, render_dict['gt_waypoint'], render_dict['world_2_camera'], 100, COLOR_RED)
        #im_rgb = waypoint.draw_waypoints(im_rgb, render_dict['pred_waypoint'], render_dict['world_2_camera'], 100, COLOR_BLUE)

        
        number_of_steps = render_dict['pred_attention_map'].shape[0]

        h = im_birdview.shape[0]
        h_rgb, w_rgb = im_rgb.shape[0:2]
        w = int(w_rgb*(h/h_rgb))
        im = np.zeros([h, w+h, 3], dtype=np.uint8)
        im[:h, :w] = cv2.resize(im_rgb, (w, h))
        im[:h, w:w+h] = im_birdview


        


        action_control_str = np.array2string(render_dict['action_control'], precision=2, separator=',', suppress_small=True)
        action_trajectory_str = np.array2string(np.array(render_dict['action_trajectory']), precision=2, separator=',', suppress_small=True)
        action_fusion_str = np.array2string(np.array(render_dict['action_fusion']), precision=2, separator=',', suppress_small=True)
        control_output_str = np.array2string(render_dict['control_output'], precision=2, separator=',', suppress_small=True)
        trajectory_output_str = np.array2string(render_dict['trajectory_output'], precision=2, separator=',', suppress_small=True)
        predicted_waypoint_str = np.array2string(np.array(render_dict['pred_waypoint']), precision=2, separator=',', suppress_small=True)
        state_str = np.array2string(render_dict['policy_input']['state'].numpy(),
                                    precision=2, separator=',', suppress_small=True)

        txt = f'a_control:{action_control_str}'
        im = cv2.putText(im, txt, (w, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        txt = f'a_trajectory:{action_trajectory_str}'
        im = cv2.putText(im, txt, (w,24), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        txt = f'a_fusion:{action_fusion_str}'
        im = cv2.putText(im, txt, (w,36), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        txt = f'c_output:{control_output_str}'
        im = cv2.putText(im, txt, (w,48), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        txt = f't_output:{trajectory_output_str}'
        im = cv2.putText(im, txt, (w,60), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        txt = f'pre_v:{render_dict["pred_speed"]}'
        im = cv2.putText(im, txt, (w, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        txt = f'pred_waypoint:{predicted_waypoint_str}'
        im = cv2.putText(im, txt, (0, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        txt = f'cmd: {render_dict["command"][0]} s{state_str}'
        im = cv2.putText(im, txt, (0, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        if render_dict["trajectory_specialized"]:
            txt = f'TRAJECTORY'
        else:
            txt = f"CONTROL"

        im = cv2.putText(im, txt, ((w//2) + 10, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)




        #for i, txt in enumerate(render_dict['reward_debug']['debug_texts'] +
        #                        render_dict['terminal_debug']['debug_texts']):
        #    im = cv2.putText(im, txt, (3, (i+1)*12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)


        im_unit_h, im_unit_w = im.shape[0:2]

        im_repeated = cv2.repeat(im, number_of_steps, 1)


        for k in range(number_of_steps):

            attention_map = render_dict['pred_attention_map'][k, :, :]
            attention_map = attention_map.cpu().numpy()
            attention_map = np.log(attention_map+1e-6)
            attention_map = ((attention_map - np.min(attention_map)) / (np.max(attention_map) - np.min(attention_map))) * 255.0
            attention_map = attention_map.astype(np.uint8)
            attention_map = cv2.applyColorMap(attention_map, cv2.COLORMAP_JET)#np.repeat(attention_map[:, :, np.newaxis], 3, axis=2)
            attention_map_resized = cv2.resize(attention_map, (w, h))
            
        # print(im[:h, :w].shape)
        # print(attention_map_resized.shape)

            offset_h = im_unit_h * k

            im_repeated[offset_h:offset_h+h, :w] = cv2.addWeighted(im_repeated[offset_h:offset_h+h, :w], 0.5, attention_map_resized, 0.5, 0)


        return im_repeated

    @staticmethod
    def draw_gnss(rendered_birdview, render_dict):
        if len(render_dict['policy_input']['state']) == 3:
            birdview_cfg = render_dict['obs_configs']['birdview']
            loc = render_dict['policy_input']['state'][1:].numpy()
            ev_xy = (int(birdview_cfg['width_in_pixels']/2),
                     int(birdview_cfg['width_in_pixels']-birdview_cfg['pixels_ev_to_bottom']))

            x = int(np.round(ev_xy[0] + loc[1]*birdview_cfg['pixels_per_meter']))
            y = int(np.round(ev_xy[1] - loc[0]*birdview_cfg['pixels_per_meter']))

            cmd = render_dict['command']

            if cmd == 0:
                # LEFT = 1
                color = COLOR_RED
            elif cmd == 1:
                # RIGHT = 2
                color = COLOR_GREEN
            elif cmd == 2:
                # STRAIGHT = 3
                color = COLOR_ORANGE_0
            elif cmd == 3:
                # LANEFOLLOW = 4
                color = COLOR_WHITE
            elif cmd == 4:
                # CHANGELANELEFT = 5
                color = COLOR_YELLOW
            elif cmd == 5:
                # CHANGELANERIGHT = 6
                color = COLOR_BLUE
            else:
                assert f'error {cmd}'

            cv2.line(rendered_birdview, ev_xy, (x, y), color=color, thickness=2)

        return rendered_birdview

    @staticmethod
    def draw_route(render_dict):
        birdview_cfg = render_dict['obs_configs']['birdview']
        rendered_birdview = render_dict['birdview']

        for i, loc in enumerate(render_dict['route_plan']['location']):
            x = int(np.round(birdview_cfg['width_in_pixels']/2 + loc[1]*birdview_cfg['pixels_per_meter']))
            y = int(np.round(birdview_cfg['width_in_pixels'] - birdview_cfg['pixels_ev_to_bottom']
                             - loc[0] * birdview_cfg['pixels_per_meter']))

            # VOID = 0
            # LEFT = 1
            # RIGHT = 2
            # STRAIGHT = 3
            # LANEFOLLOW = 4
            # CHANGELANELEFT = 5
            # CHANGELANERIGHT = 6
            cmd = render_dict['route_plan']['command'][i]

            if cmd == 1:
                # LEFT = 1
                color = COLOR_RED
            elif cmd == 2:
                # RIGHT = 2
                color = COLOR_GREEN
            elif cmd == 3:
                # STRAIGHT = 3
                color = COLOR_ORANGE_0
            elif cmd == 4:
                # LANEFOLLOW = 4
                color = COLOR_WHITE
            elif cmd == 5:
                # CHANGELANELEFT = 5
                color = COLOR_YELLOW
            elif cmd == 6:
                # CHANGELANERIGHT = 6
                color = COLOR_BLUE
            elif cmd == -1:
                # VOID = -1
                color = COLOR_BLACK
            else:
                print('error!!!!', cmd)
            cv2.circle(rendered_birdview, (x, y), 3, color, -1)

        return rendered_birdview


    @staticmethod
    def draw_waypoint(render_dict, waypoint_type, color):
        birdview_cfg = render_dict['obs_configs']['birdview']
        rendered_birdview = render_dict['birdview']

        for i, loc in enumerate(render_dict[waypoint_type]):
            x = int(np.round(birdview_cfg['width_in_pixels']/2 + loc[1]*birdview_cfg['pixels_per_meter']))
            y = int(np.round(birdview_cfg['width_in_pixels'] - birdview_cfg['pixels_ev_to_bottom']
                             - loc[0] * birdview_cfg['pixels_per_meter']))

            # VOID = 0
            # LEFT = 1
            # RIGHT = 2
            # STRAIGHT = 3
            # LANEFOLLOW = 4
            # CHANGELANELEFT = 5
            # CHANGELANERIGHT = 6
            # cmd = render_dict['route_plan']['command'][i]

            # if cmd == 1:
            #     # LEFT = 1
            #     color = COLOR_RED
            # elif cmd == 2:
            #     # RIGHT = 2
            #     color = COLOR_GREEN
            # elif cmd == 3:
            #     # STRAIGHT = 3
            #     color = COLOR_ORANGE_0
            # elif cmd == 4:
            #     # LANEFOLLOW = 4
            #     color = COLOR_WHITE
            # elif cmd == 5:
            #     # CHANGELANELEFT = 5
            #     color = COLOR_YELLOW
            # elif cmd == 6:
            #     # CHANGELANERIGHT = 6
            #     color = COLOR_BLUE
            # elif cmd == -1:
            #     # VOID = -1
            #     color = COLOR_BLACK
            # else:
            #     print('error!!!!', cmd)
            cv2.circle(rendered_birdview, (x, y), 3, color, -1)

        return rendered_birdview

