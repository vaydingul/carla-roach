from carla_gym import CARLA_GYM_ROOT_DIR
from carla_gym.carla_multi_agent_env import CarlaMultiAgentEnv
from carla_gym.utils import config_utils
import json


class LAVEnv(CarlaMultiAgentEnv):
    def __init__(self, carla_map, host, port, seed, no_rendering, obs_configs, reward_configs, terminal_configs):

        all_tasks = self.build_all_tasks(carla_map)
        super().__init__(carla_map, host, port, seed, no_rendering,
                         obs_configs, reward_configs, terminal_configs, all_tasks)

    @staticmethod
    def build_all_tasks(carla_map):
        num_zombie_vehicles = {
            'Town02': 70,
            'Town05': 120
        }
        num_zombie_walkers = {
            'Town02': 70,
            'Town05': 120
        }

        description_folder = CARLA_GYM_ROOT_DIR / \
            'envs/scenario_descriptions/LAV'

        actor_configs_dict = json.load(
            open(description_folder / carla_map / 'actors.json'))
        route_descriptions_dict = config_utils.parse_routes_weather_file(
            description_folder / carla_map / 'routes.xml')

        all_tasks = []
        for route_id, route_description in route_descriptions_dict.items():
            carla_map = route_description['town']
            route_description["weather"]["lav"] = True
            task = {
                'weather': route_description['weather'],
                'description_folder': description_folder,
                'route_id': route_id,
                'num_zombie_vehicles': num_zombie_vehicles[carla_map],
                'num_zombie_walkers': num_zombie_walkers[carla_map],
                'ego_vehicles': {
                    'routes': route_description['ego_vehicles'],
                    'actors': actor_configs_dict['ego_vehicles'],
                },
                'scenario_actors': {}
            }
            all_tasks.append(task)

        return all_tasks
