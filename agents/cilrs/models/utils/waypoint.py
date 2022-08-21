import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import carla
import math
import cv2
from cv2 import VideoWriter, VideoWriter_fourcc
import time
import torch
FOV = 100
WIDTH = 900
HEIGHTH = 256
FOCAL = WIDTH / (2.0 * np.tan(FOV * np.pi / 360.0))

FPS = 20

K = np.identity(3)
K[0, 0] = FOCAL
K[1, 1] = FOCAL
K[0, 2] = WIDTH / 2.0
K[1, 2] = HEIGHTH / 2.0

OFFSET = 0
STEP = 4  # 4
STRIDE = 1

THICKNESS_MAX = 5
THICKNESS_MIN = 1

COLOR_MAX = 255
COLOR_MIN = 50


EARTH_RADIUS_EQUA = 6378137.0


def world_2_ev(world_point, world_2_ev):

    world_point_ = torch.ones((4,), dtype=torch.float64)
    world_point_[:3] = world_point

    world_point_ = world_point_.unsqueeze(1)
    # Transform the points from world space to camera space.
    ev_point = torch.matmul(world_2_ev, world_point_)
    ev_point = ev_point.squeeze(1)
    return ev_point[:2].float()


def gnss_2_world(gps):

    lat, lon, z = gps
    lat = float(lat)
    lon = float(lon)
    z = float(z)

    x = lon / 180.0 * (math.pi * EARTH_RADIUS_EQUA)

    y = -1.0 * math.log(math.tan((lat + 90.0) * math.pi /
                                 360.0)) * EARTH_RADIUS_EQUA

    return np.array([x, y, z])


def world_2_pixel(world_point, world_2_camera):
    """
    Convert world coordinates to pixel coordinates.
    """
    world_point_ = np.ones((4, ))
    world_point_[:3] = world_point

    # Transform the points from world space to camera space.
    sensor_points = np.dot(world_2_camera, world_point_)

    # New we must change from UE4's coordinate system to an "standard"
    # camera coordinate system (the same used by OpenCV):

    # ^ z                       . z
    # |                        /
    # |              to:      +-------> x
    # | . x                   |
    # |/                      |
    # +-------> y             v y

    # This can be achieved by multiplying by the following matrix:
    # [[ 0,  1,  0 ],
    #  [ 0,  0, -1 ],
    #  [ 1,  0,  0 ]]

    # Or, in this case, is the same as swapping:
    # (x, y ,z) -> (y, -z, x)
    point_in_camera_coords = np.array([
        sensor_points[1],
        sensor_points[2] * -1,
        sensor_points[0]])
    # print(point_in_camera_coords[2])
    point_in_camera_coords[0] -= 0
    point_in_camera_coords[1] -= 1.7
    # point_in_camera_coords[1] -= 1
    # point_in_camera_coords[2] += 5
    # Finally we can use our K matrix to do the actual 3D -> 2D.
    points_2d = K @ point_in_camera_coords

    # Remember to normalize the x, y values by the 3rd value.
    points_2d = np.array([
        points_2d[0] / points_2d[2],
        points_2d[1] / points_2d[2],
        points_2d[2]])

    # At this point, points_2d[0, :] contains all the x and points_2d[1, :]
    # contains all the y values of our points. In order to properly
    # visualize everything on a screen, the points that are out of the screen
    # must be discarded, the same with points behind the camera projection plane.
    points_2d = points_2d.T
    points_in_canvas_mask = \
        (points_2d[0] > 0.0) & (points_2d[0] < WIDTH) & \
        (points_2d[1] > 0.0) & (points_2d[1] < HEIGHTH) & \
        (points_2d[2] > 0.0)
    pixel_points = points_2d[points_in_canvas_mask]

    return pixel_points


def main(dataset_path, episode):

    h5_file = os.path.join(dataset_path, f'{episode:04}.h5')
    f = h5py.File(h5_file, 'r')

    fourcc = VideoWriter_fourcc(*'mp4v')
    video=VideoWriter('./wp-ground-truth.mp4', fourcc,
                      float(FPS), (WIDTH//2, HEIGHTH//2))

    i=0
    while i < 3000:

        # Read the data
        try:
            img=f[f'step_{int(i)}/obs/central_rgb/data'][()]
        except KeyError:
            break
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        camera_2_world=f[f'step_{int(i)}/obs/central_rgb/camera_2_world'][()]
        world_2_camera=f[f'step_{int(i)}/obs/central_rgb/world_2_camera'][()]
        ev_transform=f[f'step_{int(i)}/obs/ego_vehicle_route/ev_transform'][()]
        ev_transform_inverse=f[f'step_{int(i)}/obs/ego_vehicle_route/ev_transform_inverse'][()]


        waypoints=[]

        for k in range(i + OFFSET, i + OFFSET + STEP * STRIDE, STRIDE):

            wp_loc=f[f'step_{int(k)}/obs/ego_vehicle_route/ev_wp'][()]
            print(world_2_ev(wp_loc, ev_transform_inverse))
            pixel_point=world_2_pixel(wp_loc, world_2_camera)
            waypoints.append(pixel_point)

        thickness=np.linspace(
            THICKNESS_MIN, THICKNESS_MAX, len(waypoints))[::-1]
        color=np.linspace(COLOR_MIN, COLOR_MAX, len(waypoints))[::-1]

        for (ix, waypoint) in enumerate(waypoints):

            if waypoint.shape[0] > 0:
                waypoint=waypoint.squeeze()
                img=cv2.circle(img, (int(waypoint[0]), int(waypoint[1])), int(
                    thickness[ix]), (0, 0, int(color[ix])), -1)

        cv2.imshow('Waypoint Animation', img)

        if cv2.waitKey(1) == ord('q'):

            # press q to terminate the loop
            cv2.destroyAllWindows()
            break

        video.write(cv2.resize(img, (WIDTH//2, HEIGHTH//2)))
        time.sleep(0.01)
        i += 1

    video.release()


if __name__ == '__main__':

    # Fetch the h5 files
    dataset_path='/home/vaydingul/Documents/Codes/4-episode/expert/'
    episode=0
    main(dataset_path, episode)
