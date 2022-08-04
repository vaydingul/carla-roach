

import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import carla
import math
import cv2
import time

FOV = 100
WIDTH = 900
HEIGHTH = 256
FOCAL = WIDTH / (2.0 * np.tan(FOV * np.pi / 360.0))

K = np.identity(3)
K[0, 0] = FOCAL
K[1, 1] = FOCAL
K[0, 2] = WIDTH / 2.0
K[1, 2] = HEIGHTH / 2.0

STEP = 4
EARTH_RADIUS_EQUA = 6378137.0

def gnss_2_world(gps):
    lat, lon, z = gps
    lat = float(lat)
    lon = float(lon)
    z = float(z)


    x = lon / 180.0 * (math.pi * EARTH_RADIUS_EQUA)

    y = -1.0 * math.log(math.tan((lat + 90.0) * math.pi / 360.0)) * EARTH_RADIUS_EQUA

    return np.array([x, y, z])


def world_2_pixel(world_point, world_2_camera):
    """
    Convert world coordinates to pixel coordinates.
    """
    world_point_ = np.ones((4, ))
    world_point_[:3] = gnss_2_world(world_point)

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
    # point_in_camera_coords = np.array([
    #             sensor_points[1],
    #             sensor_points[2] * -1,
    #             sensor_points[0]])
    point_in_camera_coords = sensor_points[:3]
    print(point_in_camera_coords)
    #point_in_camera_coords = np.array([sensor_points[2] * -1, sensor_points[0], sensor_points[1]]) 
    # point_in_camera_coords = np.array([0, 0, 1])
    # Finally we can use our K matrix to do the actual 3D -> 2D.
    #point_in_camera_coords[2] = 0
    points_2d = K @ point_in_camera_coords


    # Remember to normalize the x, y values by the 3rd value.
    points_2d = np.array([
        points_2d[0] / points_2d[2],
        points_2d[1] / points_2d[2],
        -points_2d[2]])

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




# Fetch the h5 files
dataset_path = '/home/vaydingul20/Documents/Codes/ONE_EPISODE/expert/'
h5_files = [f for f in os.listdir(dataset_path) if f.endswith('.h5')]
f = os.path.join(dataset_path, np.random.choice(h5_files))
file_name = f.split('/')[-1]
f = h5py.File(f, 'r')


i = 0
while True:


    # Read the data
    img = f[f'step_{int(i)}/obs/central_rgb/data'][()]
    camera_2_world = f[f'step_{int(i)}/obs/central_rgb/camera_2_world'][()]
    world_2_camera = f[f'step_{int(i)}/obs/central_rgb/world_2_camera'][()]

    camera_location = camera_2_world[:3, 3]


    waypoints = []

    for k in range(i + 1, i + 1 + STEP):
        world_point = f[f'step_{int(k)}/obs/gnss/gnss'][()]
        # print 3(gnss_2_world(world_point) + camera_2_world[:3, :3] @ np.array([-1.5, 0.0, 2.0]))
        # print(camera_location)
        pixel_point = world_2_pixel(world_point, world_2_camera)
        waypoints.append(pixel_point)

    for waypoint in waypoints:
    
        if waypoint.shape[0] > 0:
            waypoint = waypoint.squeeze()
            #print(waypoint)
            img = cv2.circle(img, (int(waypoint[0]), int(waypoint[1])), 5, (0, 0, 255), -1)
           

    cv2.imshow('animation', img)
 
    if cv2.waitKey(1) == ord('q'):
       
        # press q to terminate the loop
        cv2.destroyAllWindows()
        break

     
    i += 1
    

