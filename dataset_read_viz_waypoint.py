

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
K[0, 0] = K[1, 1] = FOCAL
K[0, 2] = WIDTH / 2.0
K[1, 2] = HEIGHTH / 2.0

STEP = 4
EARTH_RADIUS_EQUA = 6378137.0


a = 6378137
b = 6356752.3142
f = (a - b) / a
e_sq = f * (2 - f)


def geodetic_to_ecef(lat, lon, h):
    # (lat, lon) in WSG-84 degrees
    # h in meters
    lamb = math.radians(lat)
    phi = math.radians(lon)
    s = math.sin(lamb)
    N = a / math.sqrt(1 - e_sq * s * s)

    sin_lambda = math.sin(lamb)
    cos_lambda = math.cos(lamb)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)

    x = (h + N) * cos_lambda * cos_phi
    y = (h + N) * cos_lambda * sin_phi
    z = (h + (1 - e_sq) * N) * sin_lambda

    return x, y, z


def ecef_to_enu(x, y, z, lat0, lon0, h0):
    lamb = math.radians(lat0)
    phi = math.radians(lon0)
    s = math.sin(lamb)
    N = a / math.sqrt(1 - e_sq * s * s)

    sin_lambda = math.sin(lamb)
    cos_lambda = math.cos(lamb)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)

    x0 = (h0 + N) * cos_lambda * cos_phi
    y0 = (h0 + N) * cos_lambda * sin_phi
    z0 = (h0 + (1 - e_sq) * N) * sin_lambda

    xd = x - x0
    yd = y - y0
    zd = z - z0

    xEast = -sin_phi * xd + cos_phi * yd
    yNorth = -cos_phi * sin_lambda * xd - sin_lambda * sin_phi * yd + cos_lambda * zd
    zUp = cos_lambda * cos_phi * xd + cos_lambda * sin_phi * yd + sin_lambda * zd

    return xEast, yNorth, zUp


def geodetic_to_enu(lat, lon, h, lat_ref, lon_ref, h_ref):
    x, y, z = geodetic_to_ecef(lat, lon, h)

    return ecef_to_enu(x, y, z, lat_ref, lon_ref, h_ref)


def gnss_to_carla_coord(lat, lon, alt):
    x, y, z = geodetic_to_ecef(lat, lon, alt)
    carla_x, carla_y, carla_z = ecef_to_enu(x, y, z, 0.0, 0.0, 0.0)
    return carla_x, -carla_y, carla_z


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
    world_point_[:3] = gnss_to_carla_coord(*world_point)

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

    # Finally we can use our K matrix to do the actual 3D -> 2D.
    points_2d = np.dot(K, point_in_camera_coords)


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




# Fetch the h5 files
dataset_path = '/home/vaydingul20/Documents/Codes/ONE_EPISODE/expert/'
h5_files = [f for f in os.listdir(dataset_path) if f.endswith('.h5')]
f = os.path.join(dataset_path, np.random.choice(h5_files))
file_name = f.split('/')[-1]
f = h5py.File(f, 'r')


i = 0
while True:


    # Read the data
    img = f['step_%d/obs/central_rgb/data' % i][()]
    world_2_camera = f['step_%d/obs/central_rgb/world_2_camera' % i][()]

    waypoints = []

    for k in range(i + 1, i + 1 + STEP):
        world_point = f['step_%d/obs/gnss/gnss' % k][()]
        pixel_point = world_2_pixel(world_point, world_2_camera)
        waypoints.append(pixel_point)

    for waypoint in waypoints:
    
        if waypoint.shape[0] > 0:
            waypoint = waypoint.squeeze()
            print(waypoint)
            img = cv2.circle(img, (int(waypoint[0]), int(waypoint[1])), 1, (0, 0, 255), -1)
           

    cv2.imshow('animation', img)
 
    if cv2.waitKey(1) == ord('q'):
       
        # press q to terminate the loop
        cv2.destroyAllWindows()
        break

    time.sleep(0.01)
     
    i += 1
    

