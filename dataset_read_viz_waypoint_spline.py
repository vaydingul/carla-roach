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
from scipy import signal

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

OFFSET = 10
STEP = 50  # 4
STRIDE = 1
NUM_WAYPOINTS = 4
WINDOW_LEN = STEP // 2
THICKNESS_MAX = 5
THICKNESS_MIN = 5

COLOR_MAX = 255
COLOR_MIN = 50


EARTH_RADIUS_EQUA = 6378137.0


def noisy_2_smooth(points, window_len = WINDOW_LEN, degree = 2, num_points = None):
    """
    Convert noisy points to spline.
    """
    
    if window_len % 2 == 0:
        window_len += 1

    y = np.vstack(points)

    y_smooth = signal.savgol_filter(y, window_len, degree, axis = 0)

    if num_points is not None:
        indices = np.linspace(0, y_smooth.shape[0] - 1, num_points)
        y_smooth = y_smooth[indices.astype(int)]    

    return y_smooth
    

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
    point_in_camera_coords = np.array([
        sensor_points[1],
        sensor_points[2] * -1,
        sensor_points[0]])

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

    return pixel_points, point_in_camera_coords


def main(dataset_path, episode):

    h5_file = os.path.join(dataset_path, f'{episode:04}.h5')
    f = h5py.File(h5_file, 'r')

    fourcc = VideoWriter_fourcc(*'mp4v')
    video = VideoWriter('./wp-gnss-smooth.mp4', fourcc, float(FPS), (WIDTH//2, HEIGHTH//2))

    i = 0
    while i < 3000:

        # Read the data
        try:
            img = f[f'step_{int(i)}/obs/central_rgb/data'][()]
        except KeyError:
            break
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        camera_2_world = f[f'step_{int(i)}/obs/central_rgb/camera_2_world'][()]
        world_2_camera = f[f'step_{int(i)}/obs/central_rgb/world_2_camera'][()]

        camera_location = camera_2_world[:3, 3]
        ego_point = f[f'step_{int(i)}/obs/gnss/gnss'][()]

        waypoints = []
        points_in_camera_coords = []

        for k in range(i + OFFSET, i + OFFSET + STEP * STRIDE, STRIDE):

            try:
                world_point = f[f'step_{int(k)}/obs/gnss/gnss'][()]
            except KeyError:
                break
            pixel_point, point_in_camera_coords = world_2_pixel(world_point, world_2_camera)
            if pixel_point.shape[0] > 0:
                waypoints.append(pixel_point)
                points_in_camera_coords.append(point_in_camera_coords)

        thickness = np.linspace(
            THICKNESS_MIN, THICKNESS_MAX, len(waypoints))[::-1]
        color = np.linspace(COLOR_MIN, COLOR_MAX, len(waypoints))[::-1]

        #if len(waypoints) == STEP:
        if len(waypoints) > WINDOW_LEN:
            waypoints_spline = noisy_2_smooth(waypoints, num_points = 4)
            waypoints = np.vstack(waypoints)
            points_in_camera_coords_spline = noisy_2_smooth(points_in_camera_coords, num_points = 4)
            points_in_camera_coords = np.vstack(points_in_camera_coords)

            amplitude_original = 0
            amplitude_smooth = 0
            rotation_original = 0
            rotation_smooth = 0
            for j in range(waypoints.shape[0]):

                
                img = cv2.circle(img, (int(waypoints[j, 0]), int(waypoints[j, 1])), int(
                    thickness[j]), (0, int(color[j]), 0), -1)

                if j < waypoints.shape[0] - 1:

                    amplitude_original += np.linalg.norm(points_in_camera_coords[j+1, :] - points_in_camera_coords[j, :])
                    rotation_original += (np.arctan2(points_in_camera_coords[j+1, 0] - points_in_camera_coords[j, 0], points_in_camera_coords[j+1, 2] - points_in_camera_coords[j, 2]))

            for j in range(waypoints_spline.shape[0]):

                img = cv2.circle(img, (int(waypoints_spline[j, 0]), int(waypoints_spline[j, 1])), int(
                    thickness[j]), (0, 0, int(color[j])), -1)

                if j < waypoints_spline.shape[0] - 1:

                    amplitude_smooth += np.linalg.norm(points_in_camera_coords_spline[j+1, :] - points_in_camera_coords_spline[j, :])
                    rotation_smooth += (np.arctan2(points_in_camera_coords_spline[j+1, 0] - points_in_camera_coords_spline[j, 0], points_in_camera_coords_spline[j+1, 2] - points_in_camera_coords_spline[j, 2]))


            #cv2.putText(img, f"Average Amplitude Noisy = {amplitude_original / waypoints.shape[0]:.2}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            #cv2.putText(img, f"Average Amplitude Smooth = {amplitude_smooth / waypoints_spline.shape[0]:.2}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            #cv2.putText(img, f"Average Rotation Noisy = {rotation_original / waypoints.shape[0]:.2}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            #cv2.putText(img, f"Average Rotation Smooth = {rotation_smooth / waypoints_spline.shape[0]:.2}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow('Waypoint Animation', img)

        if cv2.waitKey(1) == ord('q'):

            # press q to terminate the loop
            cv2.destroyAllWindows()
            break

        video.write(cv2.resize(img, (WIDTH//2, HEIGHTH//2)))

        i += 1
        time.sleep(0.01)
    video.release()


if __name__ == '__main__':

    # Fetch the h5 files
    dataset_path = '/home/vaydingul20/Documents/Codes/dataset-detailed/expert/'
    episode = 0
    main(dataset_path, episode)
