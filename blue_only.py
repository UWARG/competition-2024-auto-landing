"""
Auto-landing solution for 2024 competition.
"""
import os
import time
import pathlib
import math
import copy

import cv2
import numpy as np
from dotenv import load_dotenv
from pymavlink import mavutil
from picamera2 import PiCamera2, Preview


load_dotenv(".key")
secret_key = os.getenv("KEY")
LOG_DIRECTORY_PATH = pathlib.Path("logs")
SAVE_PREFIX = str(pathlib.Path(LOG_DIRECTORY_PATH, "image_" + str(int(time.time())) + "_"))

cam = PiCamera2()
camera_config = cam.create_preview_configuration(main={"size": (640, 480)})
cam.configure(camera_config)
cam.start_preview(Preview.QTGL) # comment to disable preview
cam.start()
time.sleep(0.1)

kernel = np.ones((2, 2), np.uint8)
FOV_X = 62.2  # Camera horizontal field of view
FOV_Y = 48.8  # Camera vertical field of view

vehicle = mavutil.mavlink_connection('udpin:localhost:14551')
# vehicle = mavutil.mavlink_connection('/dev/ttyUSB0', baud=57600)
vehicle.wait_heartbeat()
print("Heartbeat from system (system %u component %u)" % (vehicle.target_system, vehicle.target_component))

def current_milli_time():
    return round(time.time() * 1000)

def is_contour_circular(contour):
    perimeter = cv2.arcLength(contour, True)

    # Check if the perimeter is zero
    if perimeter == 0:
        return False

    area = cv2.contourArea(contour)
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    return circularity > 0.8  # Adjust this threshold as needed

def is_contour_large_enough(contour, min_diameter):
    (x, y), radius = cv2.minEnclosingCircle(contour)
    diameter = radius * 2
    return diameter >= min_diameter

def calc_target_distance(height_agl, x_rad, y_rad):
    x_deg = x_rad * 180 / math.pi
    y_deg = y_rad * 180 / math.pi
    x_ground_dist_m = math.tan(x_rad) * height_agl
    y_ground_dist_m = math.tan(y_rad) * height_agl
    ground_hyp = math.sqrt(math.pow(x_ground_dist_m, 2) + math.pow(y_ground_dist_m, 2))
    print("Required horizontal correction (m): ", ground_hyp)
    target_to_vehicle_dist = math.sqrt(math.pow(ground_hyp, 2) + math.pow(height_agl, 2))
    print("Distance from vehicle to target (m): ", target_to_vehicle_dist)
    return target_to_vehicle_dist

pos_message = vehicle.mav.command_long_encode(
    vehicle.target_system,  # Target system ID
    vehicle.target_component,  # Target component ID
    mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,  # ID of command to send
    0,  # Confirmation
    33,  # param1: Message ID to be streamed
    250000,  # param2: Interval in microseconds
    0,  # param3 (unused)
    0,  # param4 (unused)
    0,  # param5 (unused)
    0,  # param5 (unused)
    0  # param6 (unused)
)

vehicle.setup_signing(secret_key, True, None, int(time.time()), 0)
vehicle.mav.send(pos_message)

altitude_m = 0
loop_counter = 0
last_time = current_milli_time()
while True:
    image = cam.capture_array()
    im_h, im_w, c = image.shape
    print("Input image width: " + str(im_w))
    print("Input image height: " + str(im_h))
    try:
        altitude_mm = vehicle.messages[
            'GLOBAL_POSITION_INT'].relative_alt  # Note, you can access message fields as attributes!
        altitude_m = altitude_mm / 1000
        if altitude_m < 0:
            altitude_m = 0
        print("Altitude AGL: ", altitude_m)
    except:
        print('No GLOBAL_POSITION_INT message received')

    # log image every 30 loops
    if(loop_counter % 30 == 0):
        cv2.imwrite(SAVE_PREFIX + str(loop_counter) + ".png", image)

    # Finding contours in original image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = 200
    im_bw = cv2.threshold(gray_image, thresh, 255, cv2.THRESH_BINARY)[1]
    im_dilation = cv2.dilate(im_bw, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(im_dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        contours_with_children = set(i for i, hier in enumerate(hierarchy[0]) if hier[2] != -1)
        parent_circular_contours = [cnt for i, cnt in enumerate(contours) if
                                    is_contour_circular(cnt) and is_contour_large_enough(cnt,
                                                                                         7) and i in contours_with_children]
        contour_image = copy.deepcopy(image)
        cv2.drawContours(contour_image, parent_circular_contours, -1, (0, 255, 0), 2)

        # Find the contour with the largest area among circular contours
        largest_contour = max(parent_circular_contours, key=cv2.contourArea, default=None)

        rect_image = copy.deepcopy(image)

        if largest_contour is not None:
            # Draw a rectangle around the largest circular contour
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(rect_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            x_center = x + w / 2
            y_center = y + h / 2
            angle_x = (x_center - im_w / 2) * (FOV_X * (math.pi / 180)) / im_w
            angle_y = (y_center - im_h / 2) * (FOV_Y * (math.pi / 180)) / im_h
            cv2.circle(rect_image, (int(x_center), int(y_center)), 2, (0, 0, 255), 2)
            print("X Angle (rad): ", angle_x)
            print("Y Angle (rad): ", angle_y)
            target_dist = calc_target_distance(altitude_m, angle_x, angle_y)
            vehicle.mav.landing_target_send(0, 0, mavutil.mavlink.MAV_FRAME_BODY_NED, angle_x, angle_y, target_dist, 0,
                                            0)

        cv2.imshow("Binary", im_dilation)
        cv2.imshow('Mask Contours', rect_image)
        cv2.waitKey(10)
        loop_counter += 1
        if current_milli_time() - last_time > 1000:
            print("FPS:", loop_counter)
            loop_counter = 0
            last_time = current_milli_time()
        # break
