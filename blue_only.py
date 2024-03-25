"""
Auto-landing script for AEAC 2024 competition.

Usage: Either with contour (default) or yolo method.
python blue_only.py --method=...
"""

import argparse
import copy
import math
import os
import pathlib
import time
import yaml

import cv2
import dotenv
import numpy as np
import pymavlink

import yolo_decision

LOG_DIRECTORY_PATH = pathlib.Path("logs")
SAVE_PREFIX = str(pathlib.Path(LOG_DIRECTORY_PATH, "image_"))

CAMERA = 0
# Camera field of view
FOV_X = 62.2
FOV_Y = 48.8

CONTOUR_MINIMUM = 0.8

CONFIG_FILE_PATH = pathlib.Path("config.yml")

parser = argparse.ArgumentParser(description="AEAC 2024 Auto-landing script")
parser.add_argument(
    "--method",
    help="Method for selecting landing pads",
    choices=["contour", "yolo"],
    default="contour",
)
args = parser.parse_args()


def current_milli_time() -> int:
    """
    Returns the current time in milliseconds.
    """
    return round(time.time() * 1000)


def is_contour_circular(contour: np.ndarray) -> bool:
    """
    Checks if the shape is close to circular.

    Return: True is the shape is circular, false if it is not.
    """
    perimeter = cv2.arcLength(contour, True)

    # Check if the perimeter is zero
    if perimeter == 0.0:
        return False

    area = cv2.contourArea(contour)
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    return circularity > CONTOUR_MINIMUM


def is_contour_large_enough(contour: np.ndarray, min_diameter: float) -> bool:
    """
    Checks if the shape is larger than the provided diameter.

    Return: True if it is, false if it not.
    """
    _, radius = cv2.minEnclosingCircle(contour)
    diameter = radius * 2
    return diameter >= min_diameter


def calc_target_distance(height_agl: float, x_rad: float, y_rad: float) -> float:
    """
    Get the horizontal distance TODO
    """
    x_ground_dist_m = math.tan(x_rad) * height_agl
    y_ground_dist_m = math.tan(y_rad) * height_agl
    ground_hyp = math.sqrt(math.pow(x_ground_dist_m, 2) + math.pow(y_ground_dist_m, 2))
    print("Required horizontal correction (m): ", ground_hyp)
    target_to_vehicle_dist = math.sqrt(math.pow(ground_hyp, 2) + math.pow(height_agl, 2))
    print("Distance from vehicle to target (m): ", target_to_vehicle_dist)
    return target_to_vehicle_dist


# Callback expects `self` as first argument
# pylint: disable-next=unused-argument
def my_allow_unsigned_callback(self: object, message_id: int) -> bool:
    """
    Specify which messages to accept.
    """
    # Allow radio status messages
    return message_id == pymavlink.mavutil.mavlink.MAVLINK_MSG_ID_RADIO_STATUS


def run_with_contours() -> int:
    """
    Run script based on contours.
    """
    dotenv.load_dotenv(".key")
    secret_key = os.getenv("KEY")

    cam = cv2.VideoCapture(CAMERA)

    kernel = np.ones((2, 2), np.uint8)

    vehicle = pymavlink.mavutil.mavlink_connection("tcp:localhost:14550")
    # vehicle = pymavlink.mavutil.mavlink_connection('/dev/ttyUSB0', baud=57600)
    vehicle.wait_heartbeat()
    print(
        f"Heartbeat from system (system {vehicle.target_system} component {vehicle.target_component})"
    )

    pos_message = vehicle.mav.command_long_encode(
        vehicle.target_system,  # Target system ID
        vehicle.target_component,  # Target component ID
        pymavlink.mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,  # ID of command to send
        0,  # Confirmation
        33,  # param1: Message ID to be streamed
        250000,  # param2: Interval in microseconds
        0,  # param3 (unused)
        0,  # param4 (unused)
        0,  # param5 (unused)
        0,  # param5 (unused)
        0,  # param6 (unused)
    )

    secret_key = bytearray(secret_key, "utf-8")
    vehicle.setup_signing(secret_key, True, my_allow_unsigned_callback, int(time.time()), 0)
    vehicle.mav.send(pos_message)

    altitude_m = 0
    loop_counter = 0
    last_time = current_milli_time()
    last_image_time = current_milli_time()
    while True:
        is_pad_detected = False
        result, image = cam.read()
        if not result:
            print("ERROR: Could not get image from camera")
            continue

        image = cv2.flip(image, 0)
        image = cv2.flip(image, 1)
        im_h, im_w, _ = image.shape
        print("Input image width: " + str(im_w))
        print("Input image height: " + str(im_h))
        try:
            altitude_mm = vehicle.messages[
                "GLOBAL_POSITION_INT"
            ].relative_alt  # Note, you can access message fields as attributes!
            altitude_m = max(altitude_mm / 1000, 0.0)
            print("Altitude AGL: ", altitude_m)
        except (KeyError, AttributeError):
            print("No GLOBAL_POSITION_INT message received")
            continue

        # Finding contours in original image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        threshold = 180
        im_bw = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)[1]
        im_dilation = cv2.dilate(im_bw, kernel, iterations=1)
        contours, hierarchy = cv2.findContours(im_dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 0:
            contours_with_children = set(i for i, hier in enumerate(hierarchy[0]) if hier[2] != -1)
            parent_circular_contours = [
                cnt
                for i, cnt in enumerate(contours)
                if is_contour_circular(cnt)
                and is_contour_large_enough(cnt, 7)
                and i in contours_with_children
            ]
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
                is_pad_detected = True
                vehicle.mav.landing_target_send(
                    0,
                    0,
                    pymavlink.mavutil.mavlink.MAV_FRAME_BODY_NED,
                    angle_x,
                    angle_y,
                    target_dist,
                    0,
                    0,
                )

            # cv2.imshow("Binary", im_dilation)
            # cv2.imshow('Mask Contours', rect_image)
            # cv2.waitKey(10)
            loop_counter += 1
            if current_milli_time() - last_time > 1000:
                print("FPS:", loop_counter)
                loop_counter = 0
                last_time = current_milli_time()
            if current_milli_time() - last_image_time > 100:
                if is_pad_detected:
                    print("Bounding Box Image Write")
                    cv2.imwrite(SAVE_PREFIX + "_" + str(time.time()) + ".png", rect_image)
                else:
                    print("Plain Image Write")
                    cv2.imwrite(SAVE_PREFIX + "_" + str(time.time()) + ".png", image)
                last_image_time = current_milli_time()
            # break

    return 0


def yolo_detect(model_device: "str | int", detect_confidence: int, model_path: str) -> int:
    """
    Run script with YOLOv8 model.

    model_device: name of target device to run inference on (i.e. "cpu" or cuda device 0, 1, 2, 3).
    detect_confidence: model confidence threshold
    model_path: path to 2023 YOLO model
    """
    dotenv.load_dotenv(".key")
    secret_key = os.getenv("KEY")

    cam = cv2.VideoCapture(CAMERA)

    vehicle = pymavlink.mavutil.mavlink_connection("tcp:localhost:14550")
    # vehicle = pymavlink.mavutil.mavlink_connection('/dev/ttyUSB0', baud=57600)
    vehicle.wait_heartbeat()
    print(
        f"Heartbeat from system (system {vehicle.target_system} component {vehicle.target_component})"
    )

    pos_message = vehicle.mav.command_long_encode(
        vehicle.target_system,  # Target system ID
        vehicle.target_component,  # Target component ID
        pymavlink.mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,  # ID of command to send
        0,  # Confirmation
        33,  # param1: Message ID to be streamed
        250000,  # param2: Interval in microseconds
        0,  # param3 (unused)
        0,  # param4 (unused)
        0,  # param5 (unused)
        0,  # param5 (unused)
        0,  # param6 (unused)
    )

    secret_key = bytearray(secret_key, "utf-8")
    vehicle.setup_signing(secret_key, True, my_allow_unsigned_callback, int(time.time()), 0)
    vehicle.mav.send(pos_message)

    altitude_m = 0
    loop_counter = 0
    last_time = current_milli_time()
    last_image_time = current_milli_time()

    yolo_model = yolo_decision.DetectLandingPad(model_device, detect_confidence, model_path)

    while True:
        is_pad_detected = False
        result, image = cam.read()
        if not result:
            print("ERROR: Could not get image from camera")
            continue

        image = cv2.flip(image, 0)
        image = cv2.flip(image, 1)
        im_h, im_w, _ = image.shape
        print("Input image width: " + str(im_w))
        print("Input image height: " + str(im_h))
        try:
            altitude_mm = vehicle.messages[
                "GLOBAL_POSITION_INT"
            ].relative_alt  # Note, you can access message fields as attributes!
            altitude_m = max(altitude_mm / 1000, 0.0)
            print("Altitude AGL: ", altitude_m)
        except (KeyError, AttributeError):
            print("No GLOBAL_POSITION_INT message received")
            continue

        result, detections = yolo_model.get_landing_pads(image)
        if not result:
            continue

        best_landing_pad = yolo_model.find_best_pad(detections)
        if not best_landing_pad:
            continue

        x, y = best_landing_pad.x_1, best_landing_pad.y_1
        w = best_landing_pad.x_2 - best_landing_pad.x_1
        h = best_landing_pad.y_2 - best_landing_pad.y_1

        rect_image = copy.deepcopy(image)

        # Draw a rectangle around the largest circular contour0
        cv2.rectangle(rect_image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        x_center = x + w / 2
        y_center = y + h / 2
        angle_x = (x_center - im_w / 2) * (FOV_X * (math.pi / 180)) / im_w
        angle_y = (y_center - im_h / 2) * (FOV_Y * (math.pi / 180)) / im_h
        cv2.circle(rect_image, (int(x_center), int(y_center)), 2, (0, 0, 255), 2)
        print("X Angle (rad): ", angle_x)
        print("Y Angle (rad): ", angle_y)
        target_dist = calc_target_distance(altitude_m, angle_x, angle_y)
        is_pad_detected = True
        vehicle.mav.landing_target_send(
            0,
            0,
            pymavlink.mavutil.mavlink.MAV_FRAME_BODY_NED,
            angle_x,
            angle_y,
            target_dist,
            0,
            0,
        )

        loop_counter += 1
        if current_milli_time() - last_time > 1000:
            print("FPS:", loop_counter)
            loop_counter = 0
            last_time = current_milli_time()
        if current_milli_time() - last_image_time > 100:
            if is_pad_detected:
                print("Bounding Box Image Write")
                cv2.imwrite(SAVE_PREFIX + "_" + str(time.time()) + ".png", rect_image)
            else:
                print("Plain Image Write")
                cv2.imwrite(SAVE_PREFIX + "_" + str(time.time()) + ".png", image)
            last_image_time = current_milli_time()

    return 0


def main() -> int:
    """
    Main function.
    """
    try:
        with CONFIG_FILE_PATH.open("r", encoding="utf8") as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(f"Error parsing YAML file: {exc}")
                return -1
    except FileNotFoundError:
        print(f"File not found: {CONFIG_FILE_PATH}")
        return -1
    except IOError as exc:
        print(f"Error when opening file: {exc}")

    if args.method == "contour":
        return run_with_contours()

    if args.method == "yolo":
        model_device = config["yolo_detect_target"]["device"]
        detect_confidence = config["yolo_detect_target"]["confidence"]
        model_path = config["yolo_detect_target"]["model_path"]

        return yolo_detect(model_device, detect_confidence, model_path)

    return -1


if __name__ == "__main__":
    result_main = main()
    if result_main < 0:
        print(f"ERROR: Status code: {result_main}")

    print("Done!")
