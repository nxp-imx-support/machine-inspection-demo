#! /usr/bin/env python3
#
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Copyright 2022, 2024 NXP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
from time import clock_gettime_ns
from time import CLOCK_MONOTONIC_RAW as time_clock
from math import sqrt
import argparse
import io
import time
import numpy as np

import cv2
from PIL import Image
import tflite_runtime.interpreter as tflite

import threading
from uarm.wrapper import SwiftAPI

CV2_RED = (0, 0, 255)
CV2_GREEN = (0, 255, 0)
CV2_BLUE = (255, 0, 0)
CV2_BLACK = (0, 0, 0)

SPEED_DECAY_RATE = 0.01
FPS_DECAY_RATE = 0.05

# Main state machine
# Changing state is done by pressing the corresponding key (one of 'p', 'c', 'v').
# Starting state is calibration.
#
#                              ┌────────────┐    c    ┌─────────────┐
#                              │            │ ──────▶ │             │
#                              │            │         │ calibration │
#      ┌─────────────┐         │            │ ◀────── │             │
#      │             │ ──────▶ │            │         └─────────────┘
#      │    pause    │    p    │    play    │    p
#      │             │ ◀────── │            │         ┌─────────────┐
#      └─────────────┘         │            │ ◀────── │             │
#                              │            │         │ validation  │
#                              │            │ ──────▶ │             │
#                              └────────────┘    v    └─────────────┘

STATE_CALIB = 1
STATE_VALIDATE = 2
STATE_PLAY = 3
STATE_PAUSE = 4


def compute_loc(locs, width, height):
    y0 = int(locs[0] * height)
    x0 = int(locs[1] * width)
    y1 = int(locs[2] * height)
    x1 = int(locs[3] * width)
    return (x0, y0, x1, y1)


def set_input_tensor(interpreter, image):
    """Sets the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor


def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()

    # Get all output details
    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {'bounding_box': boxes[i], 'class_id': classes[i], 'score': scores[i]}
            results.append(result)
    return results


def load_labels(path):
    items = {}
    item_id = None
    item_name = None
    with open(path, 'r') as file:
        for line in file:
            line.replace(" ","")
            if line == "item{":
                pass
            elif line == "}":
                pass
            elif "id" in line:
                item_id = int(line.split(":",1)[1].strip())
            elif "display_name" in line:
                item_name = line.split(" ")[-1].replace("\"", " ").strip()

            if item_id is not None and item_name is not None:
                items[item_id] = item_name
                item_id = None
                item_name = None

    return items

class Fruit(object):
    def __init__(self, timestamp, score, centerX, centerY):
        self.first_timestamp = timestamp
        self.clicked = False
        self.first_y = centerY
        self.score = 0
        self.lock = threading.Lock()
        self.update(timestamp, score, centerX, centerY)

    def update(self, timestamp, score, centerX, centerY):
        with self.lock:
            self.timestamp = timestamp
            self.score += score
            self.x = centerX
            self.y = centerY

    def get_position(self):
        with self.lock:
            return (self.timestamp, self.x, self.y)


class Robot(threading.Thread):
    def __init__(self, fruits, port):
        threading.Thread.__init__(self, name="robot")
        self.daemon = True
        self.play_trigger = threading.Event()

        # Robot movement is assumed to folow  the equation T = a + b*d with T time of movement and d length
        self.a = 0.252
        self.b = 0.00151
        self.WORLDSEND = -85
        self.rest_x = 210
        self.rest_y = -25
        self.rest_z = 50
        self.avg_height = 0
        self.conveyor_speed = 100
        self.robot_data = []
        self.fruits = fruits

        self.swift = SwiftAPI(port=port, cmd_pending_size=8, callback_thread_pool_size=8, enable_write_thread=True)

        self.swift.waiting_ready(timeout=3)

        device_info = self.swift.get_device_info()
        print(device_info)
        firmware_version = device_info['firmware_version']
        if firmware_version and not firmware_version.startswith(('0.', '1.', '2.', '3.')):
            self.swift.set_speed_factor(1)
        self.swift.set_mode(0)
        self.swift.reset(wait=True, speed=500)

    def set_position(self, wait=False, **kwargs):
        if 'x' in kwargs.keys():
            kwargs['x'] = round(kwargs['x'])
        if 'y' in kwargs.keys():
            kwargs['y'] = round(kwargs['y'])
        if 'z' in kwargs.keys():
            kwargs['z'] = round(kwargs['z'])
        res = self.swift.set_position(wait=wait, timeout=2, **kwargs)
        if res != 'OK' and res != None:
            print("swift.set_position returned error {} with wait={}".format(res, wait))

    def go_to_rest(self):
        self.set_position(x=self.rest_x, y=self.rest_y, z=self.rest_z, speed=1000, wait=True)

    def play(self):
        self.play_trigger.set()

    def pause(self):
        self.play_trigger.clear()

    def find_intersect(self, T, X0, Y0, X, Y):
        _Sb = (self.conveyor_speed * self.b)**2
        _R = T / 1000000000.0 + self.a
        _A = _Sb - 1
        _B0 = _R * self.conveyor_speed - Y
        _B1 = _Sb * Y0
        _B = _B0 + _B1
        _C = _Sb * (Y0**2 + (X0 - X)**2) - _B0**2
        _D = _B**2 - _A * _C
        if _D < 0:
            return -400 # Make sure we skip this one
        _D = sqrt(_D)
        Yp = (_B + _D) / _A
        sol = "a"
        if Yp > Y:
            sol = "b"
            Yp = (_B - _D) / _A
        return Yp

    def run(self):
        while True:
            self.play_trigger.wait()

            (best, self.conveyor_speed) = self.fruits.get_best()

            if best is not None:
                (timestamp, x, y) = best.get_position()

                best.clicked = True
                robot_start = clock_gettime_ns(time_clock)
                xt = x
                yt = self.find_intersect(robot_start - timestamp, x0, y0, x, y)
                if yt > self.WORLDSEND:
                    self.set_position(cmd='G0', x=xt, y=yt, z=self.avg_height + 5, speed=1000)
                    self.swift.flush_cmd(wait_stop=True)
                    
                    self.set_position(cmd='G0', x=xt, y=yt, z=self.avg_height - 1, speed=1000)
                    self.swift.flush_cmd(wait_stop=True)
                    
                    robot_end = clock_gettime_ns(time_clock)
                    self.set_position(cmd='G0', x=xt, y=yt, z=self.avg_height + 5, speed=1000)
                    self.swift.flush_cmd(wait_stop=True)
                    
                    start_delay = (robot_start - timestamp) / 1000000.0
                    move_time = (robot_end - robot_start) / 1000000.0
                    distance = sqrt((xt - x0)**2 + (yt - y0)**2)
                    self.robot_data.append([distance, move_time/1000.0])
                    if len(self.robot_data) > 1000:
                        self.robot_data.pop(0)

                    data = np.asarray(self.robot_data)
                    model = np.polyfit(data[:,0], data[:,1], 1)
                    self.a = model[1]
                    self.b = model[0]

                    # Disable 2 lines below for return to rest position every time
                    x0 = xt
                    y0 = yt

            else:
                self.set_position(cmd='G0', x=self.rest_x, y=self.rest_y, z=self.avg_height + 5, speed=1000, wait=True)
                x0 = self.rest_x
                y0 = self.rest_y
                time.sleep(0.01)


class Fruits(object):
    def __init__(self):
        self.db = []
        self.speed = 100
        self.lock = threading.Lock()

    def get_best(self):
        best = None
        with self.lock:
            if self.db != []:
                apples = [fruit for fruit in self.db if fruit.score >= 0.0 and not fruit.clicked]
                if apples:
                    best = sorted(apples, key=lambda fruit:fruit.y)[0]

        return (best, self.speed)


    def update_db(self, timestamp, scores, centers):
        """Takes a timestamp and 2 arrays for scores (1 for apples, -1 for others) and object centers, and returns a speed estimate and an array of apple centers."""
        if len(self.db) == 0:
            with self.lock:
                for i in range(len(scores)):
                    self.db.append(Fruit(timestamp, scores[i], centers[i][0], centers[i][1]))
        else:
            matches = []
            for i in range(len(scores)):
                best_dist = 1000000
                best_idx = 0
                for j in range(len(self.db)):
                     # We compensate for distance travelled since last observation
                     dist = (centers[i][0] - self.db[j].x)**2 + (centers[i][1] - (self.db[j].y - self.speed*(timestamp - self.db[j].timestamp)/1000000000.0))**2
                     if dist < best_dist:
                         best_dist = dist
                         best_idx = j
                matches.append([best_idx, best_dist, i])

            new_found = [new_idx for [idx, dist, new_idx] in matches if dist > 1600]

            old_matches = [match for match in matches if match[2] not in new_found]

            if len(old_matches) != 0:
                speeds = []
                for [idx, dist, new_idx] in old_matches:
                    duration = (timestamp - self.db[idx].first_timestamp) % (1 << 32)
                    if self.db[idx].timestamp != timestamp and duration > 250000000: # Ignore possible duplicates from NN
                        fruit_speed = abs(self.db[idx].first_y - centers[new_idx][1])*1000000000.0 / duration
                        if fruit_speed > 1: # Ignore static objects (GUI, robot arm, etc)
                            speeds.append(fruit_speed)
                        self.db[idx].update(timestamp, scores[new_idx], centers[new_idx][0], centers[new_idx][1])

                if len(speeds) != 0:
                    new_speed = np.mean(speeds)
                    self.speed = SPEED_DECAY_RATE*new_speed + (1-SPEED_DECAY_RATE)*self.speed

            with self.lock:
                for idx in new_found:
                    self.db.append(Fruit(timestamp, scores[idx], centers[idx][0], centers[idx][1]))

                self.db = [fruit for fruit in self.db if (((timestamp - fruit.timestamp)*self.speed/1000000000.0) < (100 + fruit.y))]

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', help='File path of .tflite file.', required=True)
    parser.add_argument('--labels', help='File path of labels file.', required=True)
    parser.add_argument('--threshold', help='Score threshold for detected objects.', required=False, type=float, default=0.4)
    args = parser.parse_args()

    labels = load_labels(args.labels)

    ext_delegate = [ tflite.load_delegate("/usr/lib/libethosu_delegate.so")  ]
    interpreter = tflite.Interpreter(model_path=args.model, experimental_delegates=ext_delegate)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details)
    print(output_details)

    _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']
    print("width: {}, height: {}".format(input_width, input_height))

    capture = cv2.VideoCapture(
        'v4l2src device=/dev/video0 do_timestamp=true ! queue max-size-time=30000 max_size-buffers=1 leaky=2 ! video/x-raw,format=YUY2,width=1920,height=1080 ! imxvideoconvert_pxp  rotation=1 ! video/x-raw,format=BGR,width=360,height=640 ! videocrop top=140 bottom=140 ! appsink drop=true max-buffers=1',
        cv2.CAP_GSTREAMER)
    cv2.namedWindow("video")

    fruits = Fruits()

    pts_file = open("/home/root/machine-inspection/pts_file", "r")
    pts = pts_file.readline()

    robot = Robot(fruits, port=pts)
    robot.start()

    camera_points = []
    robot_points = []
    robot.avg_height = 0
    (x0, y0, x1, y1) = (0, 0, 0, 0)
    state = STATE_CALIB
    previous_state = STATE_PLAY
    calibrated = False
    decay = 0.05
    fps = 30
    fps_previous = 0

    # Those values will be overridden if calibration if performed
    h = np.asarray([[-4.89377060e-01, 1.16771756e-01, 2.94950387e+02],
                    [8.80741774e-03, 4.69708977e-01, -4.74280218e+01],
                    [-2.00292156e-04, 5.00393208e-04, 1.00000000e+00]])
    robot.avg_height = 13
    print("Homography matrix: {}".format(h))

    while True:
        fps_current = clock_gettime_ns(time_clock)
        ret, cv2_im = capture.read()
        height, width, _ = cv2_im.shape

        before = clock_gettime_ns(time_clock)
        image = Image.fromarray(cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB))
        after = clock_gettime_ns(time_clock)

        image = image.convert('RGB').resize((input_width, input_height), Image.ANTIALIAS)

        # Run inference
        start = clock_gettime_ns(time_clock)
        results = detect_objects(interpreter, image, args.threshold)
        end = clock_gettime_ns(time_clock)

        label_id = 0
        prob = 0
        if len(results) != 0:
            label_id = int(results[0]['class_id'] + 1)
            prob = round(results[0]['score'] * 100)
        if label_id < 89 and label_id > 0:
            top_class = labels[label_id]

        fps = (1.0 - decay) * fps + decay * (1000000000.0 / (fps_current - fps_previous))
        fps_previous = fps_current
        inference_time = (end - start) / 1000000.0

        key = cv2.waitKey(1)
        if key != -1:
            print("Pressed: {}".format(key))
        if key == 27:
            robot.pause()
            robot.swift.disconnect()
            break
        elif key == ord('p'):
            if state == STATE_PLAY:
                state = STATE_PAUSE
            else:
                state = STATE_PLAY
        elif key == ord('c'):
            state = STATE_CALIB
        elif key == ord('v'):
            state = STATE_VALIDATE

        if state == STATE_CALIB:
            if previous_state ==  STATE_PLAY:
                robot.pause()

            if previous_state != STATE_CALIB:
                robot.pause()
                robot.set_position(x=150, y=0, z=robot.rest_z, speed=1000, wait=True)
                robot.swift.set_servo_detach()

            if key == ord('r'):
                camera_points = []
                robot_points = []
                robot.avg_height = 0
                calibrated = False

            if key == ord(' '):
                (x, y, z) = robot.swift.get_position()
                rp = [x, y]
                cp = [(x0 + x1) / 2, (y0 + y1) / 2]
                camera_points.append(cp)
                robot_points.append(rp)
                n_points = len(camera_points)
                robot.avg_height = (1.0 - 1 / n_points) * robot.avg_height + z / n_points
                print("Num_points: {} Avg height: {:2.1f} New point: camera {} robot {}".format(n_points, robot.avg_height, cp, rp))
                if n_points >= 4:
                    np_cps = np.asarray(camera_points).reshape(-1, 1, 2)
                    np_rps = np.asarray(robot_points).reshape(-1, 1, 2)
                    print("Robot coordinates: {}".format(np_cps))
                    h, _ = cv2.findHomography(np_cps, np_rps)
                    print("Homography matrix: {}".format(h))
                    reproj_points = cv2.perspectiveTransform(np_cps, h)
                    print("Reprojected robot coordinates: {}".format(reproj_points))
                    calibrated = True

            if key == ord('f') and top_class == "apple":
                (x0, y0, x1, y1) = compute_loc(results[0]['bounding_box'], width, height)

            cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), CV2_BLACK)
            cv2_im = cv2.line(cv2_im, (x0, y0), (x1, y1), CV2_BLACK)
            cv2_im = cv2.line(cv2_im, (x0, y1), (x1, y0), CV2_BLACK)

        elif state == STATE_VALIDATE:
            if previous_state ==  STATE_PLAY:
                robot.pause()

            if previous_state == STATE_CALIB:
                robot.swift.set_servo_attach(wait=True)
                robot.go_to_rest()

            if key == ord('r'):
                robot.go_to_rest()

            if top_class == "apple":
                (x0, y0, x1, y1) = compute_loc(results[0]['bounding_box'], width, height)
                cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), CV2_GREEN)
                cv2_im = cv2.line(cv2_im, (x0, y0), (x1, y1), CV2_GREEN)
                cv2_im = cv2.line(cv2_im, (x0, y1), (x1, y0), CV2_GREEN)

                if key == ord(' '):
                    cp = np.asarray([(x0 + x1) / 2, (y0 + y1) / 2]).reshape(1, 1, 2)
                    rp = cv2.perspectiveTransform(cp, h)
                    print("Detected apple at {}, sending robot to {}, height {}".format(cp, rp, robot.avg_height + 10))
                    robot.set_position(x=rp[0][0][0], y=rp[0][0][1], z=robot.avg_height + 50, speed=1000, wait=True)
                    robot.swift.flush_cmd(wait_stop=True)

        elif state == STATE_PLAY or state == STATE_PAUSE:
            if previous_state == STATE_CALIB:
                robot.swift.set_servo_attach(wait=True)

            if previous_state == STATE_CALIB or previous_state == STATE_VALIDATE:
                robot.go_to_rest()

            if state == STATE_PLAY and previous_state != STATE_PLAY:
                print("**** PLAY ****")
                robot.play()

            if state == STATE_PAUSE and previous_state != STATE_PAUSE:
                print("**** PAUSE ****")
                robot.pause()

            # Process output
            current_apples = []
            scores = []
            centers = []
            for result in results:
                class_name = labels[int(result['class_id'] + 1)]
                (x0, y0, x1, y1) = compute_loc(result['bounding_box'], width, height)
                
                if (abs((x0 - x1) * (y0 - y1)) < 40000): # Filter out full picture matches
                    score = result['score']
                    class_prob = round(score * 100)
                    cv2.putText(cv2_im, "{} {}".format(class_name, class_prob), (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, CV2_BLACK, 1)
                
                    if class_name == "apple":
                        centers.append([float(x0), float(y0)])
                        centers.append([float(x1), float(y1)])
                        scores.append(1.0)
                        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), CV2_GREEN)
                    elif class_name == "orange": # apples are pretty much never detected as oranges, so give those a big negative bias
                        centers.append([float(x0), float(y0)])
                        centers.append([float(x1), float(y1)])
                        scores.append(-128)
                        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), CV2_RED)
                    else:
                        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), CV2_BLUE)

            cv2.putText(cv2_im, "{:2.0f} fps   inference {:3.1f} ms  {:3.1f} {:3.1f} {}".format(
                fps, inference_time, (after - before) / 1000000.0, (start - after) / 1000000.0, top_class),
                (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, CV2_GREEN, 1)

            if len(centers) != 0:
                centers = np.asarray(centers).reshape(1, -1, 2)
                centers = cv2.perspectiveTransform(centers, h)
                centers =  centers.reshape(-1, 2, 2)
                # We target a bit in front of the center, to account for the web app reaction time
                centers = [ [(x0 + x1) / 2, (y0 + y1) / 2 - abs(y0 -y1)/4]  for [[x0,y0], [x1,y1]] in centers]
                fruits.update_db(fps_current, scores, centers)

        cv2.imshow("video", cv2_im)
        previous_state = state


if __name__ == '__main__':
    main()
