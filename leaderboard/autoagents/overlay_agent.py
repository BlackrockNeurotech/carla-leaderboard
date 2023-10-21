#!/usr/bin/env python

# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module provides a human agent to control the ego vehicle via keyboard
"""

import numpy as np
import json

try:
    import pygame
    from pygame.locals import K_DOWN
    from pygame.locals import K_LEFT
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError("cannot import pygame, make sure pygame package is installed")
import errno
import logging
from multiprocessing import Process, Value, Pipe, Queue
import struct
import socket
import datetime
import time
from collections import deque
from os import getenv

import carla

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

logger = logging.getLogger(__name__)


def get_entry_point():
    return "OverlayAgent"


"""
Copying NetdataReceiver here to avoid import errors
"""


def clock_sync_udp(keep_running, loglevel, hostaddr="127.0.0.1", sync_port=10001):
    logsync = logging.getLogger("ClockSync")
    logsync.setLevel(loglevel)

    recv_format = ">I"
    send_format = ">IHHHHHHI"
    recv_size = struct.calcsize(recv_format)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024)
    sock.settimeout(2.0)
    # sock.setblocking(False)
    sock.bind((hostaddr, sync_port))
    logsync.debug("ClockSync Unicast UDP Server listening on port %d", sync_port)
    try:
        while keep_running.value:
            try:
                msg, addr = sock.recvfrom(recv_size)
                recv_dt = datetime.datetime.utcnow()
                if msg:
                    logsync.debug("Received clocksync packet.")
                    data = struct.unpack(recv_format, msg)
                    response = struct.pack(
                        send_format,
                        data[0],
                        recv_dt.year,
                        recv_dt.month,
                        recv_dt.day,
                        recv_dt.hour,
                        recv_dt.minute,
                        recv_dt.second,
                        recv_dt.microsecond,
                    )
                    logsync.debug("Sending clocksync reply.")
                    sock.sendto(bytearray(response), addr)
                else:
                    time.sleep(0.1)
            except socket.error as e:
                logsync.debug("Socket timeout without receiving a message")
        logsync.debug("Received signal to stop running.")
    except KeyboardInterrupt:
        logsync.debug("Received keyboard interrupt")
    logsync.debug("Returning")


def data_receiver_tcp(
    keep_running,
    data_pipe,
    msg_id_q: Queue,
    hostaddr="127.0.0.1",
    data_port=10000,
    encoding_format="Twist",
):
    logtcp = logger.getChild("TCPClient")
    if encoding_format.lower().startswith("twist"):
        msg_format = ">ffffff"
    else:  # "ffB"
        msg_format = ">IffB"
    ts_offset = len(msg_format) - 2
    msg_format += "HHHHHHI"  # datetime tail.
    msg_size = struct.calcsize(msg_format)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((hostaddr, data_port))
    sock.listen(1)
    sock.settimeout(2.0)

    logtcp.debug("data_receiver_tcp server waiting for connection...")

    connection = None

    try:
        while keep_running.value:
            try:
                if connection is None:
                    # sock.listen(1)
                    connection, client_address = sock.accept()
                    connection.settimeout(2.0)

                pkt_data = connection.recv(msg_size)
                if pkt_data:
                    recv_dt = datetime.datetime.utcnow()
                    pkt_data = struct.unpack(msg_format, pkt_data)
                    pkt_id = pkt_data[0]
                    msg_id_q.put_nowait(pkt_id)
                    data = pkt_data[1:]
                    """
                    Twist:
                    data = [throttle, 0, 0, 0, 0, wheel_angle, ... datetime stuff]
                    
                    ffB:
                    data = [steering (-1:1), throttle (0:1), brake (bool), ... datetime stuff]
                    
                    datetime stuff:
                            timestamp.year, timestamp.month, timestamp.day,
                            timestamp.hour, timestamp.minute, timestamp.second,
                            timestamp.microsecond
                    """
                    data_pipe.send(data)

                    client_dt = datetime.datetime(*data[ts_offset:], tzinfo=None)
                    latency = client_dt - recv_dt
                    ms_latency = round(1000 * latency.total_seconds())
                    logtcp.debug("Received data with latency %2.f ms.", ms_latency)
                else:
                    time.sleep(0.001)

            except socket.error as e:
                if e.errno == errno.ECONNRESET:
                    logtcp.warning("Client lost.")
                    connection.close()
                    logtcp.debug("data_receiver_tcp server waiting for connection...")
                    connection = None
        logtcp.debug("Received signal to stop running.")
    except KeyboardInterrupt:
        logtcp.debug("Received keyboard interrupt.")
    finally:
        keep_running.value = (
            0  # Probably not necessary as above has its own KBI handler.
        )

    if connection is not None:
        connection.close()
    sock.close()


class NetDataReceiver:
    def __init__(
        self,
        msg_id_q: Queue,
        hostaddr="127.0.0.1",
        data_port=10000,
        sync_port=10001,
        encoding_format="Twist",
        verbose=False,
    ):
        logger.setLevel(logging.INFO if verbose else logging.WARNING)
        self._latest_events = deque()
        self._latest_ctrl = deque()
        self._msg_id_q = msg_id_q
        self._encoding_format = encoding_format
        self._keep_running = Value("L", 1)
        self._parent_conn, self._child_conn = Pipe(duplex=False)

        # Startup the clock sync process
        self._clock_sync = Process(
            target=clock_sync_udp,
            args=(self._keep_running, logger.level),
            kwargs={"hostaddr": hostaddr, "sync_port": sync_port},
        )
        self._clock_sync.start()

        # Startup the data receiver process
        self._data_receiver = Process(
            target=data_receiver_tcp,
            args=(self._keep_running, self._child_conn, self._msg_id_q),
            kwargs={
                "hostaddr": hostaddr,
                "data_port": data_port,
                "encoding_format": self._encoding_format,
            },
        )
        self._data_receiver.start()

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        self._keep_running.value = 0
        self._clock_sync.join()
        self._data_receiver.join()

    def _update_latest(self):
        while self._parent_conn.poll():
            obj = self._parent_conn.recv()
            logger.debug(f"received tcp control: {obj}")
            if obj is not None:
                self._latest_events.append(obj)
                self._latest_ctrl.append(obj)

    def parse_tcp_input(
        self, control: carla.VehicleControl, steer_increment: float, retry=5
    ) -> carla.VehicleControl:
        # Override to make sure _update_latest() is called before parsing inputs.
        cnt = 0
        self._update_latest()
        tmp_ctrl = None
        while tmp_ctrl is None:
            tmp_ctrl = self._get_control()
            cnt += 1
            if cnt >= retry:
                return control
        control.throttle = min(
            max(0.0, tmp_ctrl.throttle), 0.8
        )  # bounded between 0.0 and 0.8
        control.steer = round(tmp_ctrl.steer, 1)
        return control

    def _get_events(self) -> set:
        out_events = set()
        do_twist = self._encoding_format.lower().startswith("twist")
        for obj_ix in range(len(self._latest_events)):
            obj = self._latest_events.popleft()
            # Convert obj to an event string or tuple. See keyboard.py key_event_map for examples.
            if do_twist:
                logger.debug(
                    "obj[0] >0 is throttle and <0 is brake, obj[5] is steering"
                )
            else:
                # TODO: If you want to decode events from obj then this could be a place to do it.
                # logger.debug(f"obj is [steering (-1.0,1.0), throttle (0.0,1.0), brake (bool)...]")
                out_events.add(f"set_input_controller:NetDataReceiver")
        return out_events

    def _get_control(self, ctrl: carla.VehicleControl = None) -> carla.VehicleControl:
        if len(self._latest_ctrl) == 0:
            time.sleep(0.001)
            return None
        ctrl = ctrl or carla.VehicleControl()
        do_twist = self._encoding_format.lower().startswith("twist")
        for obj_ix in range(len(self._latest_ctrl)):
            obj = self._latest_ctrl.popleft()
            if do_twist:
                if obj[0] > 0:
                    ctrl.throttle = obj[0]
                else:
                    ctrl.brake = -obj[0]
                ctrl.steer = obj[5]
            else:  # ffB
                ctrl.steer = obj[0]
                ctrl.throttle = obj[1] if obj[1] > 0.0 else 0.0
                if obj[2]:  # Boolean signal for brake
                    ctrl.brake = 1.0
                elif obj[1] < 0:  # Negative throttle
                    ctrl.brake = -1 * obj[1]
                else:
                    ctrl.brake = 0.0
        return ctrl

    @property
    def notification_text(self) -> str:
        return "TCP Remote Control enabled"


"""
End NetdataReceiver
"""


class OverlayInterface(object):

    """
    Class to control a vehicle manually for debugging purposes
    """

    def __init__(
        self, width, height, side_scale, left_mirror=False, right_mirror=False
    ):
        self._width = width
        self._height = height
        self._scale = side_scale
        self._surface = None

        self._left_mirror = left_mirror
        self._right_mirror = right_mirror

        pygame.init()
        pygame.font.init()
        self._clock = pygame.time.Clock()
        self._display = pygame.display.set_mode(
            (self._width, self._height), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("Overlay Agent")

    def run_interface(self, input_data):
        """
        Run the GUI
        """

        # Process sensor data
        image_center = input_data["Center"][1][:, :, -2::-1]
        self._surface = pygame.surfarray.make_surface(image_center.swapaxes(0, 1))

        # Add the left mirror
        if self._left_mirror:
            image_left = input_data["Left"][1][:, :, -2::-1]
            left_surface = pygame.surfarray.make_surface(image_left.swapaxes(0, 1))
            self._surface.blit(left_surface, (0, (1 - self._scale) * self._height))

        # Add the right mirror
        if self._right_mirror:
            image_right = input_data["Right"][1][:, :, -2::-1]
            right_surface = pygame.surfarray.make_surface(image_right.swapaxes(0, 1))
            self._surface.blit(
                right_surface,
                ((1 - self._scale) * self._width, (1 - self._scale) * self._height),
            )

        # Display image
        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))
        pygame.display.flip()

    def set_black_screen(self):
        """Set the surface to black"""
        black_array = np.zeros([self._width, self._height])
        self._surface = pygame.surfarray.make_surface(black_array)
        if self._surface is not None:
            self._display.blit(self._surface, (0, 0))
        pygame.display.flip()

    def _quit(self):
        pygame.quit()


class OverlayAgent(AutonomousAgent):

    """
    Overlay agent to control the ego vehicle via overlay
    """

    current_control = None
    agent_engaged = False

    def setup(self, path_to_conf_file):
        """
        Setup the agent parameters
        """
        self.track = Track.SENSORS

        self.agent_engaged = False
        self.camera_width = 2000 #1280
        self.camera_height = 900 #720
        self._side_scale = 0.15 #0.3
        self._left_mirror = True
        self._right_mirror = True

        self._hic = OverlayInterface(
            self.camera_width,
            self.camera_height,
            self._side_scale,
            self._left_mirror,
            self._right_mirror,
        )
        self._pkt_id_q = Queue()
        self._netdata_receiver = NetDataReceiver(
            msg_id_q=self._pkt_id_q,
            hostaddr="127.0.0.1",
            data_port=10000,
            sync_port=10001,
            encoding_format="ffB",
            verbose=False,
        )
        self._controller = OverlayControl(path_to_conf_file)
        self._prev_timestamp = 0

    def sensors(self):
        """
        Define the sensor suite required by the agent

        :return: a list containing the required sensors in the following format:

        [
            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': -0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Left'},

            {'type': 'sensor.camera.rgb', 'x': 0.7, 'y': 0.4, 'z': 1.60, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                      'width': 300, 'height': 200, 'fov': 100, 'id': 'Right'},

            {'type': 'sensor.lidar.ray_cast', 'x': 0.7, 'y': 0.0, 'z': 1.60, 'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0,
             'id': 'LIDAR'}
        ]
        """

        pov_xyz_coords = getenv("CARLA_POV_XYZ")
        pov_fov = getenv("CARLA_POV_FOV")
        if pov_xyz_coords is not None:
            try:
                pov_xyz_coords = tuple(
                    [float(val) for val in pov_xyz_coords.split(",")]
                )
            except ValueError:
                print(
                    f"Invalid POV XYZ coords provided: {pov_xyz_coords}\nMust be provided with format x.0,y.0,z.0"
                )
                pov_xyz_coords = (0.0, -0.3, 1.25)
        else:
            pov_xyz_coords = (0.0, -0.3, 1.25)

        if pov_fov is not None:
            pov_fov = int(pov_fov)
        else:
            pov_fov = 100

        sensors = [
            {
                "type": "sensor.camera.rgb",
                "x": pov_xyz_coords[0],
                "y": pov_xyz_coords[1],
                "z": pov_xyz_coords[2],
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "width": self.camera_width,
                "height": self.camera_height,
                "fov": pov_fov,
                "id": "Center",
            },
        ]

        if self._left_mirror:
            sensors.append(
                {
                    "type": "sensor.camera.rgb",
                    "x": 0.7,
                    "y": -1.0,
                    "z": 1.0,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 210.0,
                    "width": self.camera_width * self._side_scale,
                    "height": self.camera_height * self._side_scale,
                    "fov": 100,
                    "id": "Left",
                }
            )

        if self._right_mirror:
            sensors.append(
                {
                    "type": "sensor.camera.rgb",
                    "x": 0.7,
                    "y": 1.0,
                    "z": 1.0,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 150.0,
                    "width": self.camera_width * self._side_scale,
                    "height": self.camera_height * self._side_scale,
                    "fov": 100,
                    "id": "Right",
                }
            )

        return sensors

    def run_step(self, input_data, timestamp):
        """
        Execute one step of navigation.
        """
        self.agent_engaged = True
        self._hic.run_interface(input_data)

        steer_increment = 3e-4 * (timestamp - self._prev_timestamp) * 1000
        control = self._controller.parse_events(timestamp - self._prev_timestamp)
        if control is not None:
            control = self._netdata_receiver.parse_tcp_input(control, steer_increment)
        self._prev_timestamp = timestamp

        return control

    def destroy(self):
        """
        Cleanup
        """
        self._hic.set_black_screen()
        self._hic._quit = True
        super().destroy()


class OverlayControl(object):

    """
    Overlay control for the overlay agent
    """

    def __init__(self, path_to_conf_file):
        """
        Init
        """
        self._control = carla.VehicleControl()
        self._steer_cache = 0.0
        self._clock = pygame.time.Clock()

        # Get the mode
        if path_to_conf_file:

            with (open(path_to_conf_file, "r")) as f:
                lines = f.read().split("\n")
                self._mode = lines[0].split(" ")[1]
                self._endpoint = lines[1].split(" ")[1]

            # Get the needed vars
            if self._mode == "log":
                self._log_data = {"records": []}

            elif self._mode == "playback":
                self._index = 0
                self._control_list = []

                with open(self._endpoint) as fd:
                    try:
                        self._records = json.load(fd)
                        self._json_to_control()
                    except json.JSONDecodeError:
                        pass
        else:
            self._mode = "normal"
            self._endpoint = None

    def _json_to_control(self):

        # transform strs into VehicleControl commands
        for entry in self._records["records"]:
            control = carla.VehicleControl(
                throttle=entry["control"]["throttle"],
                steer=entry["control"]["steer"],
                brake=entry["control"]["brake"],
                hand_brake=entry["control"]["hand_brake"],
                reverse=entry["control"]["reverse"],
                manual_gear_shift=entry["control"]["manual_gear_shift"],
                gear=entry["control"]["gear"],
            )
            self._control_list.append(control)

    def parse_events(self, timestamp):
        """
        Parse the keyboard events and set the vehicle controls accordingly
        """
        # Move the vehicle
        if self._mode == "playback":
            self._parse_json_control()
        else:
            self._parse_vehicle_keys(pygame.key.get_pressed(), timestamp * 1000)

        # Record the control
        if self._mode == "log":
            self._record_control()

        return self._control

    def _parse_vehicle_keys(self, keys, milliseconds):
        """
        Calculate new vehicle controls based on input keys
        """

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            elif event.type == pygame.KEYUP:
                if event.key == K_q:
                    self._control.gear = 1 if self._control.reverse else -1
                    self._control.reverse = self._control.gear < 0

        if keys[K_UP] or keys[K_w]:
            self._control.throttle = 0.8
        else:
            self._control.throttle = 0.0

        steer_increment = 3e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0

        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    def _parse_json_control(self):

        if self._index < len(self._control_list):
            self._control = self._control_list[self._index]
            self._index += 1
        else:
            print("JSON file has no more entries")

    def _record_control(self):
        new_record = {
            "control": {
                "throttle": self._control.throttle,
                "steer": self._control.steer,
                "brake": self._control.brake,
                "hand_brake": self._control.hand_brake,
                "reverse": self._control.reverse,
                "manual_gear_shift": self._control.manual_gear_shift,
                "gear": self._control.gear,
            }
        }

        self._log_data["records"].append(new_record)

    def __del__(self):
        # Get ready to log user commands
        if self._mode == "log" and self._log_data:
            with open(self._endpoint, "w") as fd:
                json.dump(self._log_data, fd, indent=4, sort_keys=True)
