#!/usr/bin/env python
#
# Copyright (c) 2020 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
#
"""
A basic AD agent using CARLA waypoints
"""

import copy
import sys
import time
import threading
import math
from carla_msgs.srv import SpawnObject
import uuid

import ros_compatibility as roscomp
from ros_compatibility.exceptions import *
from ros_compatibility.qos import QoSProfile, DurabilityPolicy

from carla_ad_agent.agent import Agent, AgentState

from carla_msgs.msg import (
    CarlaEgoVehicleInfo,
    CarlaActorList,
    CarlaTrafficLightStatusList,
    CarlaTrafficLightInfoList)
from derived_object_msgs.msg import ObjectArray
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64  # pylint: disable=import-error
from geometry_msgs.msg import PointStamped, Pose
from carla_msgs.srv import SpawnObject, DestroyObject
from diagnostic_msgs.msg import KeyValue


class CarlaAdAgent(Agent):
    """
    A basic AD agent using CARLA waypoints
    """

    def __init__(self):
        """
        Constructor
        """
        super(CarlaAdAgent, self).__init__("ad_agent")

        role_name = self.get_param("role_name", "ego_vehicle")
        self._avoid_risk = self.get_param("avoid_risk", True)

        self.data_lock = threading.Lock()
        self.vehicle_name = "npc_vehicle"
        self._ego_vehicle_pose = None
        self.my_vehicle_pose = None
        self._objects = {}
        self._lights_status = {}
        self._lights_info = {}
        self._target_speed = 0.
        self._spawned_vehicle_id = None
        self.spawn_client = self.create_client(SpawnObject, '/carla/spawn_object')
        self.destroy_client = self.create_client(DestroyObject, '/carla/destroy_object')
        
        self._pending_obstacle_spawn = {
            'x': None,
            'y': None,
            'z': None,
            'yaw': None,
            'ready': False,
            'timestamp': None
        }
        
        # self.pose_pub = self.create_publisher(
        #     Pose,
        #     "/carla/my_vehicle/set_transform",  # ⬅️ replace with your ID from objects.json
        #     10
        # )

        self.speed_command_publisher = self.new_publisher(
            Float64, "/carla/{}/speed_command".format(role_name),
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

        self._odometry_subscriber = self.new_subscription(
            Odometry,
            "/carla/{}/odometry".format(role_name),
            self.odometry_cb,
            qos_profile=10
        )
        
        self.my_odometry_subscriber = self.new_subscription(
            Odometry,
            f"/carla/{self.vehicle_name}/odometry",
            self.my_odometry_cb,
            qos_profile=10
        )

        self._target_speed_subscriber = self.new_subscription(
            Float64,
            "/carla/{}/target_speed".format(role_name),
            self.target_speed_cb,
            qos_profile=QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        )
        
        self._lidar_object_subscriber = self.create_subscription(
            PointStamped,
            '/detected/obstacle_relative_position',
            self.relative_obstacle_callback,
            10
        )
        
        self.spawn_client = self.create_client(SpawnObject, '/carla/spawn_object')
        self.pose_pub = self.create_publisher(Pose, f'/carla/{self.vehicle_name}/control/set_transform', 10)


        if self._avoid_risk:

            self._objects_subscriber = self.new_subscription(
                ObjectArray,
                "/carla/{}/objects".format(role_name),
                self.objects_cb,
                qos_profile=10
            )
            self._traffic_light_status_subscriber = self.new_subscription(
                CarlaTrafficLightStatusList,
                "/carla/traffic_lights/status",
                self.traffic_light_status_cb,
                qos_profile=QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            )
            self._traffic_light_info_subscriber = self.new_subscription(
                CarlaTrafficLightInfoList,
                "/carla/traffic_lights/info",
                self.traffic_light_info_cb,
                qos_profile=QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
            )

    def odometry_cb(self, odometry_msg):
        with self.data_lock:
            self._ego_vehicle_pose = odometry_msg.pose.pose
    
    def my_odometry_cb(self, odometry_msg):
        with self.data_lock:
            self.my_vehicle_pose = odometry_msg.pose.pose

    def target_speed_cb(self, target_speed_msg):
        with self.data_lock:
            self._target_speed = target_speed_msg.data * 3.6 # target speed from scenario is in m/s
    
    def get_yaw_from_quaternion(self, q):
            # q = geometry_msgs.msg.Quaternion
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            return math.atan2(siny_cosp, cosy_cosp)
    
    # Listen to lidar data and parse the obstacle's coordinates
    def relative_obstacle_callback(self, msg: PointStamped):
        if self._ego_vehicle_pose is None:
            self.get_logger().warn("Ego pose not ready yet.")
            return

        dx = msg.point.x
        dy = msg.point.y

        # Extract pose and yaw
        x0 = self._ego_vehicle_pose.position.x
        y0 = self._ego_vehicle_pose.position.y
        z0 = self._ego_vehicle_pose.position.z
        q = self._ego_vehicle_pose.orientation
        
        yaw = self.get_yaw_from_quaternion(q)

        # Transform to world coordinates
        world_x = x0 + dx * math.cos(yaw) - dy * math.sin(yaw)
        world_y = y0 + dx * math.sin(yaw) + dy * math.cos(yaw)
        world_z = z0
        
        # world_x = x0 
        # world_y = y0
        # world_z = z0 + 2
        
        # Store as private variable
        self._pending_obstacle_spawn.update({
            'x': world_x,
            'y': world_y,
            'z': world_z,
            'yaw': yaw,
            'ready': True,
            'timestamp': self.get_clock().now()
        })

        self.get_logger().info(
            f"Obstacle position cached for spawning at ({world_x:.2f}, {world_y:.2f})"
        )
    
    # Spawn an invisible vehicle (underground)
    def spawn_hidden_obstacle_vehicle(self):
        if self._ego_vehicle_pose is None:
            self.get_logger().warn("Ego pose not available, cannot spawn yet.")
            return

        # Extract ego pose
        pose = self._ego_vehicle_pose
        x0 = pose.position.x
        y0 = pose.position.y
        z0 = pose.position.z
        q = pose.orientation

        # Convert orientation to yaw
        yaw = self.get_yaw_from_quaternion(q)

        # Offset backwards by 20 meters
        hidden_x = x0
        hidden_y = y0
        hidden_z = z0 - 5.0 # use same z, assume flat ground

        request = SpawnObject.Request()
        request.type = "vehicle.audi.tt"
        request.id = self.vehicle_name
        request.attributes = [KeyValue(key='role_name', value='ego_vehicle')]

        spawn_pose = Pose()
        spawn_pose.position.x = hidden_x
        spawn_pose.position.y = hidden_y
        spawn_pose.position.z = hidden_z
        spawn_pose.orientation.w = math.cos(yaw / 2.0)
        spawn_pose.orientation.z = math.sin(yaw / 2.0)

        request.transform = spawn_pose
        request.random_pose = False
        request.attach_to = 0
        
        actor_control_sensor = {
            'type': 'sensor.pseudo.actor_control',
            'id': 'actor_control',
            'attributes': []
        }
        # Add the sensor to the request
        request.sensors = [actor_control_sensor]

        future = self.spawn_client.call_async(request)

        def callback(fut):
            try:
                result = fut.result()
                if result.id >= 0:
                    self._spawned_vehicle_id = result.id
                    self.get_logger().error(
                        f"Spawned hidden car with ID {result.id} at ({hidden_x:.2f}, {hidden_y:.2f})"
                    )
                else:
                    self.get_logger().error(f"Failed to spawn hidden vehicle: {result.error_string}")
            except Exception as e:
                self.get_logger().error(f"Spawn service failed: {e}")

        future.add_done_callback(callback)
    
    def move_npc_vehicle(self):
        if not self._pending_obstacle_spawn['ready']:
            # No object in front of lidar, move the vehicle somewhere no one sees
            pos = self._pending_obstacle_spawn

            # Create Pose message
            pose_msg = Pose()
            pose_msg.position.x = 230.0
            pose_msg.position.y = 195.0
            pose_msg.position.z = -8.0
            pose_msg.orientation.w = 1.0
            pose_msg.orientation.z = 0.0
            # Publish to the vehicle's pseudo-control topic
            self.pose_pub.publish(pose_msg)
            return
        
        # if self._spawned_vehicle_id is None:
        #     self.get_logger().error("my_vehicle not ready yet.")
        #     return

        pos = self._pending_obstacle_spawn

        # Create Pose message
        pose_msg = Pose()
        pose_msg.position.x = pos['x']
        pose_msg.position.y = pos['y']
        pose_msg.position.z = pos['z']
        pose_msg.orientation.w = math.cos(pos['yaw'] / 2.0)
        pose_msg.orientation.z = math.sin(pos['yaw'] / 2.0)

        # Publish to the vehicle's pseudo-control topic
        self.pose_pub.publish(pose_msg)
        # self.get_logger().error(f"Current pos is {self.my_vehicle_pose.position.x}, {self.my_vehicle_pose.position.y}, {self.my_vehicle_pose.position.z}")
        self.get_logger().error(f"Moved vehicle to ({pos['x']:.2f}, {pos['y']:.2f})")

        self._pending_obstacle_spawn['ready'] = False
    
    # Spawn a vehicle is there should be an obstacle
    def spawn_vehicle_if_ready(self):
        if not self._pending_obstacle_spawn['ready']:
            return
        if self._spawned_vehicle_id is not None:
            # Already spawned
            return

        pos = self._pending_obstacle_spawn

        request = SpawnObject.Request()
        request.type = "vehicle.audi.tt"
        request.id = str(uuid.uuid4())[:8]

        pose = Pose()
        pose.position.x = pos['x']
        pose.position.y = pos['y']
        pose.position.z = pos['z']
        pose.orientation.w = math.cos(pos['yaw'] / 2.0)
        pose.orientation.z = math.sin(pos['yaw'] / 2.0)

        request.transform = pose
        request.random_pose = False
        request.attach_to = 0

        future = self.spawn_client.call_async(request)

        def callback(fut):
            try:
                result = fut.result()
                if result.id >= 0:
                    self._spawned_vehicle_id = result.id
                    self.get_logger().error(f"Spawned vehicle at ({pos['x']:.1f}, {pos['y']:.1f}) with ID {result.id}")
                else:
                    self.get_logger().error(f"Spawn failed at ({pos['x']:.1f}, {pos['y']:.1f}): {result.error_string}")
            except Exception as e:
                self.get_logger().error(f"Spawn service failed: {e}")

        future.add_done_callback(callback)

        # Mark as consumed
        self._pending_obstacle_spawn['ready'] = False
    
    # Remove the vehicle spawned when obstacle is no longer there
    def destroy_vehicle_if_not_ready(self):
        if self._spawned_vehicle_id is None:
            return
        # if self._pending_obstacle_spawn['ready']:
        #     return  # still valid

        request = DestroyObject.Request()
        request.id = self._spawned_vehicle_id

        future = self.destroy_client.call_async(request)

        def callback(fut):
            try:
                result = fut.result()
                if result.success:
                    self.get_logger().info(f"Destroyed vehicle ID {request.id}")
                else:
                    self.get_logger().warn(f"Destroy failed for ID {request.id}")
            except Exception as e:
                self.get_logger().error(f"Destroy service call failed: {e}")

        future.add_done_callback(callback)

        self._spawned_vehicle_id = None

        
    def spawn_obstacle(self):
        spawn = self._pending_obstacle_spawn
        if not spawn['ready']:
            self.get_logger().warn("No pending spawn info available.")
            return

        request = SpawnObject.Request()
        request.type = "vehicle.tesla.model3"
        request.id = str(uuid.uuid4())[:8]

        pose = Pose()
        pose.position.x = spawn['x']
        pose.position.y = spawn['y']
        pose.position.z = spawn['z']
        pose.orientation.w = math.cos(spawn['yaw'] / 2.0)
        pose.orientation.z = math.sin(spawn['yaw'] / 2.0)
        
        self.get_logger().info(
            f"Spawning vheicle"
        )

        request.transform = pose
        request.random_pose = False
        request.attach_to = 0

        future = self.spawn_client.call_async(request)

        def callback(fut):
            try:
                result = fut.result()
                if result.id >= 0:
                    self.get_logger().info(f"Spawned car {result.id} at ({spawn['x']:.2f}, {spawn['y']:.2f})")
                else:
                    self.get_logger().error(f"Failed to spawn: {result.error_string}")
            except Exception as e:
                self.get_logger().error(f"Spawn service failed: {e}")

        future.add_done_callback(callback)

        # Clear the spawn info
        self._pending_obstacle_spawn['ready'] = False

    def objects_cb(self, objects_msg):
        objects = {}
        for obj in objects_msg.objects:
            objects[obj.id] = obj

        with self.data_lock:
            self._objects = objects

    def traffic_light_status_cb(self, traffic_light_status_msg):
        lights_status = {}
        for tl_status in traffic_light_status_msg.traffic_lights:
            lights_status[tl_status.id] = tl_status

        with self.data_lock:
            self._lights_status = lights_status

    def traffic_light_info_cb(self, traffic_light_info_msg):
        lights_info = {}
        for tl_info in traffic_light_info_msg.traffic_lights:
            lights_info[tl_info.id] = tl_info

        with self.data_lock:
            self._lights_info = lights_info

    def emergency_stop(self):
        stopping_speed = Float64()
        stopping_speed.data = 0.0
        self.speed_command_publisher.publish(stopping_speed)

    def run_step(self):
        """
        Executes one step of navigation.
        """

        # is there an obstacle in front of us?
        hazard_detected = False

        with self.data_lock:
            # retrieve relevant elements for safe navigation, i.e.: traffic lights and other vehicles.
            ego_vehicle_pose = copy.deepcopy(self._ego_vehicle_pose)
            objects = copy.deepcopy(self._objects)
            lights_info = copy.deepcopy(self._lights_info)
            lights_status = copy.deepcopy(self._lights_status)
            target_speed = copy.deepcopy(self._target_speed)

        if ego_vehicle_pose is None:
            self.loginfo("Waiting for ego vehicle pose")
            return
        
        # if self._spawned_vehicle_id is None:
        #     self.spawn_hidden_obstacle_vehicle()

        # ensure we have received all the status/info data of traffic lights.
        if set(lights_info.keys()) != set(lights_status.keys()):
            self.logwarn("Missing traffic light information")
            return

        if self._avoid_risk:
            # check possible obstacles
            vehicle_state, vehicle = self._is_vehicle_hazard(ego_vehicle_pose, objects)
            if vehicle_state:
                self._state = AgentState.BLOCKED_BY_VEHICLE
                hazard_detected = True

            # check for the state of the traffic lights
            light_state, traffic_light = self._is_light_red(ego_vehicle_pose, lights_status, lights_info)
            if light_state:
                self._state = AgentState.BLOCKED_RED_LIGHT
                hazard_detected = True

        speed_command = Float64()
        if hazard_detected:
            speed_command.data = 0.0
        else:
            self._state = AgentState.NAVIGATING
            speed_command.data = target_speed
        self.logwarn(f"Avoid risk is {self._avoid_risk}, speedu_command is {speed_command.data}")
        # self.spawn_obstacle()
        # Try to move, show, delete obstacle
        self.move_npc_vehicle()
        # self.destroy_vehicle_if_not_ready()
        # self.spawn_vehicle_if_ready()
       
        
       
        self.speed_command_publisher.publish(speed_command)


def main(args=None):
    """

    main function

    :return:
    """
    roscomp.init("ad_agent", args=args)
    controller = None
    try:
        executor = roscomp.executors.MultiThreadedExecutor()
        controller = CarlaAdAgent()
        executor.add_node(controller)

        roscomp.on_shutdown(controller.emergency_stop)

        update_timer = controller.new_timer(
            0.05, lambda timer_event=None: controller.run_step())

        controller.spin()

    except (ROSInterruptException, ROSException) as e:
        if roscomp.ok():
            roscomp.logwarn("ROS Error during exection: {}".format(e))
    except KeyboardInterrupt:
        roscomp.loginfo("User requested shut down.")
    finally:
        roscomp.shutdown()


if __name__ == "__main__":
    main()
