"""
| File: LAV2.py
| Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2024, Marcelo Jacinto. All rights reserved.
| Description: Definition of the LAV2 vehicle class, a lightweight multirotor implementation.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from omni.isaac.dynamic_control import _dynamic_control

# The vehicle interface
from pegasus.simulator.logic.vehicles.vehicle import Vehicle, get_world_transform_xform

# Mavlink interface
from pegasus.simulator.logic.backends.px4_mavlink_backend import (
    PX4MavlinkBackend,
    PX4MavlinkBackendConfig,
)

# Sensors and dynamics setup
from pegasus.simulator.logic.dynamics import LinearDrag
from pegasus.simulator.logic.thrusters import QuadraticThrustCurve
from pegasus.simulator.logic.sensors import Barometer, IMU, Magnetometer, GPS


class LAV2Config:
    """
    Configuration class for the LAV2 vehicle.
    
    This data class is used to configure all aspects of the LAV2 vehicle including
    thrust curves, drag dynamics, sensors, and communication backends.
    """

    def __init__(self):
        """
        Initialize LAV2Config with default settings.
        """

        # Stage prefix of the vehicle when spawning in the world
        self.stage_prefix = "LAV2"

        # The USD file that describes the visual aspect of the vehicle (and some properties such as mass and moments of inertia)
        self.usd_file = ""

        # The default thrust curve for a quadrotor and dynamics relating to drag
        self.thrust_curve = QuadraticThrustCurve(
            {
                "rotor_constant": [
                    1.1529035793718415e-05,
                    1.1529035793718415e-05,
                    1.1529035793718415e-05,
                    1.1529035793718415e-05,
                ],
                "rolling_moment_coefficient": [
                    1.9211597483286326e-07,
                    1.9211597483286326e-07,
                    1.9211597483286326e-07,
                    1.9211597483286326e-07,
                ],
                "max_rotor_velocity": [1050, 1050, 1050, 1050],
            }
        )
        self.drag = LinearDrag([0.50, 0.30, 0.0])

        # The default sensors for a quadrotor
        self.sensors = [Barometer(), IMU(), Magnetometer(), GPS()]

        # The default graphical sensors for a quadrotor
        self.graphical_sensors = []

        # The default omnigraphs for a quadrotor
        self.graphs = []

        # The backends for actually sending commands to the vehicle. By default use mavlink (with default mavlink configurations)
        # [Can be None as well, if we do not desired to use PX4 with this simulated vehicle]. It can also be a ROS2 backend
        # or your own custom Backend implementation!
        self.backends = [
            PX4MavlinkBackend(
                config=PX4MavlinkBackendConfig({"input_scaling": [900, 900, 900, 900]})
            )
        ]


class LAV2(Vehicle):
    """
    LAV2 vehicle class - A lightweight multirotor vehicle implementation for simulation.
    
    Inherits from the Vehicle base class and implements dynamics, control interfaces,
    and physics integration for quadrotor simulation.
    """

    def __init__(
        self,
        # Simulation specific configurations
        stage_prefix: str = "LAV2",
        usd_file: str = "",
        vehicle_id: int = 0,
        # Spawning pose of the vehicle
        init_pos=[0.0, 0.0, 0.12],
        init_orientation=[0.0, 0.0, 0.0, 1.0],
        config=LAV2Config(),
    ):
        """
        Initialize the LAV2 vehicle.

        Args:
            stage_prefix (str): Name of the vehicle in the simulator. Defaults to "LAV2".
            usd_file (str): USD file describing the vehicle's appearance and properties. Defaults to "".
            vehicle_id (int): Unique identifier for the vehicle. Defaults to 0.
            init_pos (list): Initial position in ENU convention [x, y, z] in meters. Defaults to [0.0, 0.0, 0.12].
            init_orientation (list): Initial orientation as quaternion [qx, qy, qz, qw]. Defaults to [0.0, 0.0, 0.0, 1.0].
            config (LAV2Config, optional): Configuration object. Defaults to LAV2Config().
        """

        # 1. Initiate the Vehicle object itself
        super().__init__(
            stage_prefix,
            usd_file,
            init_pos,
            init_orientation,
            config.sensors,
            config.graphical_sensors,
            config.graphs,
            config.backends,
        )

        # 2. Setup the dynamics of the system - get the thrust curve of the vehicle from the configuration
        self._thrusters = config.thrust_curve
        self._drag = config.drag

    def start(self):
        """In this case we do not need to do anything extra when the simulation starts"""
        pass

    def stop(self):
        """In this case we do not need to do anything extra when the simulation stops"""
        pass

    def update(self, dt: float):
        """
        Compute and apply forces to the vehicle based on motor commands.
        
        This method is called on every physics simulation step. It retrieves rotor
        commands from the backend, computes thrust and torque, applies forces to the
        rotors, and updates drag compensation.

        Args:
            dt (float): Time elapsed since the last physics step (seconds).
        """

        # Get the articulation root of the vehicle
        articulation = self.get_dc_interface().get_articulation(
            self._stage_prefix + "/LAV2/LAV2"
        )

        # Get the desired angular velocities for each rotor from the first backend (can be mavlink or other) expressed in rad/s
        if len(self._backends) != 0:
            desired_rotor_velocities = self._backends[0].input_reference()
        else:
            desired_rotor_velocities = [0.0 for i in range(self._thrusters._num_rotors)]

        # Input the desired rotor velocities in the thruster model
        self._thrusters.set_input_reference(desired_rotor_velocities)

        # Get the desired forces to apply to the vehicle
        forces_z, _, rolling_moment = self._thrusters.update(self._state, dt)

        # Apply force to each rotor
        for i in range(4):
            # Apply the force in Z on the rotor frame
            self.apply_force(
                [0.0, 0.0, forces_z[i]], body_part="/LAV2/rotor" + str(i + 1)
            )

            # Generate the rotating propeller visual effect
            self.handle_propeller_visual(i, forces_z[i], articulation)

        # Apply the torque to the body frame of the vehicle that corresponds to the rolling moment
        self.apply_torque([0.0, 0.0, rolling_moment], "/LAV2/LAV2")

        # Compute the total linear drag force to apply to the vehicle's body frame
        drag = self._drag.update(self._state, dt)
        self.apply_force(drag, body_part="/LAV2/LAV2")

        # Call the update methods in all backends
        for backend in self._backends:
            backend.update(dt)

    def handle_propeller_visual(self, rotor_number, force: float, articulation):
        """
        Update rotor joint velocity for visual animation based on applied force.
        
        This auxiliary method sets the joint velocity of each rotor to create a
        realistic spinning animation that corresponds to the actual thrust being applied.

        Args:
            rotor_number (int): Index of the rotor (0-3).
            force (float): Thrust force applied to the rotor (Newtons).
            articulation: The articulation group containing the rotor joints.
        """

        # Rotate the joint to yield the visual of a rotor spinning (for animation purposes only)
        joint = self.get_dc_interface().find_articulation_dof(
            articulation, "rotor" + str(rotor_number + 1) + "_joint"
        )

        # Spinning when armed but not applying force
        if 0.0 < force < 0.1:
            self.get_dc_interface().set_dof_velocity(
                joint, 5 * self._thrusters.rot_dir[rotor_number]
            )
        # Spinning when armed and applying force
        elif 0.1 <= force:
            self.get_dc_interface().set_dof_velocity(
                joint, 100 * self._thrusters.rot_dir[rotor_number]
            )
        # Not spinning
        else:
            self.get_dc_interface().set_dof_velocity(joint, 0.0)

    def force_and_torques_to_velocities(self, force: float, torque: np.ndarray):
        """
        Compute target rotor angular velocities from desired force and torque.
        
        This method converts total thrust and torque commands in the vehicle body frame
        to individual rotor angular velocities using the allocation matrix and thrust curve.
        
        Note: This method assumes a quadratic thrust curve. A general thrust allocation
        scheme may be adopted in future versions.

        Args:
            force (float): Desired thrust in the body frame Z-direction (Newtons).
            torque (np.ndarray): Desired torque vector in body frame [τx, τy, τz] (Newton-meters).

        Returns:
            np.ndarray: Target angular velocities for each rotor (rad/s).
        """

        # Get the body frame of the vehicle
        rb = self.get_dc_interface().get_rigid_body(self._stage_prefix + "/LAV2/LAV2")

        # Get the rotors of the vehicle
        rotors = [
            self.get_dc_interface().get_rigid_body(
                self._stage_prefix + "/LAV2/rotor" + str(i)
            )
            for i in range(self._thrusters._num_rotors)
        ]

        # Get the relative position of the rotors with respect to the body frame of the vehicle (ignoring the orientation for now)
        relative_poses = self.get_dc_interface().get_relative_body_poses(rb, rotors)

        # Define the alocation matrix
        aloc_matrix = np.zeros((4, self._thrusters._num_rotors))

        # Define the first line of the matrix (T [N])
        aloc_matrix[0, :] = np.array(self._thrusters._rotor_constant)

        # Define the second and third lines of the matrix (\tau_x [Nm] and \tau_y [Nm])
        aloc_matrix[1, :] = np.array(
            [
                relative_poses[i].p[1] * self._thrusters._rotor_constant[i]
                for i in range(self._thrusters._num_rotors)
            ]
        )
        aloc_matrix[2, :] = np.array(
            [
                -relative_poses[i].p[0] * self._thrusters._rotor_constant[i]
                for i in range(self._thrusters._num_rotors)
            ]
        )

        # Define the forth line of the matrix (\tau_z [Nm])
        aloc_matrix[3, :] = np.array(
            [
                self._thrusters._rolling_moment_coefficient[i]
                * self._thrusters._rot_dir[i]
                for i in range(self._thrusters._num_rotors)
            ]
        )

        # Compute the inverse allocation matrix to obtain squared angular velocities from thrust and torques
        aloc_inv = np.linalg.pinv(aloc_matrix)

        # Compute the target angular velocities (squared)
        squared_ang_vel = aloc_inv @ np.array([force, torque[0], torque[1], torque[2]])

        # Making sure that there is no negative value on the target squared angular velocities
        squared_ang_vel[squared_ang_vel < 0] = 0.0

        # Saturate rotor velocities while preserving their mutual relationships via normalization
        max_thrust_vel_squared = np.power(self._thrusters.max_rotor_velocity[0], 2)
        max_val = np.max(squared_ang_vel)

        if max_val >= max_thrust_vel_squared:
            normalize = np.maximum(max_val / max_thrust_vel_squared, 1.0)

            squared_ang_vel = squared_ang_vel / normalize

        # Compute the angular velocities for each rotor in [rad/s]
        ang_vel = np.sqrt(squared_ang_vel)

        return ang_vel

    def update_state(self, dt: float):
        """
        Retrieve and update the vehicle's state from the physics engine.
        
        This method is called at every physics step to synchronize the vehicle's state,
        including position, orientation, linear/angular velocities, and acceleration
        from the simulator with the internal state representation.

        Args:
            dt (float): Time elapsed since the last physics step (seconds).
        """

        # Get the body frame interface of the vehicle (this will be the frame used to get the position, orientation, etc.)
        body = self.get_dc_interface().get_rigid_body(self._stage_prefix + "/LAV2/LAV2")

        # Get the current position and orientation in the inertial frame
        pose = self.get_dc_interface().get_rigid_body_pose(body)

        # Get the attitude according to the convention [w, x, y, z]
        prim = self._world.stage.GetPrimAtPath(self._stage_prefix + "/LAV2/LAV2")
        rotation_quat = get_world_transform_xform(prim).GetQuaternion()
        rotation_quat_real = rotation_quat.GetReal()
        rotation_quat_img = rotation_quat.GetImaginary()

        # Get the angular velocity of the vehicle in the body frame of reference
        ang_vel = self.get_dc_interface().get_rigid_body_angular_velocity(body)

        # Linear velocity [x_dot, y_dot, z_dot] of the vehicle in the inertial frame
        linear_vel = self.get_dc_interface().get_rigid_body_linear_velocity(body)

        # Approximate linear acceleration from velocity difference
        # Note: Isaac Sim doesn't provide rigid body acceleration directly, so we approximate it
        linear_acceleration = (np.array(linear_vel) - self._state.linear_velocity) / dt

        # Update the state variable X = [x,y,z]
        self._state.position = np.array(pose.p)

        # Get the quaternion according in the [qx,qy,qz,qw] standard
        self._state.attitude = np.array(
            [
                rotation_quat_img[0],
                rotation_quat_img[1],
                rotation_quat_img[2],
                rotation_quat_real,
            ]
        )

        # Express the velocity of the vehicle in the inertial frame X_dot = [x_dot, y_dot, z_dot]
        self._state.linear_velocity = np.array(linear_vel)

        # Linear velocity [u, v, w] in the body frame of reference
        # Computed by rotating inertial frame velocity to body frame (inverse rotation)
        self._state.linear_body_velocity = (
            Rotation.from_quat(self._state.attitude)
            .inv()
            .apply(self._state.linear_velocity)
        )

        # omega = [p,q,r]
        self._state.angular_velocity = (
            Rotation.from_quat(self._state.attitude).inv().apply(np.array(ang_vel))
        )

        # The acceleration of the vehicle expressed in the inertial frame X_ddot = [x_ddot, y_ddot, z_ddot]
        self._state.linear_acceleration = linear_acceleration
