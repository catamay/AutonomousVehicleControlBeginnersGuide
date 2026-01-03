"""
mpc_controller.py

Author: Aidan Copinga
"""

#import path setting
import sys
from pathlib import Path
from math import tan, atan2
import numpy as np
import scipy.linalg as la

abs_dir_path = str(Path(__file__).absolute().parent)
relative_path = "/../../../components/"

sys.path.append(abs_dir_path + relative_path + "common")

#import component modules
from angle_lib import pi_to_pi
# These are required libraries, which makes this fork infeasible to push, but it's a good addition regardless
import cvxopt
from cvxopt import matrix
from cvxopt.solvers import qp

class MpcController():
    def __init__(self, spec, control_const, course=None, horizon=100, state_const=None):
        """
        Constructor

        spec: VehicleSpecification object
        control_const: 2x2 array of interval constraints for heading and acceleration
        course: Course data and logic object
        horizon: Prediction horizon, default is 100 steps 
        state_const: 1x2 array of an interval for all state variables, optional
        """
        self.horizon = horizon

        self.control_const = np.array(control_const) 
        self.state_const = state_const 
        self.n_u = 2 # Control dimension

        self.WHEEL_BASE_M = spec.wheel_base_m
        self.SCALE_R = 4
        self.WEIGHT_MAT_Q = 1*np.eye(5)
        self.WEIGHT_MAT_R = self.SCALE_R * np.eye(2)
        self.MAX_ITERATION = 150
        self.THRESHOLD = 0.01

        self.course = course
        self.target_course_index = 0
        self.target_accel_mps2 = 0.0
        self.target_speed_mps = 0.0
        self.target_yaw_rad = 0.0
        self.target_yaw_rate_rps = 0.0
        self.target_steer_rad = 0.0

        self.prev_error_lat_m = 0.0
        self.prev_error_yaw_rad = 0.0

        self.solver = self._mpc_solver(self)

    class _mpc_solver():
        """
        Private class to initialize and solve the QP problem for model predictive control (MPC)
        """
        def __init__(self, controller):
            """

            controller: mpc object to take local variables from
            """
            cvxopt.solvers.options['show_progress'] = False
            cvxopt.solvers.options['maxiters'] = 20

            self.only_control = controller.state_const is None
            if not self.only_control:
                self.state_const = controller.state_const
            
            self.horizon = controller.horizon
            self.n_u = controller.n_u

            self.Q_lift, self.R_lift = self._lift_QR(controller.WEIGHT_MAT_Q, controller.WEIGHT_MAT_R)
            
            min_U = np.repeat(controller.control_const[0], self.n_u*self.horizon)
            max_U = np.repeat(controller.control_const[1], self.n_u*self.horizon)

            self.AA = np.concatenate([np.eye(self.horizon*self.n_u), -np.eye(self.horizon*self.n_u)], axis=0)

            self.bb = np.concatenate([max_U, -min_U],axis=0)

            self.QQ = np.zeros((self.n_u*(self.horizon), self.n_u*(self.horizon)))
            self.CC = np.zeros((controller.WEIGHT_MAT_Q.shape[0] * (self.horizon+1), self.n_u*(self.horizon)))
            self.previous_x = np.zeros((controller.WEIGHT_MAT_Q.shape[0], 1))

        def _lift_QR(self, Q, R):
            """
            Private function to create lifted Q and R matrix based on horizon. 

            Q: nxn weight matrix on state
            R: mxm weight matrix on control
            """

            # Compute lifted Q matrix
            ones_Q = np.ones((self.horizon+1))
            ones_Q = np.diag(ones_Q)

            ones_S = np.eye(self.horizon+1) - ones_Q
            Q_lift = np.kron(ones_Q, Q)

            # Compute lifted R matrix
            R_lift = np.kron(np.eye(self.horizon), R)

            return Q_lift, R_lift

        def _lift_AB(self, A,B):
            """
            Private function to create lifted A and B matrix based on horizon.

            A: nxn state matrix
            B: nxm control matrix
            """

            n = A.shape[0]
            m = B.shape[1]

            B_lift = np.concatenate([np.zeros((n,m)), np.eye(n) @ B],axis=0)
            A_lift = np.concatenate([np.eye(n), A], axis=0)

            for i in range(1,self.horizon):
                B_lift = np.concatenate([np.zeros((n, m*(i+1))), np.concatenate([A_lift  @ B, B_lift], axis=1)],axis=0)
                A_lift = np.concatenate([np.eye(n), A_lift @ A],axis=0)

            return A_lift, B_lift 

        def set_params(self, A, B, state, future_setpoints):
            """
            Creates quadratic program parameters based on provided state space matrices, current state, and future points across the horizon.

            A: nxn state matrix
            B: nxm control matrix
            state: nx1 state vector
            future_setpoints: nx(horizon+1)x1 matrix of expected state vectors
            """
            A_lift, B_lift = self._lift_AB(A, B)
            
            goal_horizon = np.zeros((state.shape[0],(self.horizon + 1)))
            goal_horizon[:, 0:future_setpoints.shape[1]] = future_setpoints[:,:,0]
            if future_setpoints.shape[1] != self.horizon+1:
                goal_horizon[:,future_setpoints.shape[1]:] = np.repeat(future_setpoints[:,-1,:],goal_horizon.shape[1]-future_setpoints.shape[1], axis=1)

            goal_horizon = goal_horizon.flatten(order='F').reshape((-1,1))

            if not self.only_control:
                AA = [B_lift, -B_lift, np.eye(self.horizon*self.n_u), -np.eye(self.horizon*self.n_u)]

                min_X = np.tile(self.state_const[:,0], (self.horizon + 1))
                max_X = np.tile(self.state_const[:,1], (self.horizon + 1))
                end = self.bb[-2*self.horizon*self.n_u:]

                self.bb = np.concatenate([max_X - A_lift @ state[:,0], -min_X + A_lift @ state[:,0], end],axis=0)
                self.AA = np.concatenate(AA, axis=0)

            QQ = B_lift.T @ self.Q_lift @ B_lift + self.R_lift

            self.QQ = (QQ + QQ.T)/2 # Force symmetry for floating point arithmetic
            self.CC = (A_lift @ state - goal_horizon).T @ self.Q_lift @ B_lift


        def solve(self):
            """
            Minimize (1/2) U' QQ U + CC U subject to AA U <= bb
            """
            P = matrix(self.QQ)
            G = matrix(self.AA)
            h = matrix(self.bb, tc='d')
            q = matrix(self.CC.T)

            try:
                qp_res = qp(P, q, G, h)
                if qp_res['status'] == 'optimal':
                    x_val = qp_res['x']
                    self.previous_x = x_val
                else: 
                    x_val = np.zeros_like(self.previous_x)
            except:
                x_val = np.zeros_like(self.previous_x)
           

            control_input = np.array(x_val).reshape((-1,1))

            return control_input


    def _calculate_feedback(self, A, B, state, setpoint, full_scale):
        """
        Private function to initialize and solve the quadratic program
        cost = sum(x[t].T * Q * x[t] + u[t].T * R * u[t])
        A: Matrix A in state equation
        B: Matrix B in state equation
        state: current state
        full_scale: boolean to provide full future horizon feedback info
        """

        self.solver.set_params(A, B, state, setpoint)
        feedback = self.solver.solve()
        if not full_scale:
            feedback = feedback[0:self.n_u,:]

        return feedback

    def _calculate_target_course_index(self, state):
        """
        Private function to calculate target point's index on course
        state: Vehicle's state object
        """

        nearest_index = self.course.search_nearest_point_index(state)
        self.target_course_index = max(self.target_course_index, nearest_index)

    def _calculate_tracking_error(self, state):
        """
        Private function to calculate tracking error against target point on the course
        state: Vehicle's state object
        """
        error_lon_m, error_lat_m, error_yaw_rad = self.course.calculate_lonlat_error(state, self.target_course_index)

        return error_lon_m, error_lat_m, error_yaw_rad

    def _decide_target_speed_mps(self):
        """
        Private function to decide target speed[m/s] over horizon+1 indices of the course
        """
        self.target_speed_mps = self.course.point_speed_mps(slice(self.target_course_index,self.target_course_index+self.horizon+1))

    def _decide_target_yaw(self):
        """
        Private function to decide target yaw[rad] over horizon+1 indices of the course
        """
        self.target_yaw_rad = self.course.point_yaw_rad(slice(self.target_course_index,self.target_course_index+self.horizon+1))

    def _decide_target_curv(self):
        """
        Private function to decide target curvature over horizon+1 indices of the course
        """
        return self.course.point_curvature(slice(self.target_course_index,self.target_course_index+self.horizon+1))

    def _calculate_control_input(self, state, error_lat_m, error_yaw_rad, time_s, full_scale=False):
        """
        Private function to calculate yaw rate input
        state: Vehicle's state object
        error_lat_m: Lateral error against reference course[m]
        error_yaw_rad: Yaw angle error against reference course[rad]
        time_s: Simulation interval time[sec]
        full_scale: boolean to provide full future horizon feedback info
        """
        curr_spd = state.get_speed_mps()
        trgt_curv = self._decide_target_curv()

        # A = [1.0, dt, 0.0, 0.0, 0.0
        #      0.0, 0.0, v, 0.0, 0.0]
        #      0.0, 0.0, 1.0, dt, 0.0]
        #      0.0, 0.0, 0.0, 0.0, 0.0]
        #      0.0, 0.0, 0.0, 0.0, 1.0]
        A = np.zeros((5, 5))
        A[0, 0] = 1.0
        A[0, 1] = time_s
        A[1, 2] = curr_spd
        A[2, 2] = 1.0
        A[2, 3] = time_s
        A[4, 4] = 1.0

        # B = [0.0, 0.0
        #     0.0, 0.0
        #     0.0, 0.0
        #     v/L, 0.0
        #     0.0, dt]
        B = np.zeros((5, 2))
        B[3, 0] = curr_spd / self.WHEEL_BASE_M
        B[4, 1] = time_s

        # state vector
        # setpoint = [0, 0, 0, 0, setpoint speed]
        # x = [lat error, lat error/s, yaw error, yaw error/s, cur_speed]

        x = np.zeros((5, 1)) 
        x[0, 0] = error_lat_m # lateral error against course
        x[1, 0] = (error_lat_m - self.prev_error_lat_m) / time_s # derivative of lateral error
        x[2, 0] = error_yaw_rad # yaw angle error against course
        x[3, 0] = (error_yaw_rad - self.prev_error_yaw_rad) / time_s # derivative of yaw angle error
        x[4, 0] = curr_spd  # current speed

        # generate future setpoints
        future_speeds = np.array(self.target_speed_mps).reshape(-1,1)

        # Future states is [0,0,0,0,speed(t)] as dynamics allow for speed constraints
        future_state = np.concat([[np.zeros_like(future_speeds)], [np.zeros_like(future_speeds)], [np.zeros_like(future_speeds)], [np.zeros_like(future_speeds)], [future_speeds]],axis=0)


        self.prev_error_lat_m = error_lat_m
        self.prev_error_yaw_rad = error_yaw_rad


        # feedback input vector
        # [[steering angle],
        #  [acceleration]]
        feedback_input = self._calculate_feedback(A, B, x, future_state, full_scale)


        # target steering angle
        feedforward_steer = atan2(self.WHEEL_BASE_M * trgt_curv[0], 1)

        feedback_steer = pi_to_pi(feedback_input[0, 0])

        self.target_steer_rad = feedforward_steer + feedback_steer
        # target yaw rate
        self.target_yaw_rate_rps = curr_spd * tan(self.target_steer_rad) / self.WHEEL_BASE_M

        # target acceleration
        self.target_accel_mps2 = feedback_input[1, 0]

    def update(self, state, time_s):
        """
        Function to update data for path tracking
        state: Vehicle's state object
        time_s: Simulation interval time[sec]
        """

        if not self.course: return

        self._calculate_target_course_index(state)

        self._decide_target_speed_mps()
        self._decide_target_yaw()

        _, error_lat_m, error_yaw_rad = self._calculate_tracking_error(state)

        self._calculate_control_input(state, error_lat_m, error_yaw_rad, time_s)


    def get_target_accel_mps2(self):
        """
        Function to get acceleration input[m/s2]
        """
        
        return self.target_accel_mps2

    def get_target_yaw_rate_rps(self):
        """
        Function to get yaw rate input[rad/s]
        """

        return self.target_yaw_rate_rps

    def get_target_steer_rad(self):
        """
        Function to get steering angle input[rad]
        """
        
        return self.target_steer_rad

    def draw(self, axes, elems):
        """
        Function to draw target point on course
        axes: Axes object of figure
        elems: plot object's list
        """

        pass