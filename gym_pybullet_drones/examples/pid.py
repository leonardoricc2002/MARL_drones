"""Script demonstrating the joint use of simulation and control.

MODIFIED VERSION: Rendez-vous -> V-Formation Spiral (World ON, Drones Hardcoded)
"""
import os
import time
import argparse
from datetime import datetime
import pdb
import math
import random
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool


# ====================================================================
# LE TUE IMPOSTAZIONI PERSONALI (Modifica direttamente qui!)
# ====================================================================
DEFAULT_NUM_DRONES = 5       # <--- SCRIVI QUI IL NUMERO DI DRONI (es. 3, 5, 7)
DEFAULT_OBSTACLES = True     # <--- QUESTO RIACCENDE IL MONDO 3D!
# ====================================================================


DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False
DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 25 
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
        ):
    
    #### POSIZIONI INIZIALI (Caos totale sul prato) #####
    INIT_XYZS = np.zeros((num_drones, 3))
    for j in range(num_drones):
        if j == 0:
            INIT_XYZS[j, :] = [0, 0, 0.1] 
        else:
            INIT_XYZS[j, :] = [random.uniform(-2.0, 2.0), random.uniform(-2.0, 2.0), 0.1]
            
    INIT_RPYS = np.zeros((num_drones, 3))

    env = CtrlAviary(drone_model=drone,
                        num_drones=num_drones,
                        initial_xyzs=INIT_XYZS,
                        initial_rpys=INIT_RPYS,
                        physics=physics,
                        neighbourhood_radius=10,
                        pyb_freq=simulation_freq_hz,
                        ctrl_freq=control_freq_hz,
                        gui=gui,
                        record=record_video,
                        obstacles=obstacles,
                        user_debug_gui=user_debug_gui
                        )

    PYB_CLIENT = env.getPyBulletClient()
    logger = Logger(logging_freq_hz=control_freq_hz, num_drones=num_drones, output_folder=output_folder, colab=colab)

    if drone in [DroneModel.CF2X, DroneModel.CF2P]:
        ctrl = [DSLPIDControl(drone_model=drone) for i in range(num_drones)]

    action = np.zeros((num_drones,4))
    START = time.time()
    
    #### LOOP DI SIMULAZIONE ###################################
    for i in range(0, int(duration_sec*env.CTRL_FREQ)):
        obs, reward, terminated, truncated, info = env.step(action)
        t = i / env.CTRL_FREQ
        
        current_targets = np.zeros((num_drones, 3))
        target_rpys = np.zeros((num_drones, 3))

        #### FASE 1: DECOLLO VERTICALE SUL POSTO (0 -> 2 secondi)
        if t < 2.0:
            for j in range(num_drones):
                current_targets[j, :] = [INIT_XYZS[j, 0], INIT_XYZS[j, 1], 1.0 * (t/2.0)]
            theta = 0
            
        #### FASE 2: RENDEZ-VOUS A FORMA DI V (2 -> 6 secondi)
        elif t < 6.0:
            progress = (t - 2.0) / 4.0 
            leader_x, leader_y, leader_z = INIT_XYZS[0, 0], INIT_XYZS[0, 1], 1.0
            theta = 0
            
            for j in range(num_drones):
                if j == 0:
                    current_targets[j, :] = [leader_x, leader_y, leader_z]
                else:
                    dist_back = 0.6 * ((j + 1) // 2)
                    dist_side = 0.6 * ((j + 1) // 2)
                    sign = 1 if j % 2 != 0 else -1
                    
                    v_target_x = leader_x - dist_back
                    v_target_y = leader_y + (dist_side * sign)
                    
                    curr_x = INIT_XYZS[j, 0] + (v_target_x - INIT_XYZS[j, 0]) * progress
                    curr_y = INIT_XYZS[j, 1] + (v_target_y - INIT_XYZS[j, 1]) * progress
                    
                    current_targets[j, :] = [curr_x, curr_y, leader_z]

        #### FASE 3: LA SPIRALE (da 6 secondi in poi)
        else:
            t_spiral = t - 6.0 
            radius = 0.0 + 0.15 * t_spiral 
            angle = 0.8 * t_spiral
            
            leader_x = INIT_XYZS[0, 0] + radius * np.cos(angle)
            leader_y = INIT_XYZS[0, 1] + radius * np.sin(angle)
            leader_z = min(1.0 + 0.1 * t_spiral, 2.5) 
            
            v_x = 0.15 * np.cos(angle) - radius * 0.8 * np.sin(angle)
            v_y = 0.15 * np.sin(angle) + radius * 0.8 * np.cos(angle)
            theta = np.arctan2(v_y, v_x) 

            for j in range(num_drones):
                target_rpys[j, :] = [0, 0, theta]
                if j == 0:
                    current_targets[j, :] = [leader_x, leader_y, leader_z]
                else:
                    dist_back = 0.6 * ((j + 1) // 2)
                    dist_side = 0.6 * ((j + 1) // 2)
                    sign = 1 if j % 2 != 0 else -1
                    
                    base_dx = -dist_back
                    base_dy = dist_side * sign
                    
                    rot_dx = base_dx * np.cos(theta) - base_dy * np.sin(theta)
                    rot_dy = base_dx * np.sin(theta) + base_dy * np.cos(theta)
                    
                    current_targets[j, :] = [leader_x + rot_dx, leader_y + rot_dy, leader_z]

        # Applico i comandi
        for j in range(num_drones):
            action[j, :], _, _ = ctrl[j].computeControlFromState(
                                            control_timestep=env.CTRL_TIMESTEP,
                                            state=obs[j],
                                            target_pos=current_targets[j, :],
                                            target_rpy=target_rpys[j, :]
                                        )
            logger.log(drone=j, timestamp=t, state=obs[j], control=np.hstack([current_targets[j, :], target_rpys[j, :], np.zeros(6)]))

        #### TELECAMERA TRACKING ####
        if gui and i % 5 == 0: 
            p.resetDebugVisualizerCamera(cameraDistance=3.5, 
                                         cameraYaw=-45, 
                                         cameraPitch=-30, 
                                         cameraTargetPosition=[current_targets[0, 0], current_targets[0, 1], current_targets[0, 2]], 
                                         physicsClientId=PYB_CLIENT)

        env.render()
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    env.close()
    logger.save()
    logger.save_as_csv("pid") 
    
    if plot:
        logger.plot()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rendezvous to Spiral with Tracking Camera')
    parser.add_argument('--drone',              default=DEFAULT_DRONES,     type=DroneModel,    help='Drone model', metavar='', choices=DroneModel)
    parser.add_argument('--num_drones',         default=DEFAULT_NUM_DRONES, type=int,           help='Number of drones', metavar='')
    parser.add_argument('--physics',            default=DEFAULT_PHYSICS,    type=Physics,       help='Physics updates', metavar='', choices=Physics)
    parser.add_argument('--gui',                default=DEFAULT_GUI,        type=str2bool,      help='Use GUI', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VISION, type=str2bool,   help='Record video', metavar='')
    parser.add_argument('--plot',               default=DEFAULT_PLOT,       type=str2bool,      help='Plot results', metavar='')
    parser.add_argument('--user_debug_gui',     default=DEFAULT_USER_DEBUG_GUI, type=str2bool,  help='Debug lines', metavar='')
    parser.add_argument('--obstacles',          default=DEFAULT_OBSTACLES,  type=str2bool,      help='Add obstacles', metavar='')
    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ, type=int,   help='Sim freq', metavar='')
    parser.add_argument('--control_freq_hz',    default=DEFAULT_CONTROL_FREQ_HZ, type=int,      help='Ctrl freq', metavar='')
    parser.add_argument('--duration_sec',       default=DEFAULT_DURATION_SEC, type=int,         help='Duration', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,        help='Folder', metavar='')
    parser.add_argument('--colab',              default=DEFAULT_COLAB, type=bool,               help='Colab mode', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))