import time
import numpy as np
import pybullet as p
import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from gym_pybullet_drones.envs.OttoAviary import OttoAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

MODEL_PATH = "results/Hovering_in_xyz/best_model.zip" 

def is_out_of_bounds(state):
    # Returns True if safety boundaries are exceeded
    if (abs(state[0]) > 3.0 or abs(state[1]) > 3.0 or state[2] > 2.5 
        or abs(state[7]) > 2.8 or abs(state[8]) > 2.8):
        return True
    return False

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at: {MODEL_PATH}")
        return

    print(f"[INFO] Loading model from: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH)

    test_env = OttoAviary(gui=True, obs=ObservationType('kin'), act=ActionType('pid'))
    test_env.EPISODE_LEN_SEC = 17.0

    obs, info = test_env.reset(seed=42, options={})
    
    p.resetDebugVisualizerCamera(cameraDistance=3.5, cameraYaw=2, cameraPitch=-35, 
                                 cameraTargetPosition=[0, 0, 1], physicsClientId=test_env.CLIENT)

    start = time.time()
    num_steps = int(test_env.EPISODE_LEN_SEC * test_env.CTRL_FREQ)
    start_wind_time = 0.0 

    print(f"[INFO] Simulation started. Constant wind active after {start_wind_time} seconds.")

    # Variables for plotting
    log_time = []
    log_pos = []
    log_rpy = []
    log_angvel = []
    log_rpm = []

    try:
        for i in range(num_steps):
            action, _states = model.predict(obs, deterministic=True)
            
            # --- WIND DISTURBANCE ---
            current_time = i / test_env.CTRL_FREQ
            if current_time > start_wind_time:
                # Constant light wind in positive Y direction
                wind_x = 0.00
                wind_y = 0.003
                p.applyExternalForce(test_env.DRONE_IDS[0], -1, [wind_x, wind_y, 0], [0,0,0], p.WORLD_FRAME)
            
            obs, reward, terminated, truncated, info = test_env.step(action)
            
            # --- DATA LOGGING ---
            real_state = test_env._getDroneStateVector(0)
            
            log_time.append(current_time)
            log_pos.append(real_state[0:3])      # X, Y, Z
            log_rpy.append(real_state[7:10])     # Roll, Pitch, Yaw
            log_angvel.append(real_state[13:16]) # wx, wy, wz
            log_rpm.append(real_state[16:20])    # RPM0, RPM1, RPM2, RPM3
            
            if is_out_of_bounds(real_state):
                print(f"[WARNING] Safety limits exceeded at {current_time:.2f}s - Drone unstable!")
            
            test_env.render()
            sync(i, start, test_env.CTRL_TIMESTEP)

            if terminated or truncated:
                print(f"[INFO] Environment simulation concluded (Terminated: {terminated}, Truncated: {truncated})")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Simulation interrupted manually.")
        
    finally:
        test_env.close()
        
        # ==========================================
        # ACADEMIC PAPER PLOT GENERATION (2x2 Grid)
        # ==========================================
        print("[INFO] Generating high-resolution plots...")
        
        log_time = np.array(log_time)
        log_pos = np.array(log_pos)
        log_rpy = np.array(log_rpy)
        log_angvel = np.array(log_angvel)
        log_rpm = np.array(log_rpm)
        
        # Create 2x2 grid (sharex='col' aligns the X axis vertically)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10), sharex='col')
        
        # ---------------------------------------------------------
        # TOP-LEFT: SPATIAL POSITION (X, Y, Z)
        # ---------------------------------------------------------
        ax1.plot(log_time, log_pos[:, 0], label='X', color='tomato', linewidth=2)
        ax1.plot(log_time, log_pos[:, 1], label='Y', color='mediumseagreen', linewidth=2)
        ax1.plot(log_time, log_pos[:, 2], label='Z', color='royalblue', linewidth=2)
        ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, label='Target (1.0m)')
        ax1.axvline(x=start_wind_time, color='black', linestyle=':', linewidth=2, label='Wind Onset')
        ax1.set_ylabel('Position [m]', fontsize=12)
        ax1.set_title('Spatial Trajectory', fontsize=13, fontweight='bold')
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.legend(loc='best')
        
        # ---------------------------------------------------------
        # TOP-RIGHT: EULER ANGLES (Roll, Pitch, Yaw)
        # ---------------------------------------------------------
        ax2.plot(log_time, np.degrees(log_rpy[:, 0]), label='Roll', color='tomato', linewidth=1.5)
        ax2.plot(log_time, np.degrees(log_rpy[:, 1]), label='Pitch', color='mediumseagreen', linewidth=1.5)
        ax2.plot(log_time, np.degrees(log_rpy[:, 2]), label='Yaw', color='royalblue', linewidth=1.5)
        ax2.axvline(x=start_wind_time, color='black', linestyle=':', linewidth=2)
        ax2.set_ylabel('Attitude [deg]', fontsize=12)
        ax2.set_title('Euler Angles', fontsize=13, fontweight='bold')
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend(loc='best')

        # ---------------------------------------------------------
        # BOTTOM-LEFT: ANGULAR VELOCITIES
        # ---------------------------------------------------------
        ax3.plot(log_time, log_angvel[:, 0], label='wx', color='tomato', linewidth=1)
        ax3.plot(log_time, log_angvel[:, 1], label='wy', color='mediumseagreen', linewidth=1)
        ax3.plot(log_time, log_angvel[:, 2], label='wz', color='royalblue', linewidth=1)
        ax3.axvline(x=start_wind_time, color='black', linestyle=':', linewidth=2)
        ax3.set_ylabel('Angular Velocity [rad/s]', fontsize=12)
        ax3.set_xlabel('Time [s]', fontsize=12)
        ax3.set_title('Rotational Dynamics', fontsize=13, fontweight='bold')
        ax3.grid(True, linestyle=':', alpha=0.6)
        ax3.legend(loc='best')

        # ---------------------------------------------------------
        # BOTTOM-RIGHT: ACTUATOR EFFORT (RPM)
        # ---------------------------------------------------------
        ax4.plot(log_time, log_rpm[:, 0], label='Motor 1', color='tomato', linewidth=1, alpha=0.8)
        ax4.plot(log_time, log_rpm[:, 1], label='Motor 2', color='mediumseagreen', linewidth=1, alpha=0.8)
        ax4.plot(log_time, log_rpm[:, 2], label='Motor 3', color='royalblue', linewidth=1, alpha=0.8)
        ax4.plot(log_time, log_rpm[:, 3], label='Motor 4', color='orange', linewidth=1, alpha=0.8)
        
        # Exact physical saturation limit from the physics engine
        max_rpm_esatto = test_env.MAX_RPM 
        ax4.axhline(y=max_rpm_esatto, color='red', linestyle='--', alpha=0.8, 
                    label=f'Max Saturation ({int(max_rpm_esatto)} RPM)')
        
        ax4.axvline(x=start_wind_time, color='black', linestyle=':', linewidth=2)
        ax4.set_ylabel('Motor Speed [RPM]', fontsize=12)
        ax4.set_xlabel('Time [s]', fontsize=12)
        ax4.set_title('Control Effort (Actuators)', fontsize=13, fontweight='bold')
        ax4.grid(True, linestyle=':', alpha=0.6)
        ax4.legend(loc='best')
        
        # Main title for the entire figure
        fig.suptitle('Robustness Analysis under Constant Wind Disturbance', fontsize=16, fontweight='bold')
        
        # Auto-adjust spacing
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save output
        save_path = os.path.join('results', 'grid_kinematic_wind_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] High-resolution plot saved: {save_path}")
        
        plt.show()

if __name__ == '__main__':
    main()