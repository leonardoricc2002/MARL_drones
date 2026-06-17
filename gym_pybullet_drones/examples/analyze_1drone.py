import time
import numpy as np
import pybullet as p 
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# ==============================================================================
# IMPORTA GLI STRUMENTI DAL TUO AMBIENTE ORIGINALE
# ==============================================================================
from gym_pybullet_drones.envs.OttoAviary import OttoAviary
from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# ==============================================================================
# CONFIGURAZIONE DEL TEST 
# ==============================================================================
# ⚠️ ATTENZIONE: INSERISCI QUI IL PERCORSO DEL MODELLO ADDESTRATO CON 1 DRONE E AZIONE PID!
MODEL_PATH = "results/Hovering_in_xyz/best_model.zip" 

OBS_TYPE = ObservationType.KIN      # 'kin'
ACT_TYPE = ActionType.PID           # Azione in PID (Roll, Pitch, Yaw, Thrust)

def run_evaluation():
    print(f"[INFO] Caricamento modello pre-addestrato da: {MODEL_PATH}")
    
    try:
        model = PPO.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"[ERRORE] Non trovo il file {MODEL_PATH}. Controlla il percorso!")
        return

    # 1. INIZIALIZZA L'AMBIENTE (HoverAviary standard, 1 Drone)
    env = OttoAviary(
        gui=True, 
        record=False, 
        obs=OBS_TYPE, 
        act=ACT_TYPE
    )
    EPISODE_STEPS = env.EPISODE_LEN_SEC * env.CTRL_FREQ 
    
    # Liste di log per 1 singolo drone
    time_axis = []
    log_z = []      
    log_err_xy = [] 
    log_roll = []   
    log_mean_rpm = [] 
    
    obs, info = env.reset()
    start_time = time.time()

    # Imposta la telecamera per guardare bene il singolo drone
    p.resetDebugVisualizerCamera(cameraDistance=3, cameraYaw=45, cameraPitch=-30, 
                                 cameraTargetPosition=[0, 0, 1.0], physicsClientId=env.CLIENT)

    print("\n[INFO] Simulazione 3D in corso... Modello Singolo Agente (PID) caricato con successo!")
    
    # 2. CICLO DI SIMULAZIONE
    for step in range(EPISODE_STEPS):
        
        # Predizione dell'azione (output in Roll, Pitch, Yaw e Thrust data l'impostazione PID)
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        current_time = step / env.CTRL_FREQ
        time_axis.append(current_time)
        
        # Estrazione telemetrica per il drone singolo (indice 0)
        state = env._getDroneStateVector(0)
        
        log_z.append(state[2])
        log_err_xy.append(np.sqrt(state[0]**2 + state[1]**2))
        log_roll.append(np.degrees(state[7])) # Stato cinematico dell'assetto (Roll)
        
        # Giri motore reali
        motori = state[16:20]
        log_mean_rpm.append(np.mean(motori))
        
        sync(step, start_time, env.CTRL_TIMESTEP)
        if terminated or truncated:
            break
            
    env.close()
    print("[INFO] Simulazione completata. Generazione dei grafici in corso...")

    # ==============================================================================
    # 3. PANNELLO DEI GRAFICI PER LA TESI (SINGOLO DRONE)
    # ==============================================================================
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10), sharex='col')
    
    color = '#1f77b4' # Blu accademico
    label = 'Singolo Agente'
    
    cut = 50 
    t_plot = time_axis[cut:]

    # --- Q1: ALTITUDINE (Z) ---
    ax1.plot(t_plot, log_z[cut:], label=label, color=color, linewidth=2)
    # Aggiungo la linea di target dell'HoverAviary (di solito z=1.0 per la baseline)
    ax1.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Target (1.0m)')
    ax1.set_ylabel('Altitude Z [m]', fontsize=12)
    ax1.set_title('Vertical Position Tracking', fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower right')
    
    # --- Q2: DERIVA PLANARE (XY) ---
    ax2.plot(t_plot, log_err_xy[cut:], label=label, color=color, linewidth=1.5)
    ax2.set_ylabel('XY Drift [m]', fontsize=12)
    ax2.set_title('Planar Error (X-Y Plane)', fontsize=13, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # --- Q3: INCLINAZIONE (ROLL) ---
    ax3.plot(t_plot, log_roll[cut:], label=label, color=color, linewidth=1.2, alpha=0.8)
    ax3.set_ylabel('Roll Angle [deg]', fontsize=12)
    ax3.set_xlabel('Time [s]', fontsize=12)
    ax3.set_title('Attitude Response', fontsize=13, fontweight='bold')
    ax3.grid(True, linestyle=':', alpha=0.6)
    
    # --- Q4: MEDIA 4 MOTORI (RPM) ---
    ax4.plot(t_plot, log_mean_rpm[cut:], label=label, color=color, linewidth=1.2, alpha=0.8)
    ax4.set_ylabel('Average RPM (All 4 Motors)', fontsize=12)
    ax4.set_xlabel('Time [s]', fontsize=12)
    ax4.set_title('Average Actuator Effort', fontsize=13, fontweight='bold')   
    ax4.grid(True, linestyle=':', alpha=0.6)
    
    fig.suptitle('Single Agent Hovering Analysis (Action: PID)', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.show()

if __name__ == '__main__':
    run_evaluation()