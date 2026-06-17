import time
import numpy as np
import pybullet as p 
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# ==============================================================================
# IMPORTA GLI STRUMENTI DAL TUO AMBIENTE ORIGINALE
# ==============================================================================
from gym_pybullet_drones.envs.LeaderFollowerAviary import LeaderFollowerAviary
from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

# ==============================================================================
# CONFIGURAZIONE DEL TEST (Sintonizzata su learn.py)
# ==============================================================================
# INSERISCI QUI IL PERCORSO DEL MODELLO MULTI-AGENTE CORRETTO!
MODEL_PATH = "results/Hover3_vel_nowind_stabile/best_model.zip"     

NUM_DRONES = 3         
OBS_TYPE = ObservationType.KIN      # 'kin'
ACT_TYPE = ActionType.VEL           # Identico a learn.py

def run_evaluation():
    print(f"[INFO] Caricamento modello pre-addestrato da: {MODEL_PATH}")
    
    try:
        model = PPO.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"[ERRORE] Non trovo il file {MODEL_PATH}. Controlla il percorso!")
        return

    # 1. INIZIALIZZA L'AMBIENTE CON I PARAMETRI REALI DI ADDESTRAMENTO
    env = LeaderFollowerAviary(
        gui=True, 
        num_drones=NUM_DRONES, 
        record=False, 
        obs=OBS_TYPE, 
        act=ACT_TYPE
    )
    EPISODE_STEPS = env.EPISODE_LEN_SEC * env.CTRL_FREQ 
    
    # Dizionari di log dinamici in base a NUM_DRONES (evita il KeyError)
    time_axis = []
    log_z = {i: [] for i in range(NUM_DRONES)}      
    log_err_xy = {i: [] for i in range(NUM_DRONES)} 
    log_roll = {i: [] for i in range(NUM_DRONES)}   
    log_mean_rpm = {i: [] for i in range(NUM_DRONES)} 
    
    obs, info = env.reset()
    start_time = time.time()

    p.resetDebugVisualizerCamera(cameraDistance=3.5, cameraYaw=45, cameraPitch=-30, 
                                 cameraTargetPosition=[0, 0, 0.9], physicsClientId=env.CLIENT)

    print(f"\n[INFO] Simulazione 3D in corso... Modello {NUM_DRONES}-Agent caricato con successo!")
    
    # 2. CICLO DI SIMULAZIONE
    for step in range(EPISODE_STEPS):
        
        # Sfruttiamo le predizioni del modello sull'osservazione nativa fornita dall'ambiente
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        current_time = step / env.CTRL_FREQ
        time_axis.append(current_time)
        
        # Estrazione telemetrica per il plot della tesi
        for d in range(NUM_DRONES): 
            state = env._getDroneStateVector(d)
            
            log_z[d].append(state[2])
            log_err_xy[d].append(np.sqrt(state[0]**2 + state[1]**2))
            log_roll[d].append(np.degrees(state[7])) # Stato cinematico dell'assetto (Roll)
            
            # Leggiamo i giri motore reali generati dall'azione ad una dimensione
            motori = state[16:20]
            log_mean_rpm[d].append(np.mean(motori))
        
        sync(step, start_time, env.CTRL_TIMESTEP)
        if terminated or truncated:
            break
            
    env.close()
    print("[INFO] Simulazione completata. Generazione dei grafici in corso...")

    # ==============================================================================
    # 3. PANNELLO DEI GRAFICI PER LA TESI (DINAMICO)
    # ==============================================================================
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10), sharex='col')
    
    # Setup dinamico di colori e labels in base al numero di droni
    colors = ['#d62728', '#2ca02c', '#1f77b4', '#ff7f0e', '#9467bd'][:NUM_DRONES] 
    labels = [f'Drone {i}' for i in range(NUM_DRONES)]
    labels[0] = 'Leader (drone_0)'
    if NUM_DRONES > 1:
        labels[1] = 'Follower 1 (drone_1)'
    if NUM_DRONES > 2:
        labels[2] = 'Follower 2 (drone_2)'
    linestyles = ['-', '--', '-.', ':', '-'][:NUM_DRONES]
    
    cut = 50 
    t_plot = time_axis[cut:]

    # --- Q1: ALTITUDINE (Z) ---
    for d in range(NUM_DRONES):
        ax1.plot(t_plot, log_z[d][cut:], label=labels[d], color=colors[d], linestyle=linestyles[d], linewidth=2)
    ax1.set_ylabel('Altitude Z [m]', fontsize=12)
    ax1.set_title('Vertical Position (Ceiling Effect)', fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower right')
    
    # --- Q2: DERIVA PLANARE (XY) ---
    for d in range(NUM_DRONES):
        ax2.plot(t_plot, log_err_xy[d][cut:], label=labels[d], color=colors[d], linestyle=linestyles[d], linewidth=1.5)
    ax2.set_ylabel('XY Drift [m]', fontsize=12)
    ax2.set_title('Planar Error (Wake Turbulence Propagation)', fontsize=13, fontweight='bold')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    # --- Q3: INCLINAZIONE (ROLL) ---
    for d in range(NUM_DRONES):
        ax3.plot(t_plot, log_roll[d][cut:], label=labels[d], color=colors[d], linestyle=linestyles[d], linewidth=1.2, alpha=0.8)
    ax3.set_ylabel('Roll Angle [deg]', fontsize=12)
    ax3.set_xlabel('Time [s]', fontsize=12)
    ax3.set_title('Attitude Response (Control Jitter)', fontsize=13, fontweight='bold')
    ax3.grid(True, linestyle=':', alpha=0.6)
    
    # --- Q4: MEDIA 4 MOTORI (RPM) ---
    for d in range(NUM_DRONES):
        ax4.plot(t_plot, log_mean_rpm[d][cut:], label=labels[d], color=colors[d], linestyle=linestyles[d], linewidth=1.2, alpha=0.8)
    ax4.set_ylabel('Average RPM (All 4 Motors)', fontsize=12)
    ax4.set_xlabel('Time [s]', fontsize=12)
    ax4.set_title('Average Actuator Effort (All 4 Motors)', fontsize=13, fontweight='bold')   
    ax4.grid(True, linestyle=':', alpha=0.6)
    
    fig.suptitle(f'Swarm Aerodynamic Analysis: {NUM_DRONES}-Agent Extraction Panel', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.show()

if __name__ == '__main__':
    run_evaluation()