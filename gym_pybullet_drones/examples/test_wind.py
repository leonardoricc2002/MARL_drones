import os
import time
import numpy as np
import pybullet as p
from stable_baselines3 import PPO

# Importiamo il TUO ambiente senza vento
from gym_pybullet_drones.envs.LeaderFollowerAviary import LeaderFollowerAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
from gym_pybullet_drones.utils.Logger import Logger  # <-- IMPORTANTE: Aggiunto il Logger

# 1. CREIAMO UN AMBIENTE CUSTOM SOLO PER IL TEST (Eredita dal tuo)
class WindyLeaderFollowerAviary(LeaderFollowerAviary):
    def step(self, action):
        """Sovrascrive il passo fisico per iniettare il vento."""
        
        # --- PROFILO VENTO 3D (Calibrato per droni da 27g) ---
        # Solo turbolenza pura, niente spinta continua in una direzione
        gust_x = np.random.normal(loc=0.0, scale=0.002) # Nessuna deriva costante, forti raffiche
        gust_y = np.random.normal(loc=0.0, scale=0.002)  
        gust_z = np.random.normal(loc=0.0, scale=0.003)
        
        wind_force = [gust_x, gust_y, gust_z]
                
        # Applichiamo la forza fisica a tutti i droni
        for i in range(self.NUM_DRONES):
            p.applyExternalForce(
                objectUniqueId=self.DRONE_IDS[i],
                linkIndex=-1, # Centro di massa
                forceObj=wind_force,
                posObj=[0, 0, 0],
                flags=p.WORLD_FRAME 
            )
            
        return super().step(action)

    def _computeTruncated(self):
        """Sovrascriviamo l'arbitro per permettere ai droni di inclinarsi controvento"""
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        for i in range(self.NUM_DRONES):
            # Limite di pitch/roll alzato a 1.2 radianti (~68 gradi) per resistere al vento senza crashare
            if (abs(states[i][0]) > 3.0 or abs(states[i][1]) > 3.0 or states[i][2] > 2.5 
             or abs(states[i][7]) > 1.2 or abs(states[i][8]) > 1.2 
            ):
                return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

# Helper per la sincronizzazione del tempo reale
def sync(i, start_time, timestep):
    """Sincronizza i frame della simulazione con il tempo reale."""
    if timestep > .04 or i%(int(1/(24*timestep))) == 0:
        elapsed = time.time() - start_time
        if elapsed < (i*timestep):
            time.sleep(timestep*i - elapsed)

# 2. FUNZIONE DI TEST CON GRAFICI
def test_model_in_wind():
    print("🌪️ Inizializzazione Ambiente con Vento e Telemetria...")
    
    env = WindyLeaderFollowerAviary(
        gui=True, 
        obs=ObservationType.KIN, 
        act=ActionType.VEL
    )
    
    # === MODIFICA DELLA TELECAMERA ===
    client_id = env.CLIENT
    p.resetDebugVisualizerCamera(
        cameraDistance=2.7,               # <--- Distanza aumentata 
        cameraYaw=50,                     # Angolo laterale
        cameraPitch=-35,                  # Inclinazione dall'alto
        cameraTargetPosition=[0, 0, 0.1], # Punto focale al centro esatto della formazione
        physicsClientId=client_id
    )
    # ==================================

    # Inizializziamo il Logger per registrare i grafici
    logger = Logger(logging_freq_hz=int(env.CTRL_FREQ),
                    num_drones=env.NUM_DRONES)
    
    # ⚠️ INSERISCI QUI IL PERCORSO DEL TUO MODELLO MIGLIORE
    model_path = "results/Hover3_vel_nowind_stabile/best_model.zip" 
    
    if not os.path.exists(model_path):
        print(f"❌ ERRORE: Modello non trovato in {model_path}")
        return
        
    print(f"🧠 Caricamento modello: {model_path}")
    model = PPO.load(model_path, env=env)
    
    obs, info = env.reset(seed=42, options={})
    start = time.time()
    
    print("🚀 Decollo (I grafici appariranno alla fine dei 15 secondi)")
    
    for i in range(env.EPISODE_LEN_SEC * env.CTRL_FREQ):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # --- RACCOLTA DATI PER I GRAFICI ---
        obs2 = obs.squeeze()
        act2 = action.squeeze()
        for d in range(env.NUM_DRONES):
            logger.log(drone=d,
                       timestamp=i/env.CTRL_FREQ,
                       state=np.hstack([obs2[d][0:3],
                                        np.zeros(4),
                                        obs2[d][3:15],
                                        act2[d]
                                        ]),
                       control=np.zeros(12)
                       )
        # -----------------------------------

        env.render()
        sync(i, start, env.CTRL_TIMESTEP)
        
        if terminated or truncated:
            print(f"🛑 Episodio interrotto! Terminated: {terminated}, Truncated: {truncated}")
            break
            
    env.close()
    
    print("📊 Generazione grafici in corso...")
    logger.plot()
    print("✅ Test Terminato.")

if __name__ == '__main__':
    test_model_in_wind()