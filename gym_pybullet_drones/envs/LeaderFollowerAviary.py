import numpy as np

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType


class LeaderFollowerAviary(BaseRLAviary):
    """
    Ambiente Custom per l'esame: Leader-Follower.
    Eredita tutto da MultiHoverAviary per una compatibilità totale con learn.py.
    """
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=3, # Assicurati che sia 3
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None, # Lascia None qui
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
                 ):
        
        # 1. DEFINIAMO LE POSIZIONI PRIMA DI INIZIALIZZARE LA FISICA
        # Definizione corretta delle posizioni
        initial_xyzs = np.array([
            [0.0, 0.0, 1.2],
            [0.0, 0.0, 0.9],
            [0.0, 0.0, 0.6]
        ]) 
        
        # 2. ORA PASSIAMO LE POSIZIONI AL COSTRUTTORE GENITORE
        self.EPISODE_LEN_SEC = 12
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs, # <--- Ora le legge subito!
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record, 
                         obs=obs,
                         act=act
                         )
        
        # 3. Target pos serve solo per la logica della reward, va bene qui
        self.TARGET_POS = self.INIT_XYZS + np.array([[0,0,1/(i+1)] for i in range(num_drones)])
    
    def _computeReward(self):
        # Estraiamo gli stati completi dei 3 droni
        state_leader = self._getDroneStateVector(0)
        state_follower_1 = self._getDroneStateVector(1)
        state_follower_2 = self._getDroneStateVector(2)
        
        # --- LA FUNZIONE DI PRECISIONE (Versione Stabile Senza Yaw) ---
        def get_drone_reward(target, state):
            # Estraiamo solo posizione e velocità lineare
            pos = state[0:3]           
            vel = state[10:13]         
            
            # 1. Penalità di Posizione (Quadratica): Lo inchioda sul target
            dist_sq = np.sum((target - pos)**2) 
            
            # 2. Penalità di Velocità Lineare: Impedisce le oscillazioni a pendolo
            speed_penalty = np.sum(vel**2)
            
            # Calcolo finale: Punteggio massimo 10, a cui sottraiamo solo distanza e velocità
            return 10.0 - (dist_sq * 20.0) - (speed_penalty * 2.0)

        # --- CALCOLO TARGET E REWARD PER OGNI DRONE ---
        # Il Leader punta al centro esatto a 1.2m
        target_leader = np.array([0.0, 0.0, 1.2])
        reward_leader = get_drone_reward(target_leader, state_leader)
        
        # I Follower puntano alla posizione *reale* del leader, ma più in basso
        pos_leader_reale = state_leader[0:3]
        target_f1 = pos_leader_reale + np.array([0.0, 0.0, -0.3])
        reward_f1 = get_drone_reward(target_f1, state_follower_1)
        
        target_f2 = pos_leader_reale + np.array([0.0, 0.0, -0.6])
        reward_f2 = get_drone_reward(target_f2, state_follower_2)
        
        premio_totale = reward_leader + reward_f1 + reward_f2
        
        # --- DESTINO CONDIVISO (CRASH PENALTY) ---
        if pos_leader_reale[2] < 0.15 or state_follower_1[2] < 0.15 or state_follower_2[2] < 0.15:
            premio_totale -= 50.0 
            
        return float(premio_totale)
    
    def _computeTerminated(self):
        """
        Non terminiamo l'episodio se raggiungono l'obiettivo.
        Vogliamo che restino in posizione per tutto il tempo (EPISODE_LEN_SEC)
        per massimizzare i punti.
        """
        return False

    ################################################################################
    
    def _computeTruncated(self):
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        for i in range(self.NUM_DRONES):
            if (abs(states[i][0]) > 3.0 or abs(states[i][1]) > 3.0 or states[i][2] > 2.5
             # Relaxed tilt limits: 1.2 radians is ~68 degrees
             or abs(states[i][7]) > 1.2 or abs(states[i][8]) > 1.2 
            ):
                return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False
    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42} #### Calculated by the Deep Thought supercomputer in 7.5M years