import numpy as np
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class LeaderFollowerAviary(BaseRLAviary):
    """
    Ambiente Custom: Leader-Follower (Drone 0 è il Leader).
    - Leader: Override attivo (Statua a 1.2m)
    - Follower: Target fissi (0.9m e 0.6m)
    """
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=3,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB_GND_DRAG_DW, 
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.VEL
                 ):
        
        # Posizioni di partenza fisse
        initial_xyzs = np.array([
            [0.0, 0.0, 1.2], # Leader
            [0.0, 0.0, 0.9], # Follower 1
            [0.0, 0.0, 0.6]  # Follower 2
        ]) 
        
        self.EPISODE_LEN_SEC = 12
        super().__init__(drone_model=drone_model, 
                         num_drones=num_drones, 
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs, 
                         initial_rpys=initial_rpys, 
                         physics=physics,
                         pyb_freq=pyb_freq, 
                         ctrl_freq=ctrl_freq, 
                         gui=gui, 
                         record=record, 
                         obs=obs, 
                         act=act)
        
        # Target fissi (non cambiano mai)
        self.TARGET_POS = initial_xyzs.copy()

    def step(self, action):
        """
        OVERRIDE: Forza il Leader a stare fermo 
        """
        action_corretta = action.copy()
        action_corretta[0, :] = 0.0 
        return super().step(action_corretta)
        
    def _computeObs(self):
        # I target restano quelli definiti nell'init
        return super()._computeObs()
    
    def _computeReward(self):
        # Stato attuale
        state_leader = self._getDroneStateVector(0)
        state_follower_1 = self._getDroneStateVector(1)
        state_follower_2 = self._getDroneStateVector(2)
        
        def get_drone_reward(target, state, weight=1.0):
            pos = state[0:3]           
            vel = state[10:13]         
            
            error_xy = np.sum((target[0:2] - pos[0:2])**2) # Lasciamo XY quadrato (va bene così)
            
            # ---   ERRORE LINEARE (L1) SU Z ---
            dist_z = abs(target[2] - pos[2]) 
            
            speed_penalty = np.sum(vel**2)
            dist = np.linalg.norm(target - pos)
            
            # Tolleranza di 5cm: se è dentro, prende bonus pieno
            if dist < 0.05:
                reward = 15.0 - (speed_penalty * 0.5) 
            else:
                # La penalità Z ora è Lineare (dist_z * 60)
                # Se dist_z è 0.2m, la penalità è 12.0 punti! Non può più ignorarla.
                reward = 15.0 - (error_xy * 2.0) - (dist_z * 60.0) - (speed_penalty * 2.0)
            
            return reward * weight

        # Calcolo reward con PESI (Weights)
        reward_leader = get_drone_reward(self.TARGET_POS[0], state_leader, weight=1.0)
        reward_f1 = get_drone_reward(self.TARGET_POS[1], state_follower_1, weight=1.5)
        reward_f2 = get_drone_reward(self.TARGET_POS[2], state_follower_2, weight=1.5)
        
        premio_totale = reward_leader + reward_f1 + reward_f2
        
        # Penalità pesanti per uscire dai limiti
        for i in range(3):
            state = self._getDroneStateVector(i)
            if state[2] < 0.15: # Contatto terra
                premio_totale -= 50.0
                if abs(state[0]) > 2.0 or abs(state[1]) > 2.0 or state[2] > 1.6:
                    premio_totale -= 100.0  

        return float(premio_totale / 100.0)
    
    def _computeTerminated(self):
        return False

    def _computeTruncated(self):
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        for i in range(self.NUM_DRONES):
            # Limiti di sicurezza
            if (abs(states[i][0]) > 3.0 or abs(states[i][1]) > 3.0 or states[i][2] > 2.5
             or abs(states[i][7]) > 1.2 or abs(states[i][8]) > 1.2 
            ):
                return True
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        return False
    
    def _computeInfo(self):
        return {}
