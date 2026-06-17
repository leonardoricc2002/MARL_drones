import numpy as np
import math

from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType

class OttoAviary(BaseRLAviary):
    """Single agent RL problem: Inseguire una traiettoria a 8 (Lemniscata)."""
    
    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 30,
                 gui=False,
                 record=False,
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM
                 ):
        # Ho aumentato la durata dell'episodio a 15 secondi per dargli il tempo di completare la figura a 8
        self.EPISODE_LEN_SEC = 15 
        super().__init__(drone_model=drone_model,
                         num_drones=1,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act
                         )
        self.TARGET_POS = np.array([1.0, 1.0, 1.0])
    ################################################################################

    def _computeObs(self):
        obs = super()._computeObs()
        state = self._getDroneStateVector(0)
        
        # Calcoliamo l'errore rispetto al target (1,1,1)
        errore = self.TARGET_POS - state[0:3]
        
        # Normalizziamo dividendo per 2.0: 
        # se il drone è a 2 metri, l'IA legge '1.0' (il suo limite).
        # Se è a 0.5 metri, legge '0.25'. Molto più preciso!
        obs[0][0:3] = errore / 2.0 
        
        return obs
    ################################################################################
    
    def _computeReward(self):
        state = self._getDroneStateVector(0)
        
        # Penale per evitare che stia a terra
        if state[2] < 0.15:
            return -1.0
            
        distanza = np.linalg.norm(self.TARGET_POS - state[0:3])
        
        # DEADZONE: Se il drone è entro 10 cm dal target, 
        # diamo il punteggio massimo (2.0) e smettiamo di tormentarlo.
        if distanza < 0.1:
            return 2.0
        
        # Altrimenti premiamo quanto è vicino al target
        return max(0.0, 2.0 - distanza)

    ################################################################################
    
    def _computeTerminated(self):
        """Computes the current terminated value."""
        # L'episodio non termina mai per "vittoria anticipata". 
        # Il drone deve continuare a seguire l'Otto finché non scade il tempo (Truncated).
        return False
    
    def _computeTruncated(self):
        """Computes the current truncated value."""
        state = self._getDroneStateVector(0)
        
        # GABBIA NORMALE (Rimosso il controllo dei 2 secondi!)
        if (abs(state[0]) > 4.0 or abs(state[1]) > 4.0 or state[2] > 2.0 
             or abs(state[7]) > .6 or abs(state[8]) > .6 
        ):
            return True
            
        if self.step_counter/self.PYB_FREQ > self.EPISODE_LEN_SEC:
            return True
        else:
            return False

    ################################################################################
    
    def _computeInfo(self):
        return {"answer": 42}