from stable_baselines3 import SAC, td3, PPO, DDPG, A2C
import numpy as np
import itertools
from sb3_contrib import TQC, TRPO

class SACAgent:
    
    def __init__(self):
        self.action_space = {}
    
    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

    def compute_action(self, observation, agent_id = None):
        """Get observation return action"""

        # Reshape observation

        # All features
        index_commun = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 24, 25, 26, 27]
        index_particular = [20, 21, 22, 23]
        normalization_value_commun = [12, 24, 32.2, 32.2, 32.2, 32.2, 100, 100, 100, 100, 1017, 1017, 1017, 1017, 953, 953, 953, 953, 0.29, 0.54, 0.54, 0.54, 0.54]
        normalization_value_particular = [8, 4, 1, 7.5]

        periodic_encoding = True

        if agent_id != None:

            if not periodic_encoding:
                observation_commun = [observation[i]/n for i, n in zip(index_commun, normalization_value_commun)]

            else:
            # periodic normalization for hours and days
                observation_commun = []
                for i, n in zip(index_commun, normalization_value_commun):
                    # if hours or months, periodic normalization
                    if i in [0, 2]:
                        x = (observation[i] * 2*np.pi) / n 
                        cosx = np.cos(x)
                        sinx = np.sin(x)
                        observation_commun.append((1 + cosx)/2)
                        observation_commun.append((1 + sinx)/2)
                    # if not classical normalization
                    else:
                        observation_commun.append(observation[0][i]/n)
            
            observation_particular = [[o[i]/n for i, n in zip(index_particular, normalization_value_particular)] for o in observation]
            observation_particular = list(itertools.chain(*observation_particular))
            obs = observation_commun + observation_particular

            model = PPO.load("PPO2_{}".format(len(observation)))
            action, _states = model.predict(obs, deterministic=True)
            return action

        else:
            if not periodic_encoding:
                observation_commun = [observation[0][i]/n for i,n in zip(index_commun, normalization_value_commun)]

            else:
                # periodic normalization for hours and days
                observation_commun = []
                for i, n in zip(index_commun, normalization_value_commun):
                    # if hours or months, periodic normalization
                    if i in [0, 2]:
                        x = (observation[0][i] * 2*np.pi) / n 
                        cosx = np.cos(x)
                        sinx = np.sin(x)
                        observation_commun.append((1 + cosx)/2)
                        observation_commun.append((1 + sinx)/2)
                    # if not classical normalization
                    else:
                        observation_commun.append(observation[0][i]/n)

            observation_particular = [[o[i]/n for i, n in zip(index_particular, normalization_value_particular)] for o in observation]
            observation_particular = list(itertools.chain(*observation_particular))
            obs = observation_commun + observation_particular
            
            model = PPO.load("PPO2_{}".format(len(observation)))

            action, _states = model.predict(obs, deterministic=True)
            action = [np.array([a], dtype=self.action_space[0].dtype) for a in action]
            return action
        