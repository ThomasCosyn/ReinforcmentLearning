from stable_baselines3 import SAC, td3, PPO, DDPG
import numpy as np
import itertools

class SACAgent:
    
    def __init__(self):
        self.action_space = {}
    
    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

    def compute_action(self, observation):
        """Get observation return action"""

        # Reshape observation

        # Notebook features
        # index_commun = [0, 2, 19, 4, 8, 24]
        # index_particular = [20, 21, 22, 23]
        # normalization_value_commun = [12, 24, 0.29, 32.2, 100, 0.54]
        # normalization = [12, 7, 24, 32.2, 32.2, 32.2, 100, 100, 100, 100, 1017, 1017, 1017, 1017, 953, 953, 953, 953, 0.29, 8, 4, 1, 7.5, 0.54, 0.54, 0.54, 0.54]

        # All features
        # index_commun = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 24, 25, 26, 27]
        # index_particular = [20, 21, 22, 23]
        # normalization_value_commun = [12, 24, 32.2, 32.2, 32.2, 32.2, 100, 100, 100, 100, 1017, 1017, 1017, 1017, 953, 953, 953, 953, 0.29, 0.54, 0.54, 0.54, 0.54]
        # normalization_value_particular = [8, 4, 1, 7.5]

        # Linear regression feature seleciton
        index_commun = [0, 2, 19, 24, 25, 26, 27]
        index_particular = [20, 21, 22, 23]
        normalization_value_commun = [12, 24, 0.29, 0.54, 0.54, 0.54, 0.54]
        normalization_value_particular = [8, 4, 1, 7.5]

        observation_commun = [observation[0][i]/n for i,n in zip(index_commun, normalization_value_commun)]
        observation_particular = [[o[i]/n for i,n in zip(index_particular, normalization_value_particular) for o in observation]]
        observation_particular = list(itertools.chain(*observation_particular))
        observation = observation_commun + observation_particular

        model = PPO.load("PPO6")
        action, _states = model.predict(observation, deterministic=True)
        action = [np.array([a], dtype=self.action_space[0].dtype) for a in action]
        return action
         
