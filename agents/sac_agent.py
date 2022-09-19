from stable_baselines3 import SAC
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
        index_commun = [0, 2, 19, 4, 8, 24]
        index_particular = [20, 21, 22, 23]
        normalization_value_commun = [12, 24, 0.29, 32.2, 100, 0.54]
        normalization_value_particular = [8, 4, 1, 7.5]
        observation_commun = [observation[0][i]/n for i,n in zip(index_commun, normalization_value_commun)]
        observation_particular = [[o[i]/n for i,n in zip(index_particular, normalization_value_particular) for o in observation]]
        observation_particular = list(itertools.chain(*observation_particular))
        observation = observation_commun + observation_particular

        model = SAC.load("sac_citylearn1")
        action, _states = model.predict(observation, deterministic=True)
        action = [np.array([a], dtype=self.action_space[0].dtype) for a in action]
        return action
         
