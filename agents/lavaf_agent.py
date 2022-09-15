import numpy as np
import pandas as pd
from scipy.stats import norm

from rewards.get_reward import get_reward

def argmax(q_values):

    top = float("-inf")
    ties = []

    for i in range(len(q_values)):
        if q_values[i] > top:
            top = q_values[i]
            ties = []

        if q_values[i] == top:
            ties.append(i)

    return np.random.choice(ties)

class LavafAgent:
    """
    Training 
    """

    def __init__(self):
        self.action_space = {}
        self.action_id = 0
        self.prev_action_id = 0
        self.w = np.zeros((32, 1))
        self.state = np.zeros((31,))
        self.prev_state = np.zeros((31,))
        self.epsilon = 0.01
        self.alpha = 0.05

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

    def compute_action(self, observation, agent_id):
        """Get observation return action"""

        # Save previous state and action
        self.prev_action_id = self.action_id
        self.prev_state = self.state

        # Get state
        stats = pd.read_pickle('observation_stats.pkl')
        correspondance = {0: 'Month', 1: 'Day Type', 2: 'Hour', 3: 'Temperature', 4: 'Temperature + 6h', 5:'Temperature + 12h', 6:'Temperature + 24h',7: 'Humidity rate', 8:'Humidity rate + 6h', 9:'Humidity rate + 12h', 10:'Humidity rate + 24h', 11:'Diffuse sun', 12:'Diffuse sun + 6h', 13:'Diffuse sun + 12h', 14:'Diffuse sun + 24h', 15:'Direct sun', 16:'Direct sun + 6h', 17:'Direct sun + 12h', 18:'Direct sun + 24h', 19:'Carbon density', 20:'Equipment power', 21:'Solar Power', 22:'Battery storage', 23:'Power consumption', 24:'Cost', 25:'Cost + 6h', 26:'Cost + 12h', 27:'Cost + 24h'}
        trigo_enc_cols = [0, 1, 2]
        state_list = []
        for i in correspondance.keys():
            if i in trigo_enc_cols:
                state_list.append((np.cos(observation[i] * 2 * np.pi)) / stats['max'][[correspondance[i]][0]])
                state_list.append((np.sin(observation[i] * 2 * np.pi)) / stats['max'][[correspondance[i]][0]])
            else:
                state_list.append(observation[i]/(stats['max'][[correspondance[i]]][0] - stats['min'][[correspondance[i]]][0]))
        self.state = np.array(state_list)

        # Action choice
        #discrete_action_space = np.linspace(-1, 1, num = 100)
        discrete_action_space = norm.rvs(0, 0.1, 100)
        if np.random.random() < self.epsilon:
            self.action_id = np.random.randint(0, len(discrete_action_space))
        else:
            q_values = [np.dot(self.w.transpose(), np.reshape(np.concatenate((self.state, np.array([action]))), (32,1)))[0][0] for action in discrete_action_space]
            self.action_id = argmax(q_values)
        
        # Q-Values update
        action = discrete_action_space[self.action_id]
        prev_action = discrete_action_space[self.prev_action_id]
        x = np.reshape(np.concatenate((self.prev_state, np.array([prev_action]))), (32,1))
        x_ = np.reshape(np.concatenate((self.state, np.array([action]))), (32,1))
        self.w += self.alpha*(get_reward(observation[23], observation[19]*observation[23], observation[23]*observation[24], agent_id) + np.dot(self.w.transpose(), x_) - np.dot(self.w.transpose(), x)) * x
        

        return np.array([action], dtype=self.action_space[agent_id].dtype)
