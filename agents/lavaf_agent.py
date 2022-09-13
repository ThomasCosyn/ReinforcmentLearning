import numpy as np

from rewards.get_reward import get_reward

class LavafAgent:
    """
    Training 
    """

    def __init__(self):
        self.action_space = {}
        self.action = 0
        self.prev_action = 0
        self.w = np.zeros((32, 1))
        self.state = np.zeros((31, 1))
        self.prev_state = np.zeros((31, 1))

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

    def compute_action(self, observation, agent_id):
        """Get observation return action"""

        # Save previous state and action
        self.prev_action_id = self.action_id
        self.prev_state = self.state

        # Get state
        self.state = np.array()

        # Action choice
        if np.random.random() < self.epsilon:
            self.action_id = np.random.randint(0, len(self.discrete_action_space))
            action = self.discrete_action_space[self.action_id]
            # print("State : {0}".format(self.state))
            # print("Action values : {}".format(self.q_values[self.state][:]))
            # input()
        else:
            self.action_id = argmax(self.q_values[self.state][:])
            action = self.discrete_action_space[self.action_id]
        
        self.choices[self.action_id] += 1

        # Q-Values update
        if self.algorithm == "SARSA":
            self.q_values[self.state][self.action_id] += self.alpha*(get_reward(observation[23], observation[19]*observation[23], observation[23]*observation[24], agent_id) + self.q_values[self.state][self.action_id] - self.q_values[self.prev_state][self.prev_action_id])
        elif self.algorithm == "Q-learning":
            self.q_values[self.state][self.action_id] += self.alpha*(get_reward(observation[23], observation[19]*observation[23], observation[23]*observation[24], agent_id) + max(self.q_values[self.state][:]) - self.q_values[self.prev_state][self.prev_action_id])

        # Saving RL model
        if observation[0] == 7 and observation[1] == 1 and observation[2] == 23:
            np.save('qvalues.npy', self.q_values)
            print(self.choices)
            input()

        return np.array([action], dtype=self.action_space[agent_id].dtype)
