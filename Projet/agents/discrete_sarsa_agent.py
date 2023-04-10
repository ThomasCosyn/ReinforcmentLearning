import numpy as np

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

        dist_to_middle = [abs(ties[i] - len(q_values)/2) for i in range(len(ties))]

        return ties[np.argmin(dist_to_middle)]
        # return np.random.choice(ties)


class TrainingDiscreteSarsaAgent:
    """
    The first training trial
    """

    def __init__(self):
        self.action_space = {}
        self.discrete_action_space = np.arange(start = -0.1, stop = 0.11, step = 0.01)
        self.action_id = len(self.discrete_action_space)//2
        self.months = [i for i in range(1, 13)]
        self.hours = [i for i in range(1, 25)]
        #self.carbon_densities = [0, 0.13, 0.15, 0.18]
        self.carbon_densities = [0, 0.126, 0.145, 0.164, 0.186]
        self.costs = [0, 0.3]
        self.battery_storages = [0, 0.25, 0.5, 0.75]
        self.q_values = np.zeros((len(self.months), len(self.hours), len(self.carbon_densities), len(self.costs), len(self.battery_storages), len(self.costs), len(self.costs), len(self.costs), len(self.discrete_action_space)))
        self.alpha = 0.1
        self.epsilon = 0.005
        self.state = (0, 0, 0, 0, 0, 0, 0, 0)
        self.prev_state = (0, 0, 0, 0, 0, 0, 0, 0)
        self.prev_action_id = len(self.discrete_action_space)//2
        self.algorithm = "SARSA"
        self.choices = [0] * len(self.discrete_action_space)
        self.discount = 1

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

    def compute_action(self, observation, agent_id):
        """Get observation return action"""

        # Save previous state and action
        self.prev_action_id = self.action_id
        self.prev_state = self.state

        # Get state
        month = observation[0]
        hour = observation[2]
        carbon_density = [observation[19] < self.carbon_densities[i] for i in range(len(self.carbon_densities))].count(False) - 1
        battery_storage = [observation[22] < self.battery_storages[i] for i in range(len(self.battery_storages))].count(False) - 1
        cost = [observation[24] < self.costs[i] for i in range(len(self.costs))].count(False) - 1
        cost6 = [observation[25] < self.costs[i] for i in range(len(self.costs))].count(False) - 1
        cost12 = [observation[26] < self.costs[i] for i in range(len(self.costs))].count(False) - 1
        cost24 = [observation[27] < self.costs[i] for i in range(len(self.costs))].count(False) - 1
        self.state = (month - 1, hour - 1, carbon_density, cost, battery_storage, cost6, cost12, cost24)

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
            self.q_values[self.state][self.action_id] += self.alpha*(self.discount * get_reward(observation[23], observation[19]*observation[23], observation[23]*observation[24], agent_id) + self.q_values[self.state][self.action_id] - self.q_values[self.prev_state][self.prev_action_id])
        elif self.algorithm == "Q-learning":
            self.q_values[self.state][self.action_id] += self.alpha*(self.discount * get_reward(observation[23], observation[19]*observation[23], observation[23]*observation[24], agent_id) + max(self.q_values[self.state][:]) - self.q_values[self.prev_state][self.prev_action_id])

        # Saving RL model
        if observation[0] == 7 and observation[1] == 1 and observation[2] == 23:
            np.save('qvalues.npy', self.q_values)
            # print(self.choices)
            # input()

        return np.array([action], dtype=self.action_space[agent_id].dtype)

class TrainedSarsaAgent:
    def __init__(self):
        self.action_space = {}
        self.discrete_action_space = np.arange(start = -0.5, stop = 0.6, step = 0.05)
        self.state = (0, 0, 0, 0, 0)
        self.action_id = 10
        self.months = [i for i in range(1, 13)]
        self.hours = [i for i in range(1, 25)]
        self.carbon_densities = [0, 0.13, 0.15, 0.18]
        self.costs = [0, 0.3]
        self.battery_storages = [0, 0.25, 0.5, 0.75]
        self.q_values = np.load("qvalues.npy")
        self.alpha = 0.1
        self.epsilon = 0.01
        self.prev_state = (0, 0, 0, 0, 0)
        self.prev_action_id = 10

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

    def compute_action(self, observation, agent_id):
        """Get observation return action"""

        # Get state
        month = observation[0]
        hour = observation[2]
        carbon_density = [observation[19] < self.carbon_densities[i] for i in range(len(self.carbon_densities))].count(False) - 1
        battery_storage = [observation[22] < self.battery_storages[i] for i in range(len(self.battery_storages))].count(False) - 1
        cost = [observation[24] < self.costs[i] for i in range(len(self.costs))].count(False) - 1
        self.state = (month - 1, hour - 1, carbon_density, cost, battery_storage)

        # Action choice
        self.action_id = argmax(self.q_values[self.state][:])
        action = self.discrete_action_space[self.action_id]

        return np.array([action], dtype=self.action_space[agent_id].dtype)