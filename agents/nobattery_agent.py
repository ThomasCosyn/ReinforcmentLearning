import numpy as np
import csv

def nobattery_policy(observation, action_space):
    """
    We never use the battery
    """
    action = 0 # Default value
    action = np.array([action], dtype=action_space.dtype)
    assert action_space.contains(action)
    return action

class NoBatteryAgent:
    """
    Basic Rule based agent adopted from official Citylearn Rule based agent
    https://github.com/intelligent-environments-lab/CityLearn/blob/6ee6396f016977968f88ab1bd163ceb045411fa2/citylearn/agents/rbc.py#L23
    """
    def __init__(self):
        self.action_space = {}

    def set_action_space(self, agent_id, action_space):
        self.action_space[agent_id] = action_space

    def compute_action(self, observation, agent_id):
        """Get observation return action"""
        return nobattery_policy(observation, self.action_space[agent_id])