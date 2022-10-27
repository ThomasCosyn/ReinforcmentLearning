from citylearn.citylearn import CityLearnEnv
from stable_baselines3 import SAC, td3, ppo, A2C, DDPG
from sb3_contrib import TQC, TRPO
import gym
import numpy as np
import itertools
import supersuit as ss

class Constants:
    episodes = 3
    schema_path = './data/citylearn_challenge_2022_phase_1/schema.json'

def action_space_to_dict(aspace):
    """ Only for box space """
    return { "high": aspace.high,
             "low": aspace.low,
             "shape": aspace.shape,
             "dtype": str(aspace.dtype)
    }

def env_reset(env):
    observations = env.reset()
    action_space = env.action_space
    observation_space = env.observation_space
    building_info = env.get_building_information()
    building_info = list(building_info.values())
    action_space_dicts = [action_space_to_dict(asp) for asp in action_space]
    observation_space_dicts = [action_space_to_dict(osp) for osp in observation_space]
    obs_dict = {"action_space": action_space_dicts,
                "observation_space": observation_space_dicts,
                "building_info": building_info,
                "observation": observations }
    return obs_dict

# Notebook features
# index_commun = [0, 2, 19, 4, 8, 24]
# index_particular = [20, 21, 22, 23]
# normalization_value_commun = [12, 24, 0.29, 32.2, 100, 0.54]
# normalization = [12, 7, 24, 32.2, 32.2, 32.2, 100, 100, 100, 100, 1017, 1017, 1017, 1017, 953, 953, 953, 953, 0.29, 8, 4, 1, 7.5, 0.54, 0.54, 0.54, 0.54]

# All features
index_commun = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 24, 25, 26, 27]
index_particular = [20, 21, 22, 23]
normalization_value_commun = [12, 24, 32.2, 32.2, 32.2, 32.2, 100, 100, 100, 100, 1017, 1017, 1017, 1017, 953, 953, 953, 953, 0.29, 0.54, 0.54, 0.54, 0.54]
normalization_value_particular = [8, 4, 1, 7.5]

# Linear regression feature seleciton
# index_commun = [0, 2, 19, 24, 25, 26, 27]
# index_particular = [20, 21, 22, 23]
# normalization_value_commun = [12, 24, 0.29, 0.54, 0.54, 0.54, 0.54]
# normalization_value_particular = [8, 4, 1, 7.5]

nb_buildings = int(input("Entrer le nombre de buildings : "))

lentot = len(index_commun) + 2 + len(index_particular) * nb_buildings

class EnvCityGym(gym.Env):
    """
    Env wrapper coming from the gym library.
    """
    def __init__(self, env):
        self.env = env

        # get the number of buildings
        self.num_buildings = len(env.action_space)

        # define action and observation space
        self.action_space = gym.spaces.Box(low=np.array([-1] * self.num_buildings), high=np.array([1] * self.num_buildings), dtype=np.float32)

        # define the observation space
        self.observation_space = gym.spaces.Box(low=np.array([0] * lentot), high=np.array([1] * lentot), dtype=np.float32)

        # TO THINK : normalize the observation space

    def reset(self):
        obs_dict = env_reset(self.env)
        obs = self.env.reset()

        # observation = []
        # for i in range(self.num_buildings):
        #     observation.append(self.get_observation(obs, i))

        observation = self.get_observation(obs)

        return observation

    def get_observation(self, obs):
        """
        We retrieve new observation from the building observation to get a proper array of observation
        Basicly the observation array will be something like obs[0][index_commun] + obs[i][index_particular] for i in range(5)

        The first element of the new observation will be "commun observation" among all building like month / hour / carbon intensity / outdoor_dry_bulb_temperature_predicted_6h ...
        The next element of the new observation will be the concatenation of certain observation specific to buildings non_shiftable_load / solar_generation / ...  
        """
        
        # we get the observation commun for each building (index_commun)
        # observation_commun = [obs[0][i]/n for i, n in zip(index_commun, normalization_value_commun)]
        # observation_particular = [[o[i]/n for i, n in zip(index_particular, normalization_value_particular)] for o in obs]
        # observation_particular = list(itertools.chain(*observation_particular))

        # periodic normalization for hours and days
        observation_commun = []
        for i, n in zip(index_commun, normalization_value_commun):
            # if hours or months, periodic normalization
            if i in [0, 2]:
                x = (obs[0][i] * 2*np.pi) / n 
                cosx = np.cos(x)
                sinx = np.sin(x)
                observation_commun.append((1 + cosx)/2)
                observation_commun.append((1 + sinx)/2)
            # if not classical normalization
            else:
                observation_commun.append(obs[0][i]/n)
        observation_particular = [[o[i]/n for i, n in zip(index_particular, normalization_value_particular)] for o in obs]
        observation_particular = list(itertools.chain(*observation_particular))

        # 1 building
        # observation_commun = [obs[0][i]/n for i, n in zip(index_commun, normalization_value_commun)]
        # observation_particular = [obs[agent_id][i]/n for i, n in zip(index_particular, normalization_value_particular)]

        # we concatenate the observation
        observation = observation_commun + observation_particular
        #observation = []

        return observation

    def step(self, action):
        """
        we apply the same action for all the buildings
        """
        # reprocessing action
        action = [[act] for act in action]

        # we do a step in the environment
        obs, reward, done, info = self.env.step(action)

        # observation = []
        # for i in range(len(obs)):
        #     observation.self.get_observation(obs, i)

        observation = self.get_observation(obs)

        return observation, sum(reward), done, info
        
    def render(self): #, mode='human'):
        return self.env.render()

class CustomEnvCityLearn(gym.Env):
    """
    Env wrapper coming from the gym library.
    """
    def __init__(self, env):
        self.env = env

        # get the number of buildings
        self.num_buildings = len(env.action_space)

        # define action and observation space
        self.action_space = gym.spaces.Box(low=np.array([-1] * 7), high=np.array([1] * 7), dtype=np.float32)

        # define the observation space
        self.observation_space = gym.spaces.Box(low=0, high=1, dtype=np.float32, shape = (lentot,))

    def reset(self):
        obs_dict = env_reset(self.env)
        obs = self.env.reset()

        # observation = []
        # for i in range(self.num_buildings):
        #     observation.append(self.get_observation(obs, i))

        observation = self.get_observation(obs)

        return observation

    def get_observation(self, obs):

        observation_commun = [obs[0][i]/n for i, n in zip(index_commun, normalization_value_commun)]
        observation_particular = [[o[i]/n for i, n in zip(index_particular, normalization_value_particular)] for o in obs]
        
        # Si on a que 5 buildings, on "simule" deux buildings fictifs en ajoutant du bruit à la moyenne des mesures des 5 autres
        if len(obs) == 5:
            b6 = [np.mean([observation_particular[i][j] for i in range(len(obs))]) for j in range(len(obs[0]))]
            b7 = [np.mean([observation_particular[i][j] for i in range(len(obs))]) for j in range(len(obs[0]))]

        observation_particular = list(itertools.chain(*observation_particular))

        return obs

    def step(self, action):
        """
        we apply the same action for all the buildings
        """
        # reprocessing action
        action = [[act] for act in action]

        # we do a step in the environment
        obs, reward, done, info = self.env.step(action)

        # observation = []
        # for i in range(len(obs)):
        #     observation.self.get_observation(obs, i)

        observation = self.get_observation(obs)

        return observation, sum(reward), done, info
        
    def render(self): #, mode='human'):
        return self.env.render()

# Définition de l'environnement
env = CityLearnEnv(schema='./data/citylearn_challenge_2022_phase_1/schema_{}.json'.format(nb_buildings))
env = EnvCityGym(env)

# Choix de l'algorithme
algo = 'PPO'

if algo == 'SAC':  
    model = SAC("MlpPolicy", env, verbose = 1)
elif algo == 'TD3':
    model = td3.TD3("MlpPolicy", env, verbose = 1)
elif algo == 'PPO':
    model = ppo.PPO("MlpPolicy", env, verbose = 1)
elif algo == 'A2C':
    model = A2C("MlpPolicy", env, verbose = 2)
elif algo == 'DDPG':
    model = DDPG("MlpPolicy", env, verbose = 1)
elif algo == "TQC":
    model = TQC("MlpPolicy", env, verbose = 1)
elif algo == 'TRPO':
    model = TRPO("MlpPolicy", env, verbose = 1)
else:
    raise ("Please set algo variable either to SAC, TD3, PPO, A2C, DDPG, TQC")

# Apprentissage
e = int(input("Entrer le nombre d'épisodes : "))
model.learn(total_timesteps=e*365*24, log_interval=1)

# Saving the model
name = algo + str(e) + "_" + str(nb_buildings)
model.save(name)

print("Training over")

# model = SAC.load("sac_citylearn1")

# obs = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#         obs = env.reset()
