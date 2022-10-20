from agents.random_agent import RandomAgent
from agents.rbc_agent import BasicRBCAgent
from agents.nobattery_agent import NoBatteryAgent
from agents.sac_agent import SACAgent
from agents.marlisa import MARLISA
import json


###################################################################
#####                Specify your agent here                  #####
###################################################################

def GenStateActionFromJson(JsonPath, BuildingCount = 5):

    with open(JsonPath) as json_file:
        buildings_states_actions = json.load(json_file)

    States = buildings_states_actions['observations']
    Actions = buildings_states_actions['actions']

    StateINFo = {}
    ActionINFo = {}
    INFos = {}
    
    for var, ins in States.items():
        #print(var, " <><> ", ins)
        if ins['active']:
            StateINFo[var] = ins['active']
    for act, ins  in Actions.items():
        if ins['active']:
            ActionINFo[act] = ins['active']

    INFos["states"] = StateINFo
    INFos["action"] = ActionINFo

    return {"Building_" + str(key): INFos for key in range(1,BuildingCount+1)}


JsonFile = 'data/citylearn_challenge_2022_phase_1/schema.json'
BuildingsStatesActions = GenStateActionFromJson(JsonFile, BuildingCount = 5)
#building_info = env.get_building_information()

#params_agent = {'building_ids':["Building_"+str(i) for i in [1,2,3,4,5]],
                #  'buildings_states_actions':BuildingsStatesActions, 
                #  'building_info':building_info,
                #  'observation_spaces':observations_spaces, 
                #  'action_spaces':actions_spaces, 
                #  'hidden_dim':[256,256], 
                #  'discount':1/12, 
                #  'tau':5e-3, 
                #  'lr':3e-4, 
                #  'batch_size':256, 
                #  'replay_buffer_capacity':1e5, 
                #  'regression_buffer_capacity':3e4, 
                #  'start_training':600, # Start updating actor-critic networks
                #  'exploration_period':7500, # Just taking random actions
                #  'start_regression':500, # Start training the regression model
                #  'information_sharing':True, # If True -> set the appropriate 'reward_function_ma' in reward_function.py
                #  'pca_compression':.95, 
                #  'action_scaling_coef':0.5, # Actions are multiplied by this factor to prevent too aggressive actions
                #  'reward_scaling':5., # Rewards are normalized and multiplied by this factor
                #  'update_per_step':2, # How many times the actor-critic networks are updated every hourly time-step
                #  'iterations_as':2,# Iterations of the iterative action selection (see MARLISA paper for more info)
                #  'safe_exploration':True} 

UserAgent = SACAgent
