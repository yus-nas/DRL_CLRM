import gym
from gym import spaces
import numpy as np
from gym.utils import seeding

from reservoir_sim import Simulator
import gc
import psutil
import copy

class ReservoirEnv(gym.Env):
    def __init__(self, env_config):
               
        self.sim_input = copy.deepcopy(env_config["sim_input"])
          
        # simulator
        self.res_sim = Simulator(self.sim_input)
        
        # initialize history collector
        self.hist = self.History()
        
        # action and observation space 
        self._setup_spaces()
        
        # cluster in cluster_train_realz
        if env_config.worker_index == 0:
            self.cluster_index = self.sim_input["cluster_train_realz"][0,0]
            self.realz_train_ind = self.sim_input["cluster_train_realz"][0,1:] - 1 # 0-based indexing
        else:
            self.cluster_index = self.sim_input["cluster_train_realz"][env_config.worker_index - 1,0]
            self.realz_train_ind = self.sim_input["cluster_train_realz"][env_config.worker_index - 1,1:] - 1
        
        # selected realization
        self.realz_train = np.argwhere(self.sim_input["cluster_labels"] == self.cluster_index)[self.realz_train_ind] + 1
        
        # track sim iterations
        self.sim_iter = 0
          
    def _setup_spaces(self):
        
        #self.action_space = spaces.Box(0.0, +1.0, shape=[4], dtype=np.float32)
        self.action_space = spaces.Box(0.0, +1.0, shape=[len(self.res_sim.reservoir.wells)], dtype=np.float32)
        self.num_obs_data = (3 * self.res_sim.num_prod + 2 * self.res_sim.num_inj) * self.sim_input["num_run_per_step"]
        self.observation_space = spaces.Box(-10, 10, shape=(self.num_obs_data,))
    
    def reset(self):   
        
        # helps with memory issues
        self.auto_garbage_collect()
        
        # select realization   
        self.sim_input["realz"] = int(self.realz_train[self.sim_iter])
        self.sim_iter += 1
        if self.sim_input["realz"] > self.sim_input["num_realz"]: # exclude true models
            self.sim_input["realz"] = np.random.choice(self.sim_input["num_realz"]) + 1
        
        # approach 1
        #self.sim_input["realz"] = np.random.choice(self.sim_input["num_realz"]) + 1
        #self.sim_input["realz"] = np.random.choice(self.sim_input["num_realz"], p=self.sim_input["sampling_prob"]) + 1
        
        # reset
        self.res_sim.reset_vars(self.sim_input)
        self.hist.reset()
        self.cum_reward = 0
        
        # run historical period (assumes length of hist = length of ctrl step) #TODO: make general
        observation, _, _, _ = self.step(self.sim_input["hist_ctrl_scaled"])

        return observation
    
    def step(self, action):
        assert self.action_space.contains(action)
        
        # add action to history
        self.hist.actions.append(action)

        # change well controls
        controls = action * (self.sim_input["upper_bound"] - self.sim_input["lower_bound"]) + self.sim_input["lower_bound"]
        self.res_sim.set_well_control(controls)

        # run simulation
        done = self.res_sim.run_single_ctrl_step()
        
        # calculate reward
        reward = self.res_sim.calculate_npv()
        self.cum_reward += reward
        
        # get observation
        observation = self.get_observation()    
             
        return observation, reward, done, {} 
    
    class History():
        def __init__(self):
            self.reset()

        def reset(self):
            self.scaled_state = []
            self.unscaled_state = []
            self.reward_dollar = []
            self.actions = []
            self.done = []
            
    def experiment(self, actions):
        # advance from "current state" according to actions
        self.hist.reset()
        
        unscaled_obs = self.get_unscaled_state()
        self.hist.unscaled_state.append(unscaled_obs)
        obs = self.get_observation()
        self.hist.scaled_state.append(obs)

        for i_action in actions:
            obs, reward, done, _ = self.step(i_action)
            self.hist.scaled_state.append(obs)
            self.hist.reward_dollar.append(reward)
            self.hist.done.append(done)
            
            unscaled_obs = self.get_unscaled_state()
            self.hist.unscaled_state.append(unscaled_obs)
    
    def get_unscaled_state(self):
        _ , unscaled_state = self.res_sim.get_observation()
        
        return unscaled_state
                        
    def get_observation(self): 
        obs, _ = self.res_sim.get_observation()
        
        return obs
    
    
    def auto_garbage_collect(self, pct=65.0):
        """
        auto_garbage_collection - Call the garbage collection if memory used is greater than 65% of total available memory.
                                  This is called to deal with an issue in Ray not freeing up used memory.

            pct - Default value of 65%.  Amount of memory in use that triggers the garbage collection call.
        """
        if psutil.virtual_memory().percent >= pct:
            gc.collect()     
