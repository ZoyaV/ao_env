import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
import yaml
import time
import pip
import sys
import soapy


def loopFrame(sim, new_commands):

    t = time.time()
    sim.scrns = sim.atmos.moveScrns()
    sim.Tatmos = time.time()-t
    sim.dmCommands = new_commands
    sim.closed_correction = sim.runDM(
            sim.dmCommands, closed=True)
    sim.slopes = sim.runWfs(dmShape=sim.closed_correction,
                              loopIter=sim.iters)

    sim.open_correction = sim.runDM(sim.dmCommands,
                                      closed=False)

    sim.combinedCorrection = sim.open_correction + sim.closed_correction
    sim.runSciCams(sim.combinedCorrection)
    sim.storeData(sim.iters)
    sim.printOutput(sim.iters, strehl=True)
    sim.addToGuiQueue()
    sim.iters += 1

class AdaptiveOptics(gym.Env):
    def __init__(self, conf_file=None):
        if conf_file:
            self.conf_file = conf_file
        else:
            self.conf_file = sys.path[-1]+"/sh_8x8.yaml"
        with open(self.conf_file, 'r') as stream:
            self.data_loaded = yaml.safe_load(stream)
        self.__counter = 0
        self.mem_img = []
        self.expert_commands = []
        self.action_space = spaces.Box(-2, 2, shape=(32,))
        self.observation_space = spaces.Box(0, 255, shape=(128, 128,3), dtype=np.uint8)
        self.pre_expert_value = None
        self._initao()

    def _initao(self):
        #self.data_loaded['Atmosphere']['windDirs'] = np.random.randint(0, 180, 4).tolist()
        self.sim = soapy.Sim(self.conf_file)
        self.sim.aoinit()
        self.sim.makeIMat()

        self.mem_img = []
        for i in range(3):
            expert_value = self.expert()
            loopFrame(self.sim, expert_value)
            img = self.sim.sciImgs[0].copy()
            img = ((img - np.min(img)) / (np.max(img) - np.min(img))) * 255
            img = img.astype(np.uint8)
            img = img.reshape(1, 128, 128)
            self.mem_img.append(img)
        self.pre_expert_value = self.expert()

        return np.vstack(self.mem_img).T


    def expert(self):
        if self.sim.config.sim.nDM:
            self.sim.dmCommands[:] = self.sim.recon.reconstruct(self.sim.slopes)
        commands = self.sim.buffer.delay(self.sim.dmCommands, self.sim.config.sim.loopDelay)
        return commands

    def step(self, action):
        loopFrame(self.sim, self.pre_expert_value)
        expert_value = self.expert()
        img = self.sim.sciImgs[0].copy()
        next_state = ((img - np.min(img)) / (np.max(img) - np.min(img)) )* 255
        next_state = next_state.astype(np.uint8)
        x = next_state.reshape(1, 128, 128)
        self.mem_img.append(x)

        state = self.mem_img[:3]
        self.mem_img = self.mem_img[1:]

        if  isinstance(self.pre_expert_value, type(None)):
            self.pre_expert_value = expert_value
        reward = (action - self.pre_expert_value)**2
        reward = - reward

        self.__counter += 1
        self.pre_expert_value = expert_value
        return np.vstack(state).T, reward.astype(np.float32), False, {}



    def reset(self):
        state = self._initao()
        return state

    def render(self):
        plt.imshow(self.sim.sciImgs[0])
        plt.show()

    def close(self):
        pass

