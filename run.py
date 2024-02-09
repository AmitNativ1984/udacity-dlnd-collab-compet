import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from unityagents import UnityEnvironment
from maddpg_agent import Agent
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a MADDPG agent to solve the Tennis environment.')
    parser.add_argument('--filename', type=str, default='Tennis_Linux_NoVis/Tennis.x86_64', help='filename for Unity environment')
    args = parser.parse_args()
    
    # ----- Define the environment -----
    env = UnityEnvironment(file_name=args.filename)
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # ----- Examine the State and Action Spaces -----
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    # ----- Define the Agent -----
    agent = Agent(state_size=state_size, action_size=action_size, num_agents=num_agents, random_seed=0)

    # load the weights from file
    agent.actor_local.load_state_dict(torch.load('checkpoint_actor.pth'))
    agent.critic_local.load_state_dict(torch.load('checkpoint_critic.pth'))

    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    score = 0
    traj_length = 0
    print("Playing...")
    while True:
        actions=agent.act(states)            # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break
        traj_length += 1
        print('\rTrajectory Length {:d}\t Agent1 Score {:.2f}\tAgent2 Score {:.2f}'.format(traj_length, scores[0], scores[1]), end="")
    
    print('\n')
    print("Game Over")
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

    env.close()