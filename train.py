import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from unityagents import UnityEnvironment
from maddpg_agent import Agent

import argparse

def maddpg(env, agent, n_agents=2, n_episodes=1000, max_t=1000, que_len=100):
    """Deep Deterministic Policy Gradient.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        print_every (int): print scores every print_every episodes
    """
    
    scores_deque = deque(maxlen=que_len)
    scores = []
    writer = SummaryWriter()  # Create a SummaryWriter instance for logging
    
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    max_agent_scores = []
    mean_deque_scores = []
    for i_episode in range(1, n_episodes+1):
        env_info=env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        agent_scores = np.zeros(n_agents)
        traj_length = 0 
        while True:
            actions=agent.act(states)
            env_info=env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states
            agent_scores += rewards
            if any(dones):
                break
            traj_length += 1
        
        max_score = np.max(agent_scores)
        scores_deque.append(max_score)
        mean_score = np.mean(scores_deque)

        max_agent_scores.append(max_score)
        mean_deque_scores.append(mean_score)

        
        writer.add_scalar('Max Score', max_score, i_episode)  # Log the score to TensorBoard
        writer.add_scalar('Mean Score', mean_score, i_episode)  # Log the score to TensorBoard

        print('\rEpisode {}\tTrajectory Length {:d}\tMax Score: {:.2f}\t Mean Score {:.2f}'.format(i_episode, traj_length, scores_deque[-1], np.mean(mean_score)), end="")
       
        if np.mean(scores_deque) >= 0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-que_len, np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
            break     
            
    writer.close()  # Close the SummaryWriter
    
    return max_agent_scores, mean_deque_scores


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

    # # ----- Train the Agent with DDPG -----
    scores, mean_scores = maddpg(env, agent, n_agents=2, n_episodes=20000, max_t=1000, que_len=100)

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(1, len(scores)+1), scores, label='Max Score')
    ax.plot(np.arange(1, len(mean_scores)+1), mean_scores, label='Mean Score')
    ax.set_ylabel('Score')
    ax.set_xlabel('Episode #')
    ax.legend(loc='upper left')
    plt.savefig('p3_collab-compet/scores.png')

    env.close()