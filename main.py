import time

import torch
import Environment
import HelperMethods
from Agent import Agent
from functions import *

window_size = 20
params = window_size + 2
batch_size = 32
debug_messages = False
update_target_interval = 50
actions_count = 3
num_episodes = 30
max_steps_per_episode = 10000
phase = 'trainn'

data_val = HelperMethods.get_data('ADANIPORTS')
if phase == 'train':
    data_train = HelperMethods.get_data('TITAN')
else:
    data_train = HelperMethods.get_data('TCS') #SBIN,ADANIPORTS,TCS,UPL,TITAN,ITC,GAIL,NTPC
device = HelperMethods.initialize_hardware('cuda')
agent = Agent(params,actions_count,device=device)

# set up the logging variables
episode_rewards = []
avg_rewards = []
env = Environment.TradingBotEnvironment(data_train,window_size)
env_val = Environment.TradingBotEnvironment(data_val,window_size)
best_total_profit = -np.inf
env.debug_level = 1
best_reward = 0

val_rewards = []
train_rewards = []
val_profits = []
train_profits = []
losses = []
start = time.time()
if phase == 'train':
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        agent.set_mode_train()
        for step in range(max_steps_per_episode):

            action = agent.make_action(state)

            # take the action and observe the next state and reward
            next_state, reward, done, info = env.step(action)

            episode_reward += reward

            agent.add_experience(state, action, reward, next_state, done)

            # train the agent
            agent.update_policy()

            # update the target model if necessary
            if step % update_target_interval == 0:
                agent.update_target_model()

            # transition to the next state
            state = next_state

            if done:
                break
        train_rewards.append(episode_reward)
        train_profits.append(info['TotalProfit'])
        losses.append(sum(agent.history['loss'])/len(agent.history['loss']))
        agent.history['loss'] = []
        state_val = env_val.reset()
        val_reward = 0
        agent.set_mode_eval()
        for step_val in range(max_steps_per_episode):
            action_val = agent.make_eval_action(state_val)
            #print(action_val)
            # take the action and observe the next state and reward
            next_state_val, reward_val, done_val, info_val = env_val.step(action_val)
            val_reward += reward_val

            # transition to the next state
            state_val = next_state_val

            if done_val:
                break
        val_rewards.append(val_reward)
        val_profits.append(info_val['TotalProfit'])
        print("Validation profit: "+str(env_val.formatPrice(info_val['TotalProfit'])))
        print("Total Val Reward: "+str(val_reward))
        print("------------")

        # log the episode reward and average reward
        if info_val['TotalProfit'] > 0 and info['TotalProfit']>0 and episode != 0 and info_val['TotalProfit'] > best_total_profit:
        #if  info_val['TotalProfit']>= best_total_profit:
            best_total_profit = info_val['TotalProfit']
            agent.save_model()
    print("Total execution time: "+str(time.time()-start))
    #agent.save_model()
    HelperMethods.plot_result(train_rewards,val_rewards,"Rewards Plot","Reward","Episode","Train","Validation")
    HelperMethods.plot_result(train_profits,val_profits,"Profits Plot","Profit","Episode","Train","Validation")
    HelperMethods.plot_result_single(losses,'Loss Plot','Episode','MSE')
else:
    profits = []
    agent.policy_net.load_state_dict(torch.load("saved_dqn_agent_model.pth"))
    agent.set_mode_eval()
    env.debug_level = 3
    state = env.reset()
    for step in range(max_steps_per_episode):
        action = agent.make_eval_action(state)

        # take the action and observe the next state and reward
        next_state, reward, done, info = env.step(action)

        # transition to the next state
        state = next_state

        if done:
            print("The start budget was: "+str(env.formatPrice(info['StartBudget'])))
            print("The final budget is: " + str(env.formatPrice(info['CurrentBudget'])))
            break