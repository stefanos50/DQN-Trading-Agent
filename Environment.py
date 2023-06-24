import math
import random

import numpy as np

import HelperMethods


class TradingBotEnvironment:
    def __init__(self, data,window):
        self.data = data
        self.window = window
        self.max_steps = len(data) - 1
        self.inventory = []
        self.max_inventory = len(self.data)
        self.total_profit = 0
        self.budget = 10000
        self.current_budget = self.budget
        self.actions = []
        self.debug_level = 0

    def reset(self):
        self.current_step = 0
        self.inventory = []
        self.total_profit = 0
        self.current_budget = self.budget
        state = self.get_state(self.current_step)
        return np.array(state)

    def formatPrice(self,n):
        return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

    def step(self, action):
        self.actions.append(action)

        reward = 0
        if action == 0:  # buy
            if self.current_budget - self.data[self.current_step] >= 0:
                reward = (self.current_budget - self.budget) / self.budget * 0.1
                self.current_budget = self.current_budget - self.data[self.current_step]
                self.inventory.append(self.data[self.current_step])
                if self.debug_level == 3:
                    print("Buy: " + self.formatPrice(self.data[self.current_step]))
               # reward = (self.current_budget - self.budget) / self.budget
            else:
                self.actions[-1] = -100
                reward = -0.01

        elif action == 1 and len(self.inventory) > 0:  # sell
            random_asset = random.randrange(len(self.inventory))
            bought_price = self.inventory.pop(random_asset)
            self.current_budget = self.current_budget + self.data[self.current_step]
            #reward = max((self.current_budget - self.budget) / self.budget * 10 ,0)
            reward = max(((self.data[self.current_step] - bought_price) / bought_price)*10,0)
            self.total_profit += self.data[self.current_step] - bought_price
            if self.debug_level == 3:
                print("Sell: " + self.formatPrice(self.data[self.current_step]) + " | Profit: " + self.formatPrice(self.data[self.current_step] - bought_price))
        elif action == 1 and len(self.inventory) <= 0:
            self.actions[-1] = -100
            reward = -0.01

        self.current_step += 1
        next_state = self.get_state(self.current_step)

        done = self.current_step == self.max_steps

        if done:
            if len(self.inventory) > 0:
                for i in range(len(self.inventory)):
                    self.current_budget = self.current_budget + self.data[self.current_step]
                    if self.debug_level == 3:
                        print("Sell: " + self.formatPrice(self.data[self.current_step]) + " | Profit: " + self.formatPrice(
                            self.data[self.current_step] - self.inventory[i]))
                self.inventory = []


        #print(next_state)
        if done:
            if self.debug_level == 1 or self.debug_level == 2 or self.debug_level == 3:
                print("Total Profit: "+str(self.formatPrice(self.current_budget - self.budget)))
            if self.debug_level == 2 or self.debug_level == 3:
                self.plot_actions()


        return np.array(next_state), reward, done, {'TotalProfit':(self.current_budget - self.budget),'StartBudget':self.budget,'CurrentBudget':self.current_budget}


    def get_state(self,t):
        n = self.window + 1
        d = t - n + 1
        block = self.data[d:t + 1] if d >= 0 else -d * [self.data[0]] + self.data[0:t + 1]
        #res = [self.position,self.shares,self.current_budget]
        average = ((sum(block))/len(block))
        res = [np.clip(self.current_budget/self.budget,0,1),np.clip(len(self.inventory),0,1)]

        for i in range(n - 1):
            res.append((block[i + 1] - block[i]) / block[i])

        return [res]

    def plot_actions(self):
        HelperMethods.plot_actions(self.data, self.actions, 0, len(self.data))
        HelperMethods.plot_actions(self.data, self.actions, 1800, 2100, width=5)


