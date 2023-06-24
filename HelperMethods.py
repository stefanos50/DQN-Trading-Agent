import numpy as np
import torch
import pandas as pd
from matplotlib import pyplot as plt


def initialize_hardware(hw_choice='cuda'):
    if hw_choice == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    if torch.cuda.is_available() and hw_choice == 'cuda':
        print('Using device: ', device)
        print('Using gpu: ',torch.cuda.get_device_name(0))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        print('Using device: ', device)
    return device

def get_data(file_name):
    df = pd.read_csv('data/'+str(file_name)+'.csv')
    close_column = df['Close']
    close_column = close_column.values.tolist()
    return close_column

def plot_actions(prices,actions,min,max,width=2):
    prices = np.array(prices)[min:max]
    actions = np.array(actions)[min:max]
    indices = [i for i in range(1,len(prices)+1)]
    indices = np.array(indices)
    plt.plot(indices, prices, color='blue',linewidth = '0.5')

    buy_indexes = np.where(actions == 0)[0]
    sell_indexes = np.where(actions == 1)[0]

    plt.plot(indices[buy_indexes], prices[buy_indexes],'p', color='green', linewidth='0.5',label='buy', markersize=width)
    plt.plot(indices[sell_indexes], prices[sell_indexes],"v", color='red', linewidth='0.5',label='sell', markersize=width)
    plt.title("Agent Actions")
    plt.xlabel("Index")
    plt.ylabel("Price")
    plt.legend(loc='best')
    plt.show()

def plot_result(x,y,title,y_label,x_label,x_legend,y_legend):
    plt.plot(x)
    plt.plot(y)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend([x_legend, y_legend], loc='upper left')
    plt.show()

def plot_result_single(x,title,y_label,x_label):
    plt.plot(x)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.show()