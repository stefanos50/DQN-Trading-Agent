from torch import nn
import torch

class DQN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=64):
        super(DQN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

        #for layer in self.layers:
            #if isinstance(layer, nn.Linear):
                #torch.nn.init.orthogonal_(layer.weight)

    def forward(self, x):
        x = self.layers(x)
        return x

    def save_model(self):
        torch.save(self.state_dict(), "saved_dqn_agent_model.pth")