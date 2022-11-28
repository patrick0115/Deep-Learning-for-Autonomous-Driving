import torch
import torch.nn as nn

class M5(nn.Module):
    def __init__(self, cfg = None):
        super().__init__()
        if cfg is None:
            self.cfg = [128, 128, 256, 512]
        else:
            self.cfg = cfg
        
        modules = []
        modules.append(nn.Conv1d(1, self.cfg[0], kernel_size=40, stride=2, padding=19))
        modules.append(nn.BatchNorm1d(self.cfg[0]))
        modules.append(nn.ReLU())
        modules.append(nn.MaxPool1d(4))
        
        for l in range(1, 4):
            modules.append(nn.Conv1d(self.cfg[l-1], self.cfg[l], kernel_size=3, padding=1))
            modules.append(nn.BatchNorm1d(self.cfg[l]))
            modules.append(nn.ReLU())
            modules.append(nn.MaxPool1d(4))
                
        self.features = nn.Sequential(*modules)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten()
        
        self.fc = nn.Linear(self.cfg[3], 10)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
