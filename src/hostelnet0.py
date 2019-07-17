# ======================================================
# Leandro Bugnon (lbugnon@sinc.unl.edu.ar)
# sinc(i) - http://sinc.unl.edu.ar/
# ======================================================
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision  

class hostelnet0(nn.Module):
    
    def __init__(self,nclases):
        """Conv + fully connected """

        super(hostelnet0,self).__init__() 

        nclasses=nclases
        ncanales=3

        # ResNet pre-entrenada con el dataset ImageNet
        self.model=torchvision.models.resnet152(pretrained=True,progress=False) 
        nfeat=2048 # features a la salida de la penultima capa de la ResNet

        # Se reemplaza la ultima capa para resolver el problema actual
        self.model.fc=nn.Linear(nfeat,nclasses)

        
    def forward(self,x):

        return self.model(x)        
    
