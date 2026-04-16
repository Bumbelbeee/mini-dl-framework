import numpy as np
from core.layers import LAYER_REGISTRY
from core.serialization import load_model, save_model
class Sequential:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, x):
        for layer in reversed(self.layers):
            x = layer.backward(x)
        return x
    
    def save(self, name):
        save_model(self, name)
    
    def load(self, name):
        return load_model(name)

        