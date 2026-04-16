import numpy as np
LAYER_REGISTRY = {}
class Layer:
    def forward(self, x): pass
    def backward(self, x): pass
    def update(self, lr): pass
    def get_config(self): pass
    def from_config(self): pass

class Dense(Layer):
    
    def __init__(self, input_size, output_size):
        self.W = np.random.randn(output_size, input_size) * np.sqrt(2. / input_size)
        self.b = np.zeros((output_size, 1))
        self.best_W = 0
        self.best_b = 0

    def forward(self, X):
        self.X = X
        return self.W.dot(X) + self.b
    
    def backward(self, dZ):
        m = self.X.shape[1]
        self.dW = (1 / m) * dZ.dot(self.X.T)
        self.db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        return self.W.T.dot(dZ)
    
    def get_state(self):
        return {
            "type": "Dense",
            "config": self.get_config(),
            "W": self.W,
            "b": self.b
        }

    def get_config(self):
        return {
            "input_size": self.W.shape[1],
            "output_size": self.W.shape[0]
        }
    
    @classmethod
    def from_state(cls, state):
        layer = cls(**state["config"])
        layer.W = state["W"]
        layer.b = state["b"]
        return layer
    

LAYER_REGISTRY["Dense"] = Dense

class ReLU():
    def update(self, lr): pass
    def save(self): pass    
    
    def forward(self, X):
        self.Z = X
        return np.maximum(0, X)

    def backward(self, dA):
        return dA * (self.Z > 0)
    
    def get_state(self):
        return {
            "type": "ReLU",
            "config": {}
        }

    @classmethod
    def from_state(cls, state):
        return cls()
    
    
LAYER_REGISTRY["ReLU"] = ReLU

class Softmax():
    def save(self): pass
    def update(self, lr): pass

    def backward(self, dA):
        return dA
    
    def forward(self, X):
        exp_Z = np.exp(X - np.max(X, axis=0, keepdims=True))
        self.A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
        return self.A
    
    def get_state(self):
        return {
            "type": "Softmax",
            "config": {}
        }

    @classmethod
    def from_state(cls, state):
        return cls()
    
LAYER_REGISTRY["Softmax"] = Softmax


