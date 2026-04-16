import numpy as np
from core.layers import LAYER_REGISTRY
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(PROJECT_ROOT, "runs")


def save_model(model, name):
    model_path = os.path.join(RUNS_DIR, name)

    data = []
    for layer in model.layers:
        data.append(layer.get_state())

    np.save(model_path, data, allow_pickle=True)

def load_model(name):
    model_path = os.path.join(RUNS_DIR, f"{name}.npy")
    data = np.load(model_path, allow_pickle=True)
    
    layers = []
    for state in data:
        layer_type = state["type"]
        layer_class = LAYER_REGISTRY[layer_type]

        layer = layer_class.from_state(state)
        layers.append(layer)
    
    return layers
