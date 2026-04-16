import numpy as np
import matplotlib.pyplot as plt

class Metrics:
    def compute(self, preds, y): pass

class Accuracy():
    def __init__(self):
        self.name = "accuracy"

    def compute(self, preds, y):
        return np.mean(preds == y)
    

class ConfusionMatrix():
    def __init__(self):
        self.name = "confusion_matrix"
        
    def compute(self, predictions, actuals):
        labels = np.unique(np.concatenate((actuals, predictions)))
        label_to_index = {label: idx for idx, label in enumerate(labels)}
        matrix = np.zeros((len(labels), len(labels)), dtype=int)
        for i in range(len(actuals)):
            true = label_to_index[actuals[i]]
            pred = label_to_index[predictions[i]]
            matrix[true][pred] += 1
        matrix = matrix / matrix.sum(axis=1, keepdims=True)

        return {
            "matrix": matrix,
            "labels": labels
        }
    
