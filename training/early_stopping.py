import numpy as np

class EarlyStop():
    def __init__(self, min_delta = 0.001, patience = 10):
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.patience = patience
        self.no_improvement_counter = 0


    def check(self, loss):
        if loss < self.best_loss - self.min_delta:
            self.best_loss=loss
            self.no_improvement_counter = 0
        else:
            self.no_improvement_counter += 1
        
        if self.no_improvement_counter < self.patience:
            return False
        else:
            print("STOPPED EARLY")
            return True
