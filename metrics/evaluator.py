import numpy as np
class Evaluator:
    def __init__(self, model, loss, metrics=None):
        self.model = model
        self.loss = loss
        self.metrics = metrics
    
    def evaluate(self, X, Y):
        A = self.model.forward(X)
        preds = np.argmax(A, axis=0)
        results = {
            "loss": self.loss.forward(A, Y)
        }
        if self.metrics:
            for m in self.metrics:
                results[m.name] = m.compute(preds, Y)

        return results