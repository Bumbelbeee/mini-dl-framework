class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr

    def step(self, layers):
        for layer in layers:
            if hasattr(layer, "W"):
                layer.W -= self.lr * layer.dW
            if hasattr(layer, "b"):
                layer.b -= self.lr * layer.db