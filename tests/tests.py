import numpy as np 

class Tester:
    def overfit_tester(self, trainer, X ,Y):
        trainer.train(
            X, Y,
            X, Y,
            epochs = 10000,
            batchsize=X.shape[1]
        )
        print("Final Train Acc:", trainer.train_acc[-1])
        print("Final Train Loss:", trainer.train_losses[-1])