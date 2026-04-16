from core.model import Sequential
from core.layers import Dense, ReLU, Softmax
from core.loss import CrossEntropy
from core.optimizer import SGD
from training.trainer import Trainer
from training.early_stopping import EarlyStop
from data.mnist import MNIST
from metrics.evaluator import Evaluator
from metrics.metrics import Accuracy, ConfusionMatrix
from visualisation.plots import Visualizer
from core.serialization import load_model
from tests.tests import Tester
dataset = MNIST()

X_train, Y_train, X_val, Y_val = dataset.train_val_split()
loss = CrossEntropy(num_classes=10)


model = Sequential([
    Dense(784, 256),
    ReLU(),
    Dense(256, 128),
    ReLU(),
    Dense(128, 10),
    Softmax()
]) 

loss = CrossEntropy(num_classes=10)
optimizer = SGD(lr=0.01)

early_stop = EarlyStop(patience=10)

# Training

trainer = Trainer(
    model=model,
    loss=loss,
    optimizer=optimizer,
    early_stopping=early_stop
) 

trainer.train(
    X_train, Y_train,
    X_val, Y_val,
    epochs=20,
    model_name="MNIST"
) 


metrics = [
    Accuracy(),
    ConfusionMatrix()
]


# Evaluation
evaluator = Evaluator(model=model,
                      loss=loss,
                      metrics=metrics
                      )

X_test, Y_test = dataset.get_test()
results = evaluator.evaluate(X_test, Y_test)

# Visualisation


vis = Visualizer()

vis.plot_loss(trainer=trainer)
vis.plot_accuracy(trainer=trainer)
vis.plot_confusion_matrix(results["confusion_matrix"]["matrix"], results["confusion_matrix"]["labels"])
