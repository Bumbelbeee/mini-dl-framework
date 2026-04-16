import matplotlib.pyplot as plt
import numpy
class Visualizer:
    def plot_loss(self, trainer):
        plt.plot(trainer.train_losses, label="Train Loss")
        plt.plot(trainer.val_losses, label="Validation Loss")

        plt.legend()
        plt.title("Loss over Time")
        plt.show()

    def plot_accuracy(self, trainer):
        plt.figure()

        plt.plot(trainer.train_acc, label="Train Accuracy")
        plt.plot(trainer.val_acc, label="Validation Accuracy")

        plt.legend()
        plt.title("Accuracy over time")
        plt.show()

    def plot_confusion_matrix(self, matrix, labels):
        plt.figure()

        plt.imshow(matrix, cmap="Blues")
        plt.colorbar()

        plt.xticks(range(len(labels)), labels)
        plt.yticks(range(len(labels)), labels)

        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")

        plt.show()