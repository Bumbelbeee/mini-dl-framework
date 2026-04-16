import numpy as np
class Trainer():
    def __init__(self, model, loss, optimizer, early_stopping=None):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.early_stopping = early_stopping

        self.train_acc = []
        self.train_losses = []
        self.val_losses = []
        self.val_acc = []

    def train(self, X_train, Y_train, X_val, Y_val, epochs=10, batchsize=32, model_name="best_model"):
        # Shuffle
        

        
        best_loss = np.inf
        for epoch in range(epochs):
            perm = np.random.permutation(X_train.shape[1])
            X_train = X_train[:, perm]
            Y_train = Y_train[perm]
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0
            # TRAIN
            for k in range(0, X_train.shape[1], batchsize):
                batch_X = X_train[:, k : k+batchsize]
                batch_Y = Y_train[k : k+batchsize]
                preds = self.model.forward(batch_X)
                loss = self.loss.forward(preds, batch_Y)
                grad = self.loss.backward(preds, batch_Y)
                
                self.model.backward(grad)
                self.optimizer.step(self.model.layers)

                pred_labels = np.argmax(preds, axis=0)

                epoch_loss += loss * batch_Y.size
                epoch_correct += np.sum(pred_labels == batch_Y)
                epoch_total += batch_Y.size

            train_loss = epoch_loss / epoch_total
            train_acc = epoch_correct / epoch_total
            

            self.train_losses.append(train_loss)
            self.train_acc.append(train_acc)


            # Validation

            val_preds = self.model.forward(X_val)
            val_loss = self.loss.forward(val_preds, Y_val)

            if val_loss < best_loss:
                best_loss = val_loss
                self.model.save(model_name)

            pred_labels = np.argmax(val_preds, axis=0)

            val_acc = np.mean(pred_labels == Y_val)

            self.val_acc.append(val_acc)
            self.val_losses.append(val_loss)

            print(f"Epoch {epoch}: train_loss={train_loss}, train_acc={train_acc}, val_loss={val_loss}, val_acc={val_acc}")

            # Early Stop
            if self.early_stopping:
                if self.early_stopping.check(val_loss):
                    print("Stopped early")
                    break