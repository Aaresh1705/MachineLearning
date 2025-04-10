# NeuralNetworkScratch/losses/base.py

class Loss:
    def __init__(self):
        self.d_inputs = None

    def forward(self, y_pred, y_true):
        raise NotImplementedError

    def backward(self, y_pred, y_true):
        raise NotImplementedError

    def __call__(self, y_pred, y_true):
        return self.forward(y_pred, y_true)

    def __bool__(self):
        return False