# mycelium/optimizers/sdg.py


class Optimizer:
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate

    def update(self, layer):
        raise NotImplementedError

    def __bool__(self):
        return False

    def post_update_params(self):
        raise NotImplementedError

    def pre_update_params(self):
        raise NotImplementedError
