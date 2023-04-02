# External libraries
import numpy as np


class EarlyStopper:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_counter = 0
        self.min_validation_loss = np.inf

    def __call__(self, validation_loss, debug=False):
        # if new validation loss is found, set new minimum and reset counter
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0

        # if validation loss is above the delta threshold, increase counter
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter > self.max_counter:
                self.max_counter = max(self.counter, self.max_counter)
                if debug and self.max_counter % (self.patience // 20) == 0:
                    print(f'EarlyStopper New Counter Record: {self.max_counter}')
            # return True when counter reaches patience threshold
            if self.counter >= self.patience:
                return True

        # False otherwise
        return False
