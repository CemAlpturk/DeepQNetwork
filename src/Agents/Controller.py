from os import path

import numpy as np
from tensorflow import keras


class Controller():
    """
    The class that will generate the necessary actions
    for the simulator.
    """

    def __init__(self, action_space, model, idx):
        """
        TODO: Fill
        """

        self.action_space = action_space
        self.model = model
        self.input_shape = model.input_shape[1]
        self.idx = idx

    def act(self, state, t):
        """
        Given a state, returns an action

        # TODO: Complete and talk about the summary:
            returns the external force to apply to the system.
        """
        state = state[self.idx].reshape(1, self.input_shape)
        action_ind = np.argmax(self.model.predict(state)[0])
        return self.action_space[action_ind]

    def save_controller(
            self,
            file_path : str):
        """
        Saves the model used for the controller to specified file.

        TODO: Complete summary.
        """

        assert isinstance(file_path, str), "Invalid 'file_path' type, expected 'file_path' to be of type 'str'"
        assert not file_path == False, "Invalid 'file_path' value"

        self.model.save(file_path)

    @staticmethod
    def load_controller(
            action_space,
            file_path : str,
            idx):
        """
        Loads a controller with model from specified file paht and action space.

        TODO: Complete summary
        """

        assert isinstance(file_path, str), "Invalid 'file_path' type, expected 'file_path' to be of type 'str'"
        assert not file_path == False, "Invalid 'file_path' value"

        model = keras.models.load_model(file_path, compile=False)
        #model.compile()
        return Controller(action_space, model, idx)


if __name__ == "__main__":
    c = Controller.load_controller([-10, 0, 10], "test.txt")
    print("Successfully loaded controller")
    c.model.summary()
