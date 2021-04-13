##################################
#
#   Builds a neural network
#
##################################

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import Input

class NetworkBuilder:
    """
    Builds a neural network.
    """

    def Build(network_parameters : dict) -> Sequential:
        """
        Builds a network with specified parameters.
        """
        assert isinstance(network_parameters, dict), "Invalid input type. Expecting argument 'network_parameters' to be of type 'dict'"
        NetworkBuilder._validate_network_parameters(network_parameters)

        model = Sequential()
        model.add(Input(shape=network_parameters["input_shape"]))

        for layer in network_parameters["layers"]:
            nodes, activation_function = layer
            model.add(Dense(nodes, activation=activation_function))

        model.compile(
            loss=network_parameters["loss_function"],
            optimizer=network_parameters["optimizer"])

        return model

    def _validate_network_parameters(network_parameters : dict):
        """
        Validates inputs.

        TODO: Complete validation -- or ignore if it's too much.
        """

        assert 'layers' in network_parameters, "Missing key 'layers' in network parameters"
        assert isinstance(network_parameters['layers'], list), "Invalid type, expected 'layers' to be of type 'list'"

        assert 'input_shape' in network_parameters, "Missing key 'input_shape' in network parameters"
        assert 'loss_function' in network_parameters, "Missing key 'loss_function' in network parameters"
        assert 'optimizer' in network_parameters, "Missing key 'optimizer' in network parameters"


if __name__ == "__main__":
    """
    Demonstrates usage of the NetworkBuilder class.
    """

    from tensorflow.keras.optimizers import Adam
    optimizer = Adam(lr=0.01)

    params = {
        "input_shape" : (4,),                                    # input shape
        "layers" : [(10, 'relu'), (10, 'relu'), (3, 'linear')],  # [(nodes, activation function)]
        "optimizer" : optimizer,                                 # optimizer
        "loss_function" : "mse",                                 # loss function ('mse', etc.)
    }

    model = NetworkBuilder.Build(params)
    model.summary()

    x0 = [[1., 2., 3., 4.]]
    prediction = model.predict(x0)
    print(f"Prediction: {prediction[0]}")
