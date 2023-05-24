# Neural Network from scratch


class Layer:
    """ Creates a base layer """
    def __init__(self) -> None:
        self.input = None
        self.output = None

    def forward(self):
        pass

    def backward(self):
        pass


class Dense(Layer):
    """ Creates a neural network layer where each neuron in one layer connects to every other neuron in the next layer """
    def __init__(self, input_size: int, output_size: int) -> None:
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, _input):
        self.input = _input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        input_gradient -= learning_rate * np.dot(np.transpose(self.weights), output_gradient)
        self.weights -= learning_rate * np.dot(output_gradient, np.transpose(self.input))
        self.bias -= learning_rate * output_gradient
        return input_gradient


class Activation(Layer):
    """ Applies Activation Function to neural network layer """
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, _input):
        self.input = _input
        return activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, activation_prime(self.input))
