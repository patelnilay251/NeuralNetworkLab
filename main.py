import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Dropout:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.mask = None

    def generate_mask(self, shape):
        self.mask = np.random.rand(*shape) < self.dropout_rate

    def apply_dropout(self, inputs):
        if self.mask is None:
            return inputs
        return inputs * self.mask / self.dropout_rate if self.dropout_rate != 0 else inputs


class Regularization:
    def __init__(self, reg_type, reg_strength):
        self.reg_type = reg_type
        self.reg_strength = reg_strength

    def compute_regularization(self, weights):
        if self.reg_type == 'l1':
            return self.reg_strength * np.sign(weights)
        elif self.reg_type == 'l2':
            return self.reg_strength * weights

class Parameters:
    def __init__(self, input_size, output_size, learning_rate, regularization=None, dropout=None):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.zeros(output_size)
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.dropout = dropout

    def apply_regularization(self):
        if self.regularization:
            self.weights -= self.regularization.compute_regularization(self.weights)

class Neuron:
    def __init__(self, activation):
        self.activation = activation

    def compute_activation(self, inputs):
        return self.activation(inputs)

    def compute_derivative(self, inputs):
        if callable(self.activation):
            return self.activation(inputs, derivative=True)
        else:
            if self.activation == 'relu':
                return Activation.relu_derivative(inputs, derivative=True)
            elif self.activation == 'sigmoid':
                return Activation.sigmoid_derivative(inputs, derivative=True)
            elif self.activation == 'tanh':
                return Activation.tanh_derivative(inputs, derivative=True)
            elif self.activation == 'softmax':
                return Activation.softmax_derivative(inputs, derivative=True)


class Activation:
    
    @staticmethod
    def relu(x, derivative=False):
        if derivative:
            return np.where(x > 0, 1, 0)
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x, derivative=False):
        if derivative:
            return Activation.sigmoid(x) * (1 - Activation.sigmoid(x))
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh(x, derivative=False):
        if derivative:
            return 1 - np.square(Activation.tanh(x))
        return np.tanh(x)

    @staticmethod
    def softmax(x, derivative=False):
        # Softmax derivative is a bit more complex, generally not used in backpropagation
        raise NotImplementedError("Softmax derivative not implemented")

        
    @staticmethod
    def relu_derivative(x, derivative=False):
        if derivative:
            return np.where(x > 0, 1, 0)
        return np.maximum(0, x)

    @staticmethod
    def sigmoid_derivative(x, derivative=False):
        if derivative:
            return Activation.sigmoid(x) * (1 - Activation.sigmoid(x))
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def tanh_derivative(x, derivative=False):
        if derivative:
            return 1 - np.square(Activation.tanh(x))
        return np.tanh(x)

    @staticmethod
    def softmax_derivative(x, derivative=False):
        # Softmax derivative is a bit more complex, generally not used in backpropagation
        raise NotImplementedError("Softmax derivative not implemented")

        
class Layer:
    def __init__(self, neuron, parameters):
        self.neuron = neuron
        self.parameters = parameters

    def forward(self, inputs):
        aggregated = np.dot(inputs, self.parameters.weights) + self.parameters.bias
        activated = self.neuron.compute_activation(aggregated)
        if self.parameters.dropout:
            self.parameters.dropout.generate_mask(activated.shape)
            activated = self.parameters.dropout.apply_dropout(activated)
        return activated

    def backward(self, inputs, gradients):
        print("Gradients shape:", gradients.shape)
        print("Inputs shape:", inputs.shape)
        activation_gradients = gradients * self.neuron.compute_derivative(inputs)
        print("Activation gradients shape:", activation_gradients.shape)
        if self.parameters.dropout:
            activation_gradients = self.parameters.dropout.apply_dropout(activation_gradients)
        weights_gradients = np.dot(inputs.T, activation_gradients)
        bias_gradients = np.sum(activation_gradients, axis=0)
        new_gradients = np.dot(activation_gradients, self.parameters.weights.T)
        if self.parameters.regularization:
            weights_gradients += self.parameters.regularization.compute_regularization(self.parameters.weights)
        self.parameters.weights -= self.parameters.learning_rate * weights_gradients
        self.parameters.bias -= self.parameters.learning_rate * bias_gradients
        return new_gradients


class ForwardPropagation:
    @staticmethod
    def forward(inputs, layers):
        activations = [inputs]
        for layer in layers:
            activated = layer.forward(activations[-1])
            activations.append(activated)
        return activations

class BackwardPropagation:
    @staticmethod
    def backward(inputs, outputs, gradients, layers):
        for i in range(len(layers) - 1, 0, -1):
            gradients = layers[i].backward(inputs, gradients)
            inputs = outputs[i-1]
        return gradients

class Model:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward_propagation(self, inputs):
        return ForwardPropagation.forward(inputs, self.layers)

    def backward_propagation(self, inputs, outputs, gradients):
        return BackwardPropagation.backward(inputs, outputs, gradients, self.layers)


# Load Breast Cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Preprocess the dataset
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Update input size of the first layer
input_size = X_train.shape[1]

# Model configuration
model = Model()
model.add_layer(Layer(Neuron(Activation.relu), Parameters(input_size=input_size, output_size=4, learning_rate=0.01)))
model.add_layer(Layer(Neuron(Activation.relu), Parameters(input_size=4, output_size=4, learning_rate=0.01)))
model.add_layer(Layer(Neuron(Activation.sigmoid), Parameters(input_size=4, output_size=1, learning_rate=0.01)))

epochs = 1000
for epoch in range(epochs):
    # Forward propagation
    outputs = model.forward_propagation(X_train)
    predictions = outputs[-1]

    # Compute loss (MSE)
    loss = np.mean((predictions - y_train) ** 2)
    
    # Backward propagation
    gradients = 2 * (predictions - y_train) / len(X_train)
    model.backward_propagation(X_train, outputs, gradients)
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss}")

test_outputs = model.forward_propagation(X_test)[-1]
test_predictions = (test_outputs > 0.5).astype(int)  # Assuming binary classification
accuracy = np.mean(test_predictions == y_test)
print("Test Accuracy:", accuracy)
