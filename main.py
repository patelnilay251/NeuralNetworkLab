import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.base import BaseEstimator, RegressorMixin


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
            return getattr(Activation, f"{self.activation}_derivative")(inputs, derivative=True)


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
        activation = self.neuron.compute_activation(inputs)
        activation_gradients = gradients * self.neuron.compute_derivative(activation)
        if self.parameters.dropout:
            activation_gradients = self.parameters.dropout.apply_dropout(activation_gradients)
        weights_gradients = np.dot(activation.T, activation_gradients.T).T
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
            inputs = outputs[i - 1]
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


def preprocess_data(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y


def train_model(X_train, y_train, model, epochs=1000):
    for epoch in range(epochs):
        outputs = model.forward_propagation(X_train)
        predictions = outputs[-1]
        loss = np.mean((predictions - y_train) ** 2)
        gradients = 2 * (predictions - y_train) / len(X_train)
        model.backward_propagation(X_train, outputs, gradients)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss}")


def evaluate_model(X_test, y_test, model):
    test_outputs = model.forward_propagation(X_test)[-1]
    test_predictions = (test_outputs > 0.5).astype(int)
    accuracy = np.mean(test_predictions == y_test)
    print("Test Accuracy:", accuracy)


# Load Breast Cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Preprocess the dataset
X_processed, y_processed = preprocess_data(X, y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y_processed, test_size=0.2, random_state=42)

# Update input size of the first layer
input_size = X_train.shape[1]


class ModelWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, learning_rate=0.01, dropout_rate=0, reg_strength=0, reg_type=None):
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.reg_strength = reg_strength
        self.reg_type = reg_type
        self.model = None

    def fit(self, X, y):
        input_size = X.shape[1]
        self.model = create_model(self.learning_rate, self.dropout_rate, self.reg_strength, self.reg_type)
        train_model(X, y, self.model)
        return self

    def predict(self, X):
        return (self.model.forward_propagation(X)[-1] > 0.5).astype(int)

# Perform grid search with ModelWrapper
grid_search = GridSearchCV(estimator=ModelWrapper(), param_grid=param_grid, cv=3, scoring=make_scorer(mean_squared_error), verbose=2)
grid_result = grid_search.fit(X_train, y_train)

# Define a function to create the model
def create_model(learning_rate=0.01, dropout_rate=0, reg_strength=0, reg_type=None):
    model = Model()
    model.add_layer(Layer(Neuron(Activation.relu), Parameters(input_size=input_size, output_size=4, learning_rate=learning_rate, regularization=Regularization(reg_type, reg_strength), dropout=Dropout(dropout_rate))))
    model.add_layer(Layer(Neuron(Activation.relu), Parameters(input_size=4, output_size=4, learning_rate=learning_rate, regularization=Regularization(reg_type, reg_strength), dropout=Dropout(dropout_rate))))
    model.add_layer(Layer(Neuron(Activation.sigmoid), Parameters(input_size=4, output_size=1, learning_rate=learning_rate, regularization=Regularization(reg_type, reg_strength), dropout=Dropout(dropout_rate))))
    return model

# Define the parameter grid
param_grid = {
    'learning_rate': [0.001, 0.01, 0.1],
    'dropout_rate': [0, 0.1, 0.2],
    'reg_strength': [0, 0.001, 0.01],
    'reg_type': [None, 'l1', 'l2']
}

# Create the model
model = create_model()

# Perform grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring=make_scorer(mean_squared_error), verbose=2)
grid_result = grid_search.fit(X_train, y_train)

# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
