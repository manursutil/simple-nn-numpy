import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork():
    def __init__(self, input_size, hidden_size, output_size):
        np.random.seed(1)
        
        # (input -> hidden)
        self.hidden_weights = np.random.randn(input_size, hidden_size) * 0.01
        # (hidden -> output)
        self.output_weights = np.random.randn(hidden_size, output_size) * 0.01
        
        self.hidden_bias = np.zeros((1, hidden_size))
        self.output_bias = np.zeros((1, output_size))
        
    def forward(self, inputs):
        inputs = inputs.astype(float)
        
        self.hidden_layer_input = np.dot(inputs, self.hidden_weights) + self.hidden_bias
        self.hidden_layer_output = relu(self.hidden_layer_input)
        
        self.output_layer_input = np.dot(self.hidden_layer_output, self.output_weights) + self.output_bias
        final_output = sigmoid(self.output_layer_input)
        
        return final_output
    
    def train(self, training_inputs, training_output, training_iterations, learning_rate=0.1):
        for i in range(training_iterations):
            final_output = self.forward(training_inputs)
            
            # Error of output layer
            output_error = training_output - final_output
            output_delta = output_error * sigmoid_derivative(final_output)
            
            # Error of hidden layer
            hidden_error = np.dot(output_delta, self.output_weights.T)
            hidden_delta = hidden_error * relu_derivative(self.hidden_layer_output)
            
            # Weight adjustments
            output_weight_adjustments = np.dot(self.hidden_layer_output.T, output_delta)
            hidden_weight_adjustments = np.dot(training_inputs.T, hidden_delta)
            
            # Bias adjustments
            output_bias_adjustment = np.sum(output_delta, axis=0, keepdims=True)
            hidden_bias_adjustment = np.sum(hidden_delta, axis=0, keepdims=True)
            
            self.output_weights += learning_rate * output_weight_adjustments
            self.hidden_weights += learning_rate * hidden_weight_adjustments
            
            self.output_bias += learning_rate * output_bias_adjustment
            self.hidden_bias += learning_rate * hidden_bias_adjustment
            
            if (i % 1000) == 0:
                loss = np.mean(np.square(output_error))
                print(f"Iteration {i}, Loss: {loss:.4f}")
    
    def think(self, inputs):
        return self.forward(inputs)
    
if __name__ == "__main__":
    
    input_size = 3
    hidden_size = 4
    output_size = 1
    
    neural_network = NeuralNetwork(input_size, hidden_size, output_size)
    
    print("Random initial hidden weights: ")
    print(neural_network.hidden_weights)
    print("Random initial output weights: ")
    print(neural_network.output_weights)
    
    # Training data:
    training_inputs = np.array([[0, 0, 1],
                                [1, 1, 1],
                                [1, 0, 1],
                                [0, 1, 1]])
    
    training_output = np.array([[0, 1, 1, 0]]).T
    
    neural_network.train(training_inputs, training_output, training_iterations=20000, learning_rate=0.05)
    
    print("Hidden weights after training: ")
    print(neural_network.hidden_weights)
    print("Output weights after training: ")
    print(neural_network.output_weights)
    
    print("Hidden bias after training: ")
    print(neural_network.hidden_bias)
    print("Output bias after training: ")
    print(neural_network.output_bias)
    
    # New inputs: 
    A = str(input("Input 1: "))
    B = str(input("Input 2: "))
    C = str(input("Input 3: "))
    
    print("New situation: input data =", A, B, C)
    print("Output data: ")
    new_input = np.array([A, B, C])
    prediction = neural_network.think(new_input)
    print(prediction)