import numpy as np

class ConvLayer:
    def __init__(self, num_filters, filter_size, input_channels, stride=1, padding=0):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.filters = np.random.randn(num_filters, filter_size, filter_size, input_channels) * 0.1
        self.biases = np.zeros(num_filters)

    def forward(self, input):
        self.input = input
        H, W, C = input.shape
        F, S, P = self.filter_size, self.stride, self.padding
        out_H = (H - F + 2*P) // S + 1
        out_W = (W - F + 2*P) // S + 1
        padded_input = np.pad(input, ((P, P), (P, P), (0, 0)), mode='constant')
        self.padded_input = padded_input  # Save for backward pass
        output = np.zeros((out_H, out_W, self.num_filters))
        
        for f in range(self.num_filters):
            for i in range(out_H):
                for j in range(out_W):
                    h_start = i * S
                    h_end = h_start + F
                    w_start = j * S
                    w_end = w_start + F
                    input_patch = padded_input[h_start:h_end, w_start:w_end, :]
                    output[i, j, f] = np.sum(input_patch * self.filters[f]) + self.biases[f]
        return output

    def backward(self, grad_output):
        P = self.padding
        grad_padded_input = np.zeros_like(self.padded_input)
        self.grad_biases = np.sum(grad_output, axis=(0, 1))
        self.grad_filters = np.zeros_like(self.filters)
        flipped_filters = np.flip(self.filters, (1, 2))  # Rotate 180 degrees

        # Compute gradients for filters
        for f in range(self.num_filters):
            for i in range(grad_output.shape[0]):
                for j in range(grad_output.shape[1]):
                    h_start = i * self.stride
                    h_end = h_start + self.filter_size
                    w_start = j * self.stride
                    w_end = w_start + self.filter_size
                    input_patch = self.padded_input[h_start:h_end, w_start:w_end, :]
                    self.grad_filters[f] += grad_output[i, j, f] * input_patch

        # Compute gradient for input
        for f in range(self.num_filters):
            for i in range(grad_output.shape[0]):
                for j in range(grad_output.shape[1]):
                    h_start = i * self.stride
                    h_end = h_start + self.filter_size
                    w_start = j * self.stride
                    w_end = w_start + self.filter_size
                    grad_patch = grad_output[i, j, f] * flipped_filters[f]
                    grad_padded_input[h_start:h_end, w_start:w_end, :] += grad_patch

        # Remove padding to get grad_input
        grad_input = grad_padded_input[P:-P, P:-P, :] if P != 0 else grad_padded_input
        return grad_input
    
class ReLU:
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, grad_output):
        return grad_output * (self.input > 0)

class MaxPool2D:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, input):
        self.input = input
        H, W, C = input.shape
        PS, S = self.pool_size, self.stride
        out_H = (H - PS) // S + 1
        out_W = (W - PS) // S + 1
        output = np.zeros((out_H, out_W, C))
        self.max_indices = np.zeros((out_H, out_W, C, 2), dtype=int)

        for c in range(C):
            for i in range(out_H):
                for j in range(out_W):
                    h_start = i * S
                    h_end = h_start + PS
                    w_start = j * S
                    w_end = w_start + PS
                    patch = input[h_start:h_end, w_start:w_end, c]
                    max_val = np.max(patch)
                    output[i, j, c] = max_val
                    max_idx = np.unravel_index(np.argmax(patch), (PS, PS))
                    self.max_indices[i, j, c] = (h_start + max_idx[0], w_start + max_idx[1])

        return output

    def backward(self, grad_output):
        grad_input = np.zeros_like(self.input)
        for c in range(grad_output.shape[2]):
            for i in range(grad_output.shape[0]):
                for j in range(grad_output.shape[1]):
                    h, w = self.max_indices[i, j, c]
                    grad_input[h, w, c] += grad_output[i, j, c]
        return grad_input

class Flatten:
    def forward(self, input):
        self.input_shape = input.shape
        return input.reshape(-1)
    
    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)
    
class Dense:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.biases = np.zeros(output_size)

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.biases
    
    def backward(self, grad_output):
        self.grad_weights = np.dot(self.input.T, grad_output)
        self.grad_biases = np.sum(grad_output, axis=0)
        return np.dot(grad_output, self.weights.T)

def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / exps.sum()

class CrossEntropyLoss:
    def forward(self, x, y_true):
        self.y_true = y_true
        self.probs = softmax(x)
        return -np.log(self.probs[y_true])
    
    def backward(self):
        grad = self.probs.copy()
        grad[self.y_true] -= 1
        return grad
    
class CNN:
    def __init__(self):
        self.layers = [
            ConvLayer(num_filters=8, filter_size=3, input_channels=1, padding=1),
            ReLU(),
            MaxPool2D(pool_size=2, stride=2),
            ConvLayer(num_filters=16, filter_size=3, input_channels=8, padding=1),
            ReLU(),
            MaxPool2D(pool_size=2, stride=2),
            Flatten(),
            Dense(input_size=16*7*7, output_size=10)  # Example for MNIST: after two poolings, 7x7x16
        ]
        self.loss_fn = CrossEntropyLoss()

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def train_step(self, x, y_true, lr):
        # Forward pass
        output = self.forward(x)
        loss = self.loss_fn.forward(output, y_true)
        
        # Backward pass
        grad = self.loss_fn.backward()
        self.backward(grad)
        
        # Update parameters
        for layer in self.layers:
            if isinstance(layer, ConvLayer) or isinstance(layer, Dense):
                layer.weights -= lr * layer.grad_filters if isinstance(layer, ConvLayer) else lr * layer.grad_weights
                layer.biases -= lr * layer.grad_biases
        
        return loss
    


