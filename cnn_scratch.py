import numpy as np

class CNN:
    def __init__(self, input_shape, filter_sizes, filter_counts, stride=1, padding=0):
        """
        Initialize a Convolutional Neural Network
        
        Args:
            input_shape: Tuple of (channels, height, width)
            filter_sizes: List of filter sizes (assume square filters)
            filter_counts: List of number of filters for each convolutional layer
            stride: Stride for the convolution operation
            padding: Padding for the convolution operation
        """
        self.input_shape = input_shape
        self.filter_sizes = filter_sizes
        self.filter_counts = filter_counts
        self.stride = stride
        self.padding = padding
        self.layers = len(filter_sizes)
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        in_channels = input_shape[0]
        for i in range(self.layers):
            # Initialize weights with He initialization
            weight = np.random.randn(filter_counts[i], in_channels, filter_sizes[i], filter_sizes[i]) * np.sqrt(2 / (in_channels * filter_sizes[i] * filter_sizes[i]))
            bias = np.zeros((filter_counts[i], 1))
            
            self.weights.append(weight)
            self.biases.append(bias)
            
            in_channels = filter_counts[i]
        
        # Parameters for fully connected layers
        self.fc_weights = None
        self.fc_bias = None
        self.num_classes = None
        
        # Store activations and gradients for backpropagation
        self.activations = []
        self.conv_outputs = []
    
    def add_fully_connected(self, num_classes):
        """Add a fully connected layer to the network"""
        self.num_classes = num_classes
        
        # Calculate the size of the flattened conv output
        output_size = self._compute_output_size()
        
        # Initialize weights and bias for FC layer (Xavier initialization)
        self.fc_weights = np.random.randn(num_classes, output_size) * np.sqrt(1 / output_size)
        self.fc_bias = np.zeros((num_classes, 1))
    
    def _compute_output_size(self):
        """Calculate the size of the output from the last convolutional layer"""
        h, w = self.input_shape[1], self.input_shape[2]
        channels = self.input_shape[0]
        
        for i in range(self.layers):
            # Calculate output dimensions after convolution
            h = (h - self.filter_sizes[i] + 2 * self.padding) // self.stride + 1
            w = (w - self.filter_sizes[i] + 2 * self.padding) // self.stride + 1
            channels = self.filter_counts[i]
        
        return channels * h * w
    
    def _pad(self, x, pad):
        """Add padding to input"""
        if pad == 0:
            return x
        
        result = np.zeros((x.shape[0], x.shape[1], x.shape[2] + 2 * pad, x.shape[3] + 2 * pad))
        result[:, :, pad:-pad, pad:-pad] = x
        return result
    
    def _im2col(self, x, filter_h, filter_w, stride):
        """
        Convert image regions to columns for efficient convolution
        This is a common optimization for convolution operations
        """
        N, C, H, W = x.shape
        out_h = (H - filter_h) // stride + 1
        out_w = (W - filter_w) // stride + 1
        
        # Create an array to store the columns
        col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
        
        # Fill the columns
        for y in range(filter_h):
            y_max = y + stride * out_h
            for x in range(filter_w):
                x_max = x + stride * out_w
                col[:, :, y, x, :, :] = x[:, :, y:y_max:stride, x:x_max:stride]
        
        # Reshape to match the expected output format
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
        return col
    
    def _col2im(self, col, input_shape, filter_h, filter_w, stride):
        """Convert columns back to image format"""
        N, C, H, W = input_shape
        out_h = (H - filter_h) // stride + 1
        out_w = (W - filter_w) // stride + 1
        
        # Reshape the column array
        col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
        
        # Create an array for the result
        img = np.zeros((N, C, H, W))
        
        # Fill the image
        for y in range(filter_h):
            y_max = y + stride * out_h
            for x in range(filter_w):
                x_max = x + stride * out_w
                img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
        
        return img
    
    def _convolve(self, x, weight, bias, stride, padding):
        """Perform convolution operation"""
        N, C, H, W = x.shape
        F, _, HH, WW = weight.shape
        
        # Add padding
        x_padded = self._pad(x, padding)
        
        # Calculate output dimensions
        out_h = (H + 2 * padding - HH) // stride + 1
        out_w = (W + 2 * padding - WW) // stride + 1
        
        # Convert input to columns
        x_col = self._im2col(x_padded, HH, WW, stride)
        
        # Reshape weights for matrix multiplication
        w_col = weight.reshape(F, -1)
        
        # Perform convolution via matrix multiplication
        out = w_col @ x_col.T
        
        # Add bias
        out = out + bias
        
        # Reshape output
        out = out.reshape(F, out_h, out_w, N).transpose(3, 0, 1, 2)
        
        return out
    
    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU activation function"""
        return np.where(x > 0, 1, 0)
    
    def softmax(self, x):
        """Softmax activation function"""
        # Subtract max for numerical stability
        shifted = x - np.max(x, axis=0, keepdims=True)
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores, axis=0, keepdims=True)
    
    def forward(self, x):
        """Forward pass through the network"""
        self.activations = []
        self.conv_outputs = []
        
        # Store input for backprop
        self.activations.append(x)
        
        # Forward through convolutional layers
        for i in range(self.layers):
            # Convolution
            conv_output = self._convolve(x, self.weights[i], self.biases[i], self.stride, self.padding)
            self.conv_outputs.append(conv_output)
            
            # Apply ReLU activation
            activation = self.relu(conv_output)
            self.activations.append(activation)
            
            # Update x for next layer
            x = activation
        
        # Flatten the output for fully connected layer
        flattened = x.reshape(x.shape[0], -1).T
        
        # Forward through fully connected layer
        if self.fc_weights is not None:
            fc_output = self.fc_weights @ flattened + self.fc_bias
            self.fc_output = fc_output
            
            # Apply softmax for classification
            probs = self.softmax(fc_output)
            return probs
        else:
            return flattened
    
    def backward(self, dout, learning_rate=0.01):
        """Backward pass through the network"""
        batch_size = dout.shape[1]
        
        # Backpropagation through fully connected layer
        if self.fc_weights is not None:
            dfc_weights = dout @ self.activations[-1].reshape(batch_size, -1)
            dfc_bias = np.sum(dout, axis=1, keepdims=True)
            
            # Update fully connected weights and bias
            self.fc_weights -= learning_rate * dfc_weights / batch_size
            self.fc_bias -= learning_rate * dfc_bias / batch_size
            
            # Compute gradient for the input to the fully connected layer
            dflattened = self.fc_weights.T @ dout
            
            # Reshape back to match the shape of the last convolutional output
            last_conv_shape = self.activations[-1].shape
            dx = dflattened.reshape(last_conv_shape)
        else:
            dx = dout.T
        
        # Backpropagation through convolutional layers
        for i in range(self.layers - 1, -1, -1):
            # Gradient of ReLU
            dx = dx * self.relu_derivative(self.conv_outputs[i])
            
            # Prepare dimensions
            N, F, out_h, out_w = dx.shape
            _, C, HH, WW = self.weights[i].shape
            
            # Gradient with respect to bias
            db = np.sum(dx, axis=(0, 2, 3)).reshape(F, 1)
            
            # Gradient with respect to weights
            x_padded = self._pad(self.activations[i], self.padding)
            dw = np.zeros_like(self.weights[i])
            
            # Calculate weight gradients
            for n in range(N):
                for f in range(F):
                    for h in range(out_h):
                        for w in range(out_w):
                            h_start = h * self.stride
                            h_end = h_start + HH
                            w_start = w * self.stride
                            w_end = w_start + WW
                            
                            # Extract the region from the input
                            x_region = x_padded[n, :, h_start:h_end, w_start:w_end]
                            
                            # Update the gradient
                            dw[f] += dx[n, f, h, w] * x_region
            
            # Update weights and biases
            self.weights[i] -= learning_rate * dw / batch_size
            self.biases[i] -= learning_rate * db / batch_size
            
            # Calculate gradient for input to this layer
            if i > 0:
                dx_prev = np.zeros_like(self.activations[i])
                
                # Gradient with respect to input
                for n in range(N):
                    for f in range(F):
                        for h in range(out_h):
                            for w in range(out_w):
                                h_start = h * self.stride
                                h_end = h_start + HH
                                w_start = w * self.stride
                                w_end = w_start + WW
                                
                                # Update the gradient
                                dx_prev[n, :, h_start:h_end, w_start:w_end] += self.weights[i][f] * dx[n, f, h, w]
                
                dx = dx_prev
    
    def cross_entropy_loss(self, probs, y):
        """Calculate cross-entropy loss"""
        batch_size = probs.shape[1]
        
        # Convert one-hot encoding to class indices
        if y.shape[0] > 1:
            y_indices = np.argmax(y, axis=0)
        else:
            y_indices = y
        
        # Get probabilities for correct classes
        probs_correct = probs[y_indices, np.arange(batch_size)]
        
        # Calculate loss (-log(p))
        log_likelihood = -np.log(probs_correct + 1e-10)  # Add small constant for numerical stability
        loss = np.sum(log_likelihood) / batch_size
        
        return loss
    
    def compute_accuracy(self, X, y):
        """Calculate accuracy on a dataset"""
        probs = self.forward(X)
        predictions = np.argmax(probs, axis=0)
        
        # Convert one-hot encoding to class indices if needed
        if y.shape[0] > 1:
            y_indices = np.argmax(y, axis=0)
        else:
            y_indices = y
        
        accuracy = np.mean(predictions == y_indices)
        return accuracy
    
    def train(self, X, y, epochs=10, batch_size=32, learning_rate=0.01):
        """Train the network on a dataset"""
        N = X.shape[0]
        losses = []
        
        for epoch in range(epochs):
            # Shuffle data
            idx = np.random.permutation(N)
            X_shuffled = X[idx]
            y_shuffled = y[idx] if y.ndim == 1 else y[:, idx]
            
            loss_epoch = 0
            num_batches = 0
            
            # Train in batches
            for i in range(0, N, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size] if y.ndim == 1 else y_shuffled[:, i:i+batch_size]
                
                # Forward pass
                probs = self.forward(X_batch)
                
                # Calculate loss
                loss = self.cross_entropy_loss(probs, y_batch)
                loss_epoch += loss
                num_batches += 1
                
                # Compute gradient of softmax + cross-entropy
                dout = probs.copy()
                
                # Convert one-hot encoding to class indices if needed
                if y_batch.ndim > 1 and y_batch.shape[0] > 1:
                    y_indices = np.argmax(y_batch, axis=0)
                else:
                    y_indices = y_batch
                
                # One-hot encode y for gradient calculation
                dout[y_indices, np.arange(X_batch.shape[0])] -= 1
                dout /= X_batch.shape[0]  # Normalize
                
                # Backward pass
                self.backward(dout, learning_rate)
            
            # Calculate average loss for epoch
            avg_loss = loss_epoch / num_batches
            losses.append(avg_loss)
            
            # Calculate accuracy
            accuracy = self.compute_accuracy(X, y)
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return losses