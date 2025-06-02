import numpy as np

class Layer:
    """Base class for all layers"""
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, grad_output):
        raise NotImplementedError
    
    def update_weights(self, learning_rate):
        pass  # Override in layers that have weights

class ConvLayer(Layer):
    def __init__(self, num_filters, filter_size, input_channels, stride=1, padding=0):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.input_channels = input_channels
        self.stride = stride
        self.padding = padding
        
        # He initialization for better convergence
        self.filters = np.random.randn(num_filters, input_channels, filter_size, filter_size) * np.sqrt(2 / (input_channels * filter_size * filter_size))
        self.biases = np.zeros((num_filters, 1))
        
        # Gradients
        self.grad_filters = None
        self.grad_biases = None
        
    def _pad(self, x, pad):
        """Add padding to input (N, C, H, W)"""
        if pad == 0:
            return x
        return np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    
    def _im2col(self, x, filter_h, filter_w, stride):
        """Convert image regions to columns for efficient convolution"""
        N, C, H, W = x.shape
        out_h = (H - filter_h) // stride + 1
        out_w = (W - filter_w) // stride + 1
        
        col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
        
        for y in range(filter_h):
            y_max = y + stride * out_h
            for x_idx in range(filter_w):
                x_max = x_idx + stride * out_w
                col[:, :, y, x_idx, :, :] = x[:, :, y:y_max:stride, x_idx:x_max:stride]
        
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
        return col
        
    def _col2im(self, col, input_shape, filter_h, filter_w, stride):
        """Convert columns back to image format"""
        N, C, H, W = input_shape
        out_h = (H - filter_h) // stride + 1
        out_w = (W - filter_w) // stride + 1
        
        col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
        img = np.zeros((N, C, H, W))
        
        for y in range(filter_h):
            y_max = y + stride * out_h
            for x in range(filter_w):
                x_max = x + stride * out_w
                img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
        
        return img
    
    def forward(self, x):
        """Forward pass with batch support (N, C, H, W)"""
        self.input = x
        N, C, H, W = x.shape
        
        # Add padding
        x_padded = self._pad(x, self.padding)
        
        # Calculate output dimensions
        out_h = (H + 2 * self.padding - self.filter_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.filter_size) // self.stride + 1
        
        # Convert input to columns for efficient convolution
        x_col = self._im2col(x_padded, self.filter_size, self.filter_size, self.stride)
        
        # Reshape filters for matrix multiplication
        w_col = self.filters.reshape(self.num_filters, -1)
        
        # Perform convolution via matrix multiplication
        out = w_col @ x_col.T + self.biases
        
        # Reshape output to (N, F, out_h, out_w)
        out = out.reshape(self.num_filters, out_h, out_w, N).transpose(3, 0, 1, 2)
        
        return out
    
    def backward(self, grad_output):
        """Backward pass with batch support"""
        N, F, out_h, out_w = grad_output.shape
        
        # Gradient w.r.t. bias
        self.grad_biases = np.sum(grad_output, axis=(0, 2, 3)).reshape(-1, 1)
        
        # Gradient w.r.t. weights
        x_padded = self._pad(self.input, self.padding)
        x_col = self._im2col(x_padded, self.filter_size, self.filter_size, self.stride)
        grad_out_reshaped = grad_output.transpose(1, 0, 2, 3).reshape(F, -1)
        self.grad_filters = (grad_out_reshaped @ x_col).reshape(self.filters.shape)
        
        # Gradient w.r.t. input
        w_col = self.filters.reshape(self.num_filters, -1)
        dx_col = w_col.T @ grad_out_reshaped
        dx = self._col2im(dx_col.T, self.input.shape, self.filter_size, self.filter_size, self.stride)
        
        return dx
    
    def update_weights(self, learning_rate):
        """Update weights and biases"""
        if self.grad_filters is not None:
            self.filters -= learning_rate * self.grad_filters
            self.biases -= learning_rate * self.grad_biases

class ReLU(Layer):
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)
    
    def backward(self, grad_output):
        return grad_output * (self.input > 0)

class MaxPool2D(Layer):
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
    
    def forward(self, x):
        """Forward pass with batch support (N, C, H, W)"""
        self.input = x
        N, C, H, W = x.shape
        
        out_h = (H - self.pool_size) // self.stride + 1
        out_w = (W - self.pool_size) // self.stride + 1
        
        # Reshape for efficient pooling
        x_reshaped = x.reshape(N * C, 1, H, W)
        
        # Use im2col for efficient pooling
        x_col = self._pool_im2col(x_reshaped, self.pool_size, self.pool_size, self.stride)
        
        # Find max values and indices
        max_indices = np.argmax(x_col, axis=1)
        out = np.max(x_col, axis=1)
        
        # Store for backward pass
        self.max_indices = max_indices
        self.x_col_shape = x_col.shape
        
        # Reshape output
        out = out.reshape(N, C, out_h, out_w)
        return out
    
    def _pool_im2col(self, x, pool_h, pool_w, stride):
        """im2col for pooling operations"""
        N, C, H, W = x.shape
        out_h = (H - pool_h) // stride + 1
        out_w = (W - pool_w) // stride + 1
        
        col = np.zeros((N, C, pool_h, pool_w, out_h, out_w))
        
        for y in range(pool_h):
            y_max = y + stride * out_h
            for x_idx in range(pool_w):
                x_max = x_idx + stride * out_w
                col[:, :, y, x_idx, :, :] = x[:, :, y:y_max:stride, x_idx:x_max:stride]
        
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w * C, -1)
        return col
    
    def backward(self, grad_output):
        """Backward pass for max pooling"""
        N, C, out_h, out_w = grad_output.shape
        
        # Create gradient matrix for pooling regions
        dx_col = np.zeros(self.x_col_shape)
        grad_flat = grad_output.reshape(-1)
        
        # Distribute gradients to max positions
        dx_col[np.arange(len(grad_flat)), self.max_indices] = grad_flat
        
        # Convert back to image format
        dx = self._pool_col2im(dx_col, self.input.shape, self.pool_size, self.pool_size, self.stride)
        return dx
    
    def _pool_col2im(self, col, input_shape, pool_h, pool_w, stride):
        """col2im for pooling operations"""
        N, C, H, W = input_shape
        out_h = (H - pool_h) // stride + 1
        out_w = (W - pool_w) // stride + 1
        
        col = col.reshape(N, out_h, out_w, C, pool_h, pool_w).transpose(0, 3, 4, 5, 1, 2)
        img = np.zeros((N, C, H, W))
        
        for y in range(pool_h):
            y_max = y + stride * out_h
            for x in range(pool_w):
                x_max = x + stride * out_w
                img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]
        
        return img

class Flatten(Layer):
    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)  # Keep batch dimension
    
    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)

class Dense(Layer):
    def __init__(self, input_size, output_size):
        # Xavier initialization
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(1 / input_size)
        self.biases = np.zeros((1, output_size))
        
        # Gradients
        self.grad_weights = None
        self.grad_biases = None
    
    def forward(self, x):
        self.input = x
        return x @ self.weights + self.biases
    
    def backward(self, grad_output):
        # Gradients w.r.t. weights and biases
        self.grad_weights = self.input.T @ grad_output
        self.grad_biases = np.sum(grad_output, axis=0, keepdims=True)
        
        # Gradient w.r.t. input
        return grad_output @ self.weights.T
    
    def update_weights(self, learning_rate):
        """Update weights and biases"""
        if self.grad_weights is not None:
            self.weights -= learning_rate * self.grad_weights
            self.biases -= learning_rate * self.grad_biases

class Dropout(Layer):
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.training = True
    
    def forward(self, x):
        if self.training:
            self.mask = np.random.rand(*x.shape) > self.dropout_rate
            return x * self.mask / (1 - self.dropout_rate)
        else:
            return x
    
    def backward(self, grad_output):
        if self.training:
            return grad_output * self.mask / (1 - self.dropout_rate)
        else:
            return grad_output
    
    def set_training(self, training):
        self.training = training

class BatchNormalization(Layer):
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        
        # Learnable parameters
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))
        
        # Running statistics
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))
        
        # Training mode
        self.training = True
        
        # Gradients
        self.grad_gamma = None
        self.grad_beta = None
    
    def forward(self, x):
        if self.training:
            # Calculate batch statistics
            self.batch_mean = np.mean(x, axis=0, keepdims=True)
            self.batch_var = np.var(x, axis=0, keepdims=True)
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var
            
            # Normalize
            self.x_centered = x - self.batch_mean
            self.x_norm = self.x_centered / np.sqrt(self.batch_var + self.eps)
        else:
            # Use running statistics during inference
            self.x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        
        # Scale and shift
        return self.gamma * self.x_norm + self.beta
    
    def backward(self, grad_output):
        N = grad_output.shape[0]
        
        # Gradients w.r.t. gamma and beta
        self.grad_gamma = np.sum(grad_output * self.x_norm, axis=0, keepdims=True)
        self.grad_beta = np.sum(grad_output, axis=0, keepdims=True)
        
        # Gradient w.r.t. input
        if self.training:
            dx_norm = grad_output * self.gamma
            dvar = np.sum(dx_norm * self.x_centered * -0.5 * (self.batch_var + self.eps)**(-1.5), axis=0, keepdims=True)
            dmean = np.sum(dx_norm * -1 / np.sqrt(self.batch_var + self.eps), axis=0, keepdims=True) + dvar * np.sum(-2 * self.x_centered, axis=0, keepdims=True) / N
            dx = dx_norm / np.sqrt(self.batch_var + self.eps) + dvar * 2 * self.x_centered / N + dmean / N
        else:
            dx = grad_output * self.gamma / np.sqrt(self.running_var + self.eps)
        
        return dx
    
    def update_weights(self, learning_rate):
        if self.grad_gamma is not None:
            self.gamma -= learning_rate * self.grad_gamma
            self.beta -= learning_rate * self.grad_beta
    
    def set_training(self, training):
        self.training = training

def softmax(x):
    """Numerically stable softmax"""
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

class CrossEntropyLoss:
    def forward(self, predictions, targets):
        """
        predictions: (N, num_classes) - raw logits
        targets: (N,) - class indices or (N, num_classes) - one-hot
        """
        self.predictions = predictions
        self.targets = targets
        
        # Apply softmax
        self.probs = softmax(predictions)
        
        # Handle both one-hot and class index formats
        if targets.ndim == 1:
            # Class indices format
            batch_size = predictions.shape[0]
            log_probs = -np.log(self.probs[np.arange(batch_size), targets] + 1e-10)
        else:
            # One-hot format
            log_probs = -np.sum(targets * np.log(self.probs + 1e-10), axis=1)
        
        return np.mean(log_probs)
    
    def backward(self):
        """Return gradient w.r.t. input predictions"""
        batch_size = self.predictions.shape[0]
        grad = self.probs.copy()
        
        if self.targets.ndim == 1:
            # Class indices format
            grad[np.arange(batch_size), self.targets] -= 1
        else:
            # One-hot format
            grad -= self.targets
        
        return grad / batch_size

class CNN:
    def __init__(self):
        self.layers = []
        self.loss_fn = CrossEntropyLoss()
    
    def add_layer(self, layer):
        """Add a layer to the network"""
        self.layers.append(layer)
        return self
    
    def forward(self, x):
        """Forward pass through all layers"""
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad):
        """Backward pass through all layers"""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def update_weights(self, learning_rate):
        """Update weights for all layers"""
        for layer in self.layers:
            layer.update_weights(learning_rate)
    
    def set_training(self, training=True):
        """Set training mode for layers that need it"""
        for layer in self.layers:
            if hasattr(layer, 'set_training'):
                layer.set_training(training)
    
    def train_step(self, x, y, learning_rate=0.01):
        """Single training step"""
        # Forward pass
        predictions = self.forward(x)
        loss = self.loss_fn.forward(predictions, y)
        
        # Backward pass
        grad = self.loss_fn.backward()
        self.backward(grad)
        
        # Update weights
        self.update_weights(learning_rate)
        
        return loss
    
    def predict(self, x):
        """Make predictions"""
        self.set_training(False)
        predictions = self.forward(x)
        self.set_training(True)
        return np.argmax(softmax(predictions), axis=1)
    
    def evaluate(self, x, y):
        """Evaluate accuracy"""
        predictions = self.predict(x)
        if y.ndim > 1:
            y = np.argmax(y, axis=1)
        return np.mean(predictions == y)
    
    def train(self, X, y, epochs=10, batch_size=32, learning_rate=0.01, validation_data=None):
        """Train the network"""
        N = X.shape[0]
        history = {'loss': [], 'accuracy': []}
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(N)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            num_batches = 0
            
            # Train in batches
            for i in range(0, N, batch_size):
                batch_end = min(i + batch_size, N)
                X_batch = X_shuffled[i:batch_end]
                y_batch = y_shuffled[i:batch_end]
                
                loss = self.train_step(X_batch, y_batch, learning_rate)
                epoch_loss += loss
                num_batches += 1
            
            # Calculate metrics
            avg_loss = epoch_loss / num_batches
            train_acc = self.evaluate(X, y)
            
            history['loss'].append(avg_loss)
            history['accuracy'].append(train_acc)
            
            # Validation metrics
            val_acc = None
            if validation_data:
                X_val, y_val = validation_data
                val_acc = self.evaluate(X_val, y_val)
            
            # Print progress
            if val_acc is not None:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {train_acc:.4f} - Val Acc: {val_acc:.4f}")
            else:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {train_acc:.4f}")
        
        return history

# Example usage and architectures
def create_simple_cnn(input_shape, num_classes):
    """Create a simple CNN architecture"""
    _, C, H, W = input_shape
    
    cnn = CNN()
    cnn.add_layer(ConvLayer(32, 3, C, padding=1))
    cnn.add_layer(ReLU())
    cnn.add_layer(MaxPool2D(2, 2))
    
    cnn.add_layer(ConvLayer(64, 3, 32, padding=1))
    cnn.add_layer(ReLU())
    cnn.add_layer(MaxPool2D(2, 2))
    
    cnn.add_layer(Flatten())
    cnn.add_layer(Dense(64 * (H//4) * (W//4), 128))
    cnn.add_layer(ReLU())
    cnn.add_layer(Dropout(0.5))
    cnn.add_layer(Dense(128, num_classes))
    
    return cnn

def create_advanced_cnn(input_shape, num_classes):
    """Create an advanced CNN with batch normalization"""
    _, C, H, W = input_shape
    
    cnn = CNN()
    # First block
    cnn.add_layer(ConvLayer(32, 3, C, padding=1))
    cnn.add_layer(BatchNormalization(32))
    cnn.add_layer(ReLU())
    cnn.add_layer(MaxPool2D(2, 2))
    
    # Second block
    cnn.add_layer(ConvLayer(64, 3, 32, padding=1))
    cnn.add_layer(BatchNormalization(64))
    cnn.add_layer(ReLU())
    cnn.add_layer(MaxPool2D(2, 2))
    
    # Third block
    cnn.add_layer(ConvLayer(128, 3, 64, padding=1))
    cnn.add_layer(BatchNormalization(128))
    cnn.add_layer(ReLU())
    cnn.add_layer(MaxPool2D(2, 2))
    
    # Classifier
    cnn.add_layer(Flatten())
    cnn.add_layer(Dense(128 * (H//8) * (W//8), 256))
    cnn.add_layer(BatchNormalization(256))
    cnn.add_layer(ReLU())
    cnn.add_layer(Dropout(0.5))
    cnn.add_layer(Dense(256, num_classes))
    
    return cnn

# Example usage:
if __name__ == "__main__":
    # Create sample data (CIFAR-10 like: 32x32x3)
    X_train = np.random.randn(100, 3, 32, 32)
    y_train = np.random.randint(0, 10, 100)
    
    # Create and train model
    model = create_simple_cnn((1, 3, 32, 32), 10)
    history = model.train(X_train, y_train, epochs=5, batch_size=16, learning_rate=0.001)
    
    # Make predictions
    predictions = model.predict(X_train[:10])
    print(f"Predictions: {predictions}")
    print(f"Accuracy: {model.evaluate(X_train, y_train):.4f}")