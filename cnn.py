import numpy as np
import matplotlib.pyplot as plt

class ConvolutionLayer:
    """
    Core convolution operation - the heart of CNN
    Mathematically: (f * g)(t) = ∫ f(τ)g(t-τ)dτ
    In discrete form: (f * g)[n] = Σ f[m]g[n-m]
    """
    def __init__(self, num_filters, filter_size, input_channels):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.input_channels = input_channels
        
        # Initialize filters with small random values (Xavier initialization)
        # Shape: (num_filters, input_channels, filter_height, filter_width)
        self.filters = np.random.randn(num_filters, input_channels, filter_size, filter_size) * 0.1
        self.biases = np.zeros(num_filters)
        
    def forward(self, input_data):
        """
        Forward pass: Apply convolution operation
        Input shape: (height, width, channels) 
        Output shape: (new_height, new_width, num_filters)
        """
        self.last_input = input_data
        h, w, c = input_data.shape
        
        # Calculate output dimensions (no padding, stride=1)
        output_h = h - self.filter_size + 1
        output_w = w - self.filter_size + 1
        
        # Initialize output
        output = np.zeros((output_h, output_w, self.num_filters))
        
        # Apply each filter
        for f in range(self.num_filters):
            # Slide the filter across the input
            for i in range(output_h):
                for j in range(output_w):
                    # Extract the region of input
                    region = input_data[i:i+self.filter_size, j:j+self.filter_size, :]
                    
                    # Element-wise multiplication and sum (convolution)
                    # This is the core mathematical operation of CNN
                    output[i, j, f] = np.sum(region * self.filters[f]) + self.biases[f]
        
        return output
    
    def backward(self, d_output, learning_rate=0.01):
        """
        Backward pass: Compute gradients and update parameters
        This implements backpropagation through the convolution operation
        """
        # Gradients for filters and biases
        d_filters = np.zeros_like(self.filters)
        d_biases = np.zeros_like(self.biases)
        d_input = np.zeros_like(self.last_input)
        
        output_h, output_w, _ = d_output.shape
        
        for f in range(self.num_filters):
            # Gradient w.r.t. bias (sum of all gradients for this filter)
            d_biases[f] = np.sum(d_output[:, :, f])
            
            for i in range(output_h):
                for j in range(output_w):
                    # Gradient w.r.t. filter weights
                    region = self.last_input[i:i+self.filter_size, j:j+self.filter_size, :]
                    # Ensure proper broadcasting by iterating over channels
                    for c in range(self.input_channels):
                        d_filters[f, c] += d_output[i, j, f] * region[:, :, c]
                    
                    # Gradient w.r.t. input (for backpropagation to previous layer)
                    for c in range(self.input_channels):
                        d_input[i:i+self.filter_size, j:j+self.filter_size, c] += d_output[i, j, f] * self.filters[f, c]
        
        # Update parameters using gradient descent
        self.filters -= learning_rate * d_filters
        self.biases -= learning_rate * d_biases
        
        return d_input

class ReLUActivation:
    """
    ReLU (Rectified Linear Unit) activation function
    Mathematical definition: f(x) = max(0, x)
    Derivative: f'(x) = 1 if x > 0, else 0
    """
    def forward(self, input_data):
        self.last_input = input_data
        return np.maximum(0, input_data)
    
    def backward(self, d_output):
        # Gradient of ReLU: 1 where input > 0, 0 elsewhere
        return d_output * (self.last_input > 0)

class MaxPoolingLayer:
    """
    Max pooling reduces spatial dimensions by taking maximum value in each region
    This provides translation invariance and reduces computational load
    """
    def __init__(self, pool_size=2):
        self.pool_size = pool_size
    
    def forward(self, input_data):
        self.last_input = input_data
        h, w, c = input_data.shape
        
        # Calculate output dimensions
        new_h = h // self.pool_size
        new_w = w // self.pool_size
        
        output = np.zeros((new_h, new_w, c))
        
        # Store max positions for backpropagation
        self.max_positions = np.zeros_like(input_data, dtype=bool)
        
        for i in range(new_h):
            for j in range(new_w):
                for k in range(c):
                    # Extract pool region
                    start_i = i * self.pool_size
                    start_j = j * self.pool_size
                    pool_region = input_data[start_i:start_i+self.pool_size, 
                                           start_j:start_j+self.pool_size, k]
                    
                    # Find maximum value and its position
                    max_val = np.max(pool_region)
                    output[i, j, k] = max_val
                    
                    # Mark the position of maximum for backpropagation
                    max_pos = np.unravel_index(np.argmax(pool_region), pool_region.shape)
                    self.max_positions[start_i + max_pos[0], start_j + max_pos[1], k] = True
        
        return output
    
    def backward(self, d_output):
        """
        Backward pass: Only the maximum positions receive gradients
        """
        d_input = np.zeros_like(self.last_input)
        h, w, c = d_output.shape
        
        for i in range(h):
            for j in range(w):
                for k in range(c):
                    # Distribute gradient only to the maximum position
                    start_i = i * self.pool_size
                    start_j = j * self.pool_size
                    
                    for pi in range(self.pool_size):
                        for pj in range(self.pool_size):
                            if self.max_positions[start_i + pi, start_j + pj, k]:
                                d_input[start_i + pi, start_j + pj, k] = d_output[i, j, k]
        
        return d_input

class FlattenLayer:
    """
    Flatten multi-dimensional input to 1D for fully connected layers
    """
    def forward(self, input_data):
        self.input_shape = input_data.shape
        return input_data.flatten()
    
    def backward(self, d_output):
        return d_output.reshape(self.input_shape)

class DenseLayer:
    """
    Fully connected layer - standard neural network layer
    Mathematical operation: output = input @ weights + bias
    """
    def __init__(self, input_size, output_size):
        # Xavier initialization for better training
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros(output_size)
    
    def forward(self, input_data):
        self.last_input = input_data
        return np.dot(input_data, self.weights) + self.biases
    
    def backward(self, d_output, learning_rate=0.01):
        # Gradients
        d_weights = np.outer(self.last_input, d_output)
        d_biases = d_output
        d_input = np.dot(d_output, self.weights.T)
        
        # Update parameters
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases
        
        return d_input

class SoftmaxLayer:
    """
    Softmax activation for classification
    Mathematical definition: softmax(x_i) = exp(x_i) / Σ exp(x_j)
    Converts logits to probability distribution
    """
    def forward(self, input_data):
        # Numerical stability: subtract max value
        shifted = input_data - np.max(input_data)
        exp_values = np.exp(shifted)
        self.output = exp_values / np.sum(exp_values)
        return self.output
    
    def backward(self, d_output):
        # For softmax + cross-entropy, this is simplified in practice
        return d_output

class SimpleCNN:
    """
    Complete CNN implementation with core mathematical operations
    Architecture: Conv -> ReLU -> MaxPool -> Flatten -> Dense -> Softmax
    """
    def __init__(self, input_shape, num_classes):
        h, w, c = input_shape
        
        # Build network layers
        self.conv1 = ConvolutionLayer(num_filters=8, filter_size=3, input_channels=c)
        self.relu1 = ReLUActivation()
        self.pool1 = MaxPoolingLayer(pool_size=2)
        self.flatten = FlattenLayer()
        
        # Calculate dense layer input size after conv and pooling
        conv_output_h = h - 3 + 1  # 3 is filter size
        conv_output_w = w - 3 + 1
        pool_output_h = conv_output_h // 2
        pool_output_w = conv_output_w // 2
        dense_input_size = pool_output_h * pool_output_w * 8  # 8 filters
        
        self.dense1 = DenseLayer(dense_input_size, 32)
        self.relu2 = ReLUActivation()
        self.dense2 = DenseLayer(32, num_classes)
        self.softmax = SoftmaxLayer()
        
        print(f"Network Architecture:")
        print(f"Input: {input_shape}")
        print(f"Conv1: {conv_output_h}x{conv_output_w}x8")
        print(f"Pool1: {pool_output_h}x{pool_output_w}x8")
        print(f"Dense1: 32 neurons")
        print(f"Output: {num_classes} classes")
    
    def forward(self, x):
        """Forward pass through entire network"""
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)
        x = self.flatten.forward(x)
        x = self.dense1.forward(x)
        x = self.relu2.forward(x)
        x = self.dense2.forward(x)
        x = self.softmax.forward(x)
        return x
    
    def backward(self, predicted, actual, learning_rate=0.01):
        """Backward pass - backpropagation through entire network"""
        # Calculate loss gradient (for cross-entropy + softmax)
        loss_gradient = predicted.copy()
        loss_gradient[actual] -= 1  # Cross-entropy gradient
        
        # Backpropagate through all layers
        grad = self.softmax.backward(loss_gradient)
        grad = self.dense2.backward(grad, learning_rate)
        grad = self.relu2.backward(grad)
        grad = self.dense1.backward(grad, learning_rate)
        grad = self.flatten.backward(grad)
        grad = self.pool1.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.conv1.backward(grad, learning_rate)
    
    def train_step(self, x, y, learning_rate=0.01):
        """Single training step"""
        # Forward pass
        prediction = self.forward(x)
        
        # Calculate cross-entropy loss
        loss = -np.log(prediction[y] + 1e-10)  # Add small epsilon to prevent log(0)
        
        # Backward pass
        self.backward(prediction, y, learning_rate)
        
        return loss, prediction
    
    def predict(self, x):
        """Make prediction"""
        prediction = self.forward(x)
        return np.argmax(prediction)

# Demonstration with synthetic data
def create_sample_data():
    """Create simple synthetic data for demonstration"""
    # Create simple patterns: vertical lines, horizontal lines, and diagonal lines
    data = []
    labels = []
    
    np.random.seed(42)
    
    for _ in range(30):  # 30 samples of each class
        # Class 0: Vertical line pattern
        img = np.zeros((8, 8, 1))
        col = np.random.randint(2, 6)  # Random column position
        img[:, col, 0] = 1.0 + np.random.normal(0, 0.1, 8)  # Add slight noise
        data.append(img)
        labels.append(0)
        
        # Class 1: Horizontal line pattern  
        img = np.zeros((8, 8, 1))
        row = np.random.randint(2, 6)  # Random row position
        img[row, :, 0] = 1.0 + np.random.normal(0, 0.1, 8)  # Add slight noise
        data.append(img)
        labels.append(1)
        
        # Class 2: Diagonal pattern
        img = np.zeros((8, 8, 1))
        for i in range(8):
            if i < 8:
                img[i, i, 0] = 1.0 + np.random.normal(0, 0.1)  # Add slight noise
        data.append(img)
        labels.append(2)
    
    return np.array(data), np.array(labels)

def visualize_learning_process():
    """Demonstrate CNN learning process with visualization"""
    print("=== Basic CNN from Scratch - Mathematical Implementation ===\n")
    
    # Create sample data
    X, y = create_sample_data()
    print(f"Created dataset: {len(X)} samples, {X.shape[1:]} input shape")
    print(f"Classes: 0=Vertical lines, 1=Horizontal lines, 2=Diagonal lines\n")
    
    # Create and initialize CNN
    cnn = SimpleCNN(input_shape=(8, 8, 1), num_classes=3)
    print()
    
    # Training loop
    num_epochs = 50
    losses = []
    accuracies = []
    
    print("Training Progress:")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        correct = 0
        
        # Shuffle data
        indices = np.random.permutation(len(X))
        
        for i in indices:
            # Single sample training
            loss, prediction = cnn.train_step(X[i], y[i], learning_rate=0.1)
            epoch_loss += loss
            
            if np.argmax(prediction) == y[i]:
                correct += 1
        
        avg_loss = epoch_loss / len(X)
        accuracy = correct / len(X)
        
        losses.append(avg_loss)
        accuracies.append(accuracy)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:2d}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.3f}")
    
    print("\n" + "="*60)
    print("MATHEMATICAL CONCEPTS DEMONSTRATED:")
    print("="*60)
    print("1. CONVOLUTION: Feature detection through filter sliding")
    print("   - Each filter learns to detect specific patterns")
    print("   - Mathematical operation: Σ(input × filter) + bias")
    print()
    print("2. ACTIVATION (ReLU): Non-linearity introduction")
    print("   - f(x) = max(0, x)")
    print("   - Enables learning of complex patterns")
    print()
    print("3. POOLING: Spatial dimension reduction")
    print("   - Provides translation invariance")
    print("   - Reduces computational complexity")
    print()
    print("4. BACKPROPAGATION: Gradient-based learning")
    print("   - Chain rule application through all layers")
    print("   - Weight updates: w = w - α∇w")
    print()
    print("5. CROSS-ENTROPY LOSS: Classification objective")
    print("   - L = -log(p_correct_class)")
    print("   - Drives probability toward correct class")
    
    # Test the trained model
    print(f"\nFinal Training Accuracy: {accuracies[-1]:.3f}")
    print(f"Final Loss: {losses[-1]:.4f}")
    
    # Show some predictions
    print(f"\nSample Predictions:")
    print("-" * 30)
    for i in range(5):
        pred = cnn.predict(X[i])
        actual = y[i]
        class_names = ['Vertical', 'Horizontal', 'Diagonal']
        print(f"Sample {i+1}: Predicted={class_names[pred]}, Actual={class_names[actual]}")
    
    return cnn, losses, accuracies

# Run the demonstration
if __name__ == "__main__":
    model, training_losses, training_accuracies = visualize_learning_process()
    
    print(f"\n" + "="*60)
    print("KEY MATHEMATICAL INSIGHTS:")
    print("="*60)
    print("• Convolution detects local features through learned filters")
    print("• ReLU activation enables non-linear decision boundaries") 
    print("• Max pooling provides spatial invariance and efficiency")
    print("• Backpropagation optimizes all parameters jointly")
    print("• The network learns hierarchical feature representations")
    print(f"• Training converged from {training_accuracies[0]:.3f} to {training_accuracies[-1]:.3f} accuracy")