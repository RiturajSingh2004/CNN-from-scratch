# The Complete Guide to Convolutional Neural Networks: From Mathematical Foundations to Production-Ready Implementation

*A comprehensive journey through CNN architecture, mathematics, and implementation details*

---

## 📚 Table of Contents

1. [Introduction & Conceptual Overview](#introduction--conceptual-overview)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Core Layer Architecture](#core-layer-architecture)
4. [Advanced Features & Optimizations](#advanced-features--optimizations)
5. [Training Dynamics & Backpropagation](#training-dynamics--backpropagation)
6. [Implementation Deep Dive](#implementation-deep-dive)
7. [Performance Optimizations](#performance-optimizations)
8. [Best Practices & Production Considerations](#best-practices--production-considerations)
9. [Troubleshooting & Common Pitfalls](#troubleshooting--common-pitfalls)
10. [Advanced Topics & Extensions](#advanced-topics--extensions)

---

## 🌟 Introduction & Conceptual Overview

### What is a Convolutional Neural Network?

A Convolutional Neural Network (CNN) is a deep learning architecture specifically designed for processing grid-like data such as images. Unlike traditional neural networks that flatten images into 1D vectors (destroying spatial relationships), CNNs preserve and exploit spatial hierarchies through specialized operations.

### Why CNNs Matter

```
Traditional NN:    [Image] → [Flatten] → [Dense Layers] → [Output]
                   Loses spatial information ❌

CNN:              [Image] → [Conv + Pool] → [Conv + Pool] → [Dense] → [Output]
                  Preserves spatial relationships ✅
```

### Key Intuitions

1. **Local Connectivity**: Pixels close to each other are more related than distant ones
2. **Translation Invariance**: A cat in the top-left should be recognized as a cat in the bottom-right
3. **Hierarchical Features**: Low-level edges combine to form high-level objects
4. **Parameter Sharing**: The same filter can detect edges everywhere in the image

---

## 🧮 Mathematical Foundations

### 1. Convolution Operation

#### **Mathematical Definition**

For continuous functions:
```
(f * g)(t) = ∫ f(τ)g(t-τ)dτ
```

For discrete 2D images:
```
(I * K)(i,j) = ΣΣ I(m,n) × K(i-m, j-n)
```

Where:
- `I` = Input image
- `K` = Kernel/Filter
- `*` = Convolution operator

#### **Practical 2D Convolution Example**

Given a 3×3 input and 2×2 kernel:

```
Input I:           Kernel K:
[1  2  3]         [1  0]
[4  5  6]         [0  1]
[7  8  9]

Convolution at position (0,0):
Output(0,0) = I(0,0)×K(0,0) + I(0,1)×K(0,1) + I(1,0)×K(1,0) + I(1,1)×K(1,1)
            = 1×1 + 2×0 + 4×0 + 5×1 = 6

Full Output:
[6  8]
[12 14]
```

#### **Output Size Calculation**

```
Output_height = (Input_height - Kernel_height + 2×Padding) / Stride + 1
Output_width  = (Input_width - Kernel_width + 2×Padding) / Stride + 1
```

**Example:**
- Input: 32×32, Kernel: 5×5, Padding: 2, Stride: 1
- Output: (32 - 5 + 2×2) / 1 + 1 = 32×32

### 2. Pooling Operations

#### **Max Pooling**

Takes the maximum value in each pooling window:

```
Input (4×4):           Max Pool 2×2:
[1  3  2  4]          [5  6]
[5  1  6  2]    →     [9  8]
[7  9  3  8]
[2  4  1  5]

Process:
Top-left 2×2: max(1,3,5,1) = 5
Top-right 2×2: max(2,4,6,2) = 6
Bottom-left 2×2: max(7,9,2,4) = 9
Bottom-right 2×2: max(3,8,1,5) = 8
```

#### **Mathematical Formulation**

```
MaxPool(X)[i,j] = max(X[i×s:i×s+k, j×s:j×s+k])
```

Where:
- `s` = stride
- `k` = kernel size

### 3. Activation Functions

#### **ReLU (Rectified Linear Unit)**

```
ReLU(x) = max(0, x)

Derivative:
ReLU'(x) = {1 if x > 0, 0 if x ≤ 0}
```

**Why ReLU?**
- Solves vanishing gradient problem
- Computationally efficient
- Sparse activation (many zeros)
- Biological plausibility

#### **Softmax (for Classification)**

```
Softmax(xi) = exp(xi) / Σj exp(xj)

Properties:
- Output sums to 1
- Converts logits to probabilities
- Amplifies differences between classes
```

### 4. Loss Functions

#### **Cross-Entropy Loss**

For multiclass classification:
```
L = -Σi yi × log(ŷi)

Where:
- yi = true label (one-hot encoded)
- ŷi = predicted probability
```

**Gradient of Cross-Entropy + Softmax:**
```
∂L/∂zi = ŷi - yi

This elegant result makes backpropagation efficient!
```

---

## 🏗️ Core Layer Architecture

### Base Layer Class

```python
class Layer:
    def forward(self, x):
        """Forward pass computation"""
        raise NotImplementedError
    
    def backward(self, grad_output):
        """Backward pass - compute gradients"""
        raise NotImplementedError
    
    def update_weights(self, learning_rate):
        """Update learnable parameters"""
        pass
```

This design follows the **Template Method Pattern**, ensuring consistent interfaces across all layers.

### 1. Convolutional Layer Deep Dive

#### **Architecture Overview**

```
Input: (N, C_in, H, W)
↓
Convolution: C_in → C_out filters
↓
Output: (N, C_out, H', W')
```

#### **Weight Initialization: He Initialization**

```python
std = sqrt(2 / (input_channels × kernel_height × kernel_width))
weights = random_normal(shape) × std
```

**Why He Initialization?**
- Maintains variance of activations across layers
- Prevents vanishing/exploding gradients
- Optimal for ReLU activations

#### **Forward Pass Mathematics**

For each output position (i, j) and filter f:
```
output[n, f, i, j] = Σc Σh Σw input[n, c, i×stride+h, j×stride+w] × filter[f, c, h, w] + bias[f]
```

#### **im2col Optimization**

Traditional convolution requires 6 nested loops. The im2col trick converts convolution into matrix multiplication:

```
Original Convolution:
for batch in batches:
    for filter in filters:
        for output_h in output_height:
            for output_w in output_width:
                for input_c in input_channels:
                    for kernel_h in kernel_height:
                        for kernel_w in kernel_width:
                            # 7 nested loops! 😱

im2col Approach:
input_matrix = im2col(input)     # Shape: (N×H'×W', C×Kh×Kw)
filter_matrix = filters.reshape  # Shape: (C_out, C×Kh×Kw)
output = filter_matrix @ input_matrix.T  # Single matrix multiplication! 🚀
```

**Performance Gain:**
- Traditional: O(N × C_out × H' × W' × C_in × Kh × Kw)
- im2col: O(N × H' × W' × C_in × Kh × Kw) + O(C_out × C_in × Kh × Kw × N × H' × W')
- **Result: ~10-100x speedup on modern hardware**

#### **Backward Pass Gradients**

**Gradient w.r.t. Filters:**
```
∂L/∂W[f] = Σn Σh Σw ∂L/∂output[n,f,h,w] × input_patch[n,:,h×s:h×s+k,w×s:w×s+k]
```

**Gradient w.r.t. Input:**
```
∂L/∂input = conv2d(∂L/∂output, W_flipped, padding='full')
```

Where `W_flipped` is the 180° rotated filter (mathematical requirement for convolution).

### 2. Max Pooling Layer

#### **Forward Pass with Index Tracking**

```python
def forward(self, x):
    # For each pooling window, find max and remember its position
    max_val = np.max(window)
    max_idx = np.argmax(window)  # Store for backward pass
    return max_val
```

#### **Backward Pass: Gradient Routing**

```python
def backward(self, grad_output):
    grad_input = zeros_like(input)
    # Route gradients only to positions that were max
    grad_input[max_positions] = grad_output
    return grad_input
```

**Key Insight:** Only the maximum element in each window contributed to the output, so only it receives gradients.

### 3. Dense (Fully Connected) Layer

#### **Mathematical Operations**

Forward:
```
output = input @ weights + bias
Shape: (N, input_size) @ (input_size, output_size) = (N, output_size)
```

Backward:
```
∂L/∂weights = input.T @ grad_output
∂L/∂bias = sum(grad_output, axis=0)
∂L/∂input = grad_output @ weights.T
```

#### **Xavier Initialization**

```python
std = sqrt(1 / input_size)  # Xavier initialization for sigmoid/tanh
std = sqrt(2 / input_size)  # He initialization for ReLU
```

---

## 🚀 Advanced Features & Optimizations

### 1. Batch Normalization

#### **The Problem BN Solves**

During training, the distribution of layer inputs changes as previous layers' parameters update. This **Internal Covariate Shift** slows training and requires careful initialization.

#### **Mathematical Formulation**

For a mini-batch B = {x₁, x₂, ..., xₘ}:

**Step 1: Normalize**
```
μ_B = (1/m) × Σᵢ xᵢ                    # Batch mean
σ²_B = (1/m) × Σᵢ (xᵢ - μ_B)²          # Batch variance
x̂ᵢ = (xᵢ - μ_B) / sqrt(σ²_B + ε)      # Normalize
```

**Step 2: Scale and Shift**
```
yᵢ = γ × x̂ᵢ + β                       # Learnable parameters γ, β
```

#### **Why Scale and Shift?**

Pure normalization forces mean=0, std=1, which might be suboptimal. The network learns optimal mean and std through γ and β.

#### **Running Statistics for Inference**

During training:
```
running_mean = momentum × running_mean + (1-momentum) × batch_mean
running_var = momentum × running_var + (1-momentum) × batch_var
```

During inference:
```
output = γ × (input - running_mean) / sqrt(running_var + ε) + β
```

#### **Backward Pass Gradients**

```python
# Gradients w.r.t. learnable parameters
∂L/∂γ = Σᵢ (∂L/∂yᵢ × x̂ᵢ)
∂L/∂β = Σᵢ ∂L/∂yᵢ

# Gradient w.r.t. input (complex but crucial)
∂L/∂xᵢ = (γ/√(σ²+ε)) × [∂L/∂yᵢ - (1/m)×Σⱼ∂L/∂yⱼ - (x̂ᵢ/m)×Σⱼ(∂L/∂yⱼ×x̂ⱼ)]
```

**Benefits:**
- Faster convergence (2-10x speedup)
- Higher learning rates possible
- Less sensitive to initialization
- Acts as regularization

### 2. Dropout Regularization

#### **The Overfitting Problem**

```
Training Data: High accuracy ✅
Test Data: Low accuracy ❌
→ Model memorized training data instead of learning generalizable patterns
```

#### **Dropout Mechanism**

**Training Phase:**
```python
mask = (random_uniform(shape) > dropout_rate).astype(float)
output = input × mask / (1 - dropout_rate)  # Scale to maintain expectation
```

**Inference Phase:**
```python
output = input  # No dropout, no scaling needed (due to training scaling)
```

#### **Mathematical Intuition**

Expected value during training:
```
E[output] = input × E[mask] / (1 - p) = input × (1-p) / (1-p) = input
```

This ensures consistent expected values between training and inference.

#### **Why Dropout Works**

1. **Prevents Co-adaptation**: Neurons can't rely on specific other neurons
2. **Ensemble Effect**: Training exponentially many sub-networks
3. **Noise Injection**: Forces robust feature learning

### 3. Advanced Weight Initialization

#### **The Initialization Problem**

Poor initialization → Vanishing/Exploding Gradients → Training failure

#### **Xavier/Glorot Initialization**

For symmetric activations (tanh, sigmoid):
```
Var(W) = 1/n_in  or  Var(W) = 2/(n_in + n_out)
```

**Derivation:**
For linear transformation y = Wx + b:
```
Var(y) = n_in × Var(W) × Var(x)
```

To maintain variance: Var(W) = 1/n_in

#### **He Initialization**

For ReLU activations:
```
Var(W) = 2/n_in
```

**Why factor of 2?**
ReLU zeros out half the activations, effectively halving the variance.

---

## 🔄 Training Dynamics & Backpropagation

### The Chain Rule in Deep Networks

For a network f(g(h(x))):
```
∂L/∂x = (∂L/∂f) × (∂f/∂g) × (∂g/∂h) × (∂h/∂x)
```

### Gradient Flow Visualization

```
Input → Conv1 → ReLU → Pool → Conv2 → ReLU → Dense → Output → Loss
  ↑       ↑       ↑      ↑       ↑       ↑       ↑       ↑
  ←━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
                        Gradients flow backward
```

### Common Gradient Problems

#### **Vanishing Gradients**

**Problem:**
```
∂L/∂w₁ = (∂L/∂wₙ) × Π(∂wᵢ/∂wᵢ₋₁)
```

If each gradient term < 1, the product approaches zero exponentially.

**Solutions:**
- ReLU activations (gradient = 1 for positive inputs)
- Batch normalization
- Residual connections
- Proper initialization

#### **Exploding Gradients**

**Problem:** Gradients become too large, causing unstable training.

**Solutions:**
- Gradient clipping: `grad = grad × min(1, threshold/||grad||)`
- Proper initialization
- Learning rate scheduling

### Backpropagation Implementation Details

#### **Computational Graph**

Each operation stores:
1. **Forward values** (for gradient computation)
2. **Backward function** (gradient computation logic)

```python
# Forward pass
z = x @ w + b
a = ReLU(z)
loss = CrossEntropy(a, y)

# Backward pass (automatic differentiation)
grad_a = grad_CrossEntropy(a, y)
grad_z = grad_ReLU(z) * grad_a
grad_w = x.T @ grad_z
grad_x = grad_z @ w.T
```

---

## 💻 Implementation Deep Dive

### Memory Management

#### **Memory Usage Analysis**

For a typical CNN:
```
Input: (32, 3, 224, 224) = 32 × 3 × 224² × 4 bytes = 19.3 MB
Conv1: (32, 64, 224, 224) = 32 × 64 × 224² × 4 bytes = 411 MB
Pool1: (32, 64, 112, 112) = 32 × 64 × 112² × 4 bytes = 103 MB
...
Total: ~2-4 GB for forward pass alone!
```

#### **Memory Optimization Techniques**

1. **Gradient Checkpointing**: Trade computation for memory
2. **Mixed Precision**: Use float16 for forward, float32 for gradients
3. **Batch Size Reduction**: Linear memory scaling
4. **Layer-wise Training**: Train one layer at a time

### Numerical Stability

#### **Common Numerical Issues**

1. **Softmax Overflow**:
   ```python
   # Bad: exp(large_number) → inf
   softmax = exp(x) / sum(exp(x))
   
   # Good: Subtract max for stability
   softmax = exp(x - max(x)) / sum(exp(x - max(x)))
   ```

2. **Log of Zero**:
   ```python
   # Bad: log(0) → -inf
   loss = -log(prediction)
   
   # Good: Add epsilon
   loss = -log(prediction + 1e-10)
   ```

3. **Division by Zero in Batch Norm**:
   ```python
   # Add epsilon to variance
   normalized = (x - mean) / sqrt(var + eps)
   ```

### Threading and Parallelization

#### **Data Parallelism**

```python
# Split batch across devices
batch_size = 32
device_count = 4
per_device_batch = batch_size // device_count  # 8 per device

# Each device processes subset independently
# Gradients averaged across devices
```

#### **Model Parallelism**

```python
# Split model across devices
device_0: Conv layers
device_1: Dense layers

# Data flows between devices during forward/backward pass
```

---

## ⚡ Performance Optimizations

### 1. Algorithmic Optimizations

#### **Fast Convolution Algorithms**

**Winograd Convolution:**
- Reduces multiplications for small kernels (3×3)
- ~2.25x speedup over direct convolution
- Trade-off: Higher memory usage, numerical precision issues

**FFT Convolution:**
- Efficient for large kernels
- O(n log n) vs O(n²) complexity
- Practical for kernel size ≥ 7×7

#### **Memory Access Optimization**

**Cache-Friendly Memory Patterns:**
```python
# Bad: Non-contiguous memory access
for i in range(height):
    for j in range(width):
        for c in range(channels):
            process(data[i, j, c])

# Good: Contiguous memory access
for c in range(channels):
    for i in range(height):
        for j in range(width):
            process(data[i, j, c])
```

### 2. Hardware-Specific Optimizations

#### **SIMD Vectorization**

Modern CPUs can process multiple data points simultaneously:
```python
# Vectorized operations automatically use SIMD
result = np.add(array1, array2)  # Processes 4-8 floats simultaneously
```

#### **GPU Optimization Principles**

1. **Maximize Parallelism**: Use all cores
2. **Memory Coalescing**: Contiguous memory access
3. **Occupancy**: Balance threads per block
4. **Memory Hierarchy**: Utilize shared memory

### 3. Framework-Level Optimizations

#### **Operator Fusion**

```python
# Separate operations
x = conv(input)
x = batch_norm(x)
x = relu(x)

# Fused operation (single kernel)
x = conv_bn_relu_fused(input)  # ~30% speedup
```

#### **Dynamic Batching**

```python
# Batch small inputs together
small_inputs = [img1, img2, img3, img4]
batched = create_batch(small_inputs)
output = model(batched)
split_outputs = split_batch(output)
```

---

## 🎯 Best Practices & Production Considerations

### 1. Model Architecture Design

#### **Layer Depth vs Width Trade-offs**

```
Deeper Networks (ResNet-style):
✅ Better feature hierarchies
✅ More expressive
❌ Harder to train
❌ Vanishing gradients

Wider Networks:
✅ Easier to train
✅ Better parallelization
❌ More parameters
❌ Less feature hierarchy
```

#### **Activation Function Selection**

| Function | Use Case | Pros | Cons |
|----------|----------|------|------|
| ReLU | Hidden layers | Fast, sparse | Dead neurons |
| Leaky ReLU | Hidden layers | No dead neurons | Hyperparameter |
| Swish | Hidden layers | Smooth, learnable | Computationally expensive |
| Sigmoid | Output (binary) | Bounded [0,1] | Vanishing gradients |
| Softmax | Output (multiclass) | Probability distribution | Sensitive to outliers |

### 2. Training Strategies

#### **Learning Rate Scheduling**

```python
# Step decay
lr = initial_lr × (decay_factor)^(epoch // step_size)

# Exponential decay
lr = initial_lr × exp(-decay_rate × epoch)

# Cosine annealing
lr = min_lr + (max_lr - min_lr) × (1 + cos(π × epoch / max_epochs)) / 2
```

#### **Data Augmentation Pipeline**

```python
augmentation_pipeline = [
    RandomHorizontalFlip(p=0.5),
    RandomRotation(degrees=10),
    ColorJitter(brightness=0.2, contrast=0.2),
    RandomResizedCrop(size=224),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
```

### 3. Monitoring and Debugging

#### **Key Metrics to Track**

1. **Loss Curves**: Training vs Validation
2. **Gradient Norms**: Detect vanishing/exploding gradients
3. **Learning Rate**: Ensure proper scheduling
4. **Activation Statistics**: Mean, std, sparsity
5. **Weight Statistics**: Distribution changes over time

#### **Common Training Issues**

| Symptom | Possible Cause | Solution |
|---------|----------------|----------|
| Loss not decreasing | Learning rate too low | Increase LR |
| Loss exploding | Learning rate too high | Decrease LR, gradient clipping |
| Overfitting | Model too complex | Dropout, regularization, more data |
| Underfitting | Model too simple | More layers, more parameters |
| Oscillating loss | Batch size too small | Increase batch size |

### 4. Model Deployment

#### **Model Optimization for Production**

1. **Quantization**: Float32 → Int8 (4x smaller, faster)
2. **Pruning**: Remove unimportant weights
3. **Distillation**: Train smaller model to mimic larger one
4. **ONNX Export**: Cross-platform model format

#### **Inference Optimization**

```python
# Batch similar-sized inputs
def batch_inference(inputs):
    # Group by size
    size_groups = group_by_size(inputs)
    results = []
    
    for size, group in size_groups:
        batch = create_batch(group)
        batch_output = model(batch)
        results.extend(unbatch(batch_output))
    
    return results
```

---

## 🐛 Troubleshooting & Common Pitfalls

### 1. Training Issues

#### **Loss Not Decreasing**

**Diagnostic Steps:**
```python
# 1. Check if model can overfit single batch
single_batch = X[:1], y[:1]
for epoch in range(1000):
    loss = train_step(single_batch)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss {loss}")

# If loss doesn't decrease → Model/implementation bug
# If loss decreases → Data/regularization issue
```

**Common Causes:**
- Learning rate too low/high
- Wrong loss function
- Label encoding issues
- Data preprocessing bugs

#### **Vanishing Gradients Detection**

```python
def check_gradients(model):
    total_norm = 0
    param_count = 0
    
    for layer in model.layers:
        if hasattr(layer, 'grad_weights'):
            grad_norm = np.linalg.norm(layer.grad_weights)
            total_norm += grad_norm
            param_count += 1
            print(f"{layer.__class__.__name__}: {grad_norm:.6f}")
    
    avg_norm = total_norm / param_count
    print(f"Average gradient norm: {avg_norm:.6f}")
    
    if avg_norm < 1e-6:
        print("⚠️  Warning: Possible vanishing gradients")
    elif avg_norm > 1e2:
        print("⚠️  Warning: Possible exploding gradients")
```

### 2. Implementation Bugs

#### **Shape Mismatches**

```python
def debug_shapes(model, input_sample):
    x = input_sample
    print(f"Input shape: {x.shape}")
    
    for i, layer in enumerate(model.layers):
        x = layer.forward(x)
        print(f"Layer {i} ({layer.__class__.__name__}): {x.shape}")
```

#### **Gradient Checking**

```python
def numerical_gradient_check(layer, input_data, eps=1e-7):
    """Compare analytical gradients with numerical gradients"""
    
    # Forward pass
    output = layer.forward(input_data)
    loss = np.sum(output ** 2)  # Simple loss for testing
    
    # Analytical gradient
    grad_output = 2 * output
    analytical_grad = layer.backward(grad_output)
    
    # Numerical gradient
    numerical_grad = np.zeros_like(input_data)
    
    for i in range(input_data.size):
        # Perturb input
        input_plus = input_data.copy().flatten()
        input_minus = input_data.copy().flatten()
        input_plus[i] += eps
        input_minus[i] -= eps
        
        # Compute numerical gradient
        loss_plus = np.sum(layer.forward(input_plus.reshape(input_data.shape)) ** 2)
        loss_minus = np.sum(layer.forward(input_minus.reshape(input_data.shape)) ** 2)
        numerical_grad.flat[i] = (loss_plus - loss_minus) / (2 * eps)
    
    # Compare gradients
    diff = np.abs(analytical_grad - numerical_grad)
    relative_error = diff / (np.abs(analytical_grad) + np.abs(numerical_grad) + eps)
    
    print(f"Max absolute difference: {np.max(diff)}")
    print(f"Max relative error: {np.max(relative_error)}")
    
    if np.max(relative_error) < 1e-5:
        print("✅ Gradient check passed")
    else:
        print("❌ Gradient check failed")
```

### 3. Performance Issues

#### **Memory Profiling**

```python
import psutil
import os

def memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

# Profile memory usage
initial_memory = memory_usage()
model = create_model()
after_model = memory_usage()
print(f"Model creation: {after_model - initial_memory:.1f} MB")

# Train one batch
train_step(X_batch, y_batch)
after_training = memory_usage()
print(f"Training step: {after_training - after_model:.1f} MB")
```

#### **CPU Profiling**

```python
import time

def profile_layer(layer, input_data, num_runs=100):
    # Warm up
    for _ in range(10):
        layer.forward(input_data)
    
    # Profile forward pass
    start_time = time.time()
    for _ in range(num_runs):
        output = layer.forward(input_data)
    forward_time = (time.time() - start_time) / num_runs
    
    # Profile backward pass
    grad_output = np.ones_like(output)
    start_time = time.time()
    for _ in range(num_runs):
        layer.backward(grad_output)
    backward_time = (time.time() - start_time) / num_runs
    
    print(f"{layer.__class__.__name__}:")
    print(f"  Forward: {forward_time*1000:.2f} ms")
    print(f"  Backward: {backward_time*1000:.2f} ms")
```

---

## 🔬 Advanced Topics & Extensions

### 1. Modern CNN Architectures

#### **Residual Connections (ResNet)**

```python
class ResidualBlock(Layer):
    def __init__(self, channels):
        self.conv1 = ConvLayer(channels, 3, channels, padding=1)
        self.bn1 = BatchNormalization(channels)
        self.conv2 = ConvLayer(channels, 3, channels, padding=1)
        self.bn2 = BatchNormalization(channels)
        self.relu = ReLU()
    
    def forward(self, x):
        residual = x
        
        out = self.conv1.forward(x)
        out = self.bn1.forward(out)
        out = self.relu.forward(out)
        
        out = self.conv2.forward(out)
        out = self.bn2.forward(out)
        
        # Skip connection
        out = out + residual
        out = self.relu.forward(out)
        
        return out
```

**Why Residuals Work:**
- Enables training very deep networks (100+ layers)
- Identity mapping provides gradient highway
- Easier optimization landscape

#### **Depthwise Separable Convolutions (MobileNet)**

```python
class DepthwiseSeparableConv(Layer):
    def __init__(self, input_channels, output_channels, kernel_size=3, stride=1, padding=1):
        # Depthwise convolution: each input channel convolved separately
        self.depthwise = ConvLayer(input_channels, kernel_size, input_channels, 
                                 stride=stride, padding=padding, groups=input_channels)
        
        # Pointwise convolution: 1x1 conv to combine features
        self.pointwise = ConvLayer(output_channels, 1, input_channels)
        
    def forward(self, x):
        x = self.depthwise.forward(x)
        x = self.pointwise.forward(x)
        return x
```

**Parameter Reduction:**
```
Standard Conv: Dk × Dk × M × N parameters
Depthwise Separable: Dk × Dk × M + M × N parameters

Reduction Factor: (Dk × Dk × M × N) / (Dk × Dk × M + M × N)
For 3×3 conv: ~8-9x fewer parameters!
```

#### **Attention Mechanisms in CNNs**

```python
class SpatialAttention(Layer):
    def __init__(self, channels):
        self.conv = ConvLayer(1, 7, 2, padding=3)  # 7x7 conv
        self.sigmoid = Sigmoid()
        
    def forward(self, x):
        # Compute spatial statistics
        avg_pool = np.mean(x, axis=1, keepdims=True)  # Channel-wise average
        max_pool = np.max(x, axis=1, keepdims=True)   # Channel-wise max
        
        # Concatenate and process
        combined = np.concatenate([avg_pool, max_pool], axis=1)
        attention_map = self.conv.forward(combined)
        attention_map = self.sigmoid.forward(attention_map)
        
        # Apply attention
        return x * attention_map
```

### 2. Advanced Normalization Techniques

#### **Layer Normalization**

Unlike Batch Normalization (normalizes across batch), Layer Normalization normalizes across features:

```python
class LayerNormalization(Layer):
    def __init__(self, normalized_shape, eps=1e-5):
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = np.ones(normalized_shape)
        self.beta = np.zeros(normalized_shape)
        
    def forward(self, x):
        # Normalize across feature dimensions (not batch)
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        
        self.x_norm = (x - self.mean) / np.sqrt(self.var + self.eps)
        return self.gamma * self.x_norm + self.beta
```

**When to Use:**
- **Batch Norm**: Fixed batch sizes, training/inference consistency
- **Layer Norm**: Variable batch sizes, RNNs, Transformers
- **Group Norm**: Small batch sizes, better for CNNs than Layer Norm

---

## 💻 Complete Implementation Analysis

Let's dive deep into the complete CNN implementation provided in the code, analyzing every component with mathematical precision and practical insights.

### Base Architecture Design

#### **Layer Abstract Base Class**

```python
class Layer:
    def forward(self, x):
        raise NotImplementedError
    
    def backward(self, grad_output):
        raise NotImplementedError
    
    def update_weights(self, learning_rate):
        pass  # Override in layers that have weights
```

**Design Pattern: Template Method**
- Defines the skeleton of layer operations
- Enforces consistent interface across all layers
- Enables polymorphic behavior in network composition

### 1. Convolutional Layer Deep Analysis

#### **Initialization Strategy**

```python
# He initialization for better convergence
self.filters = np.random.randn(num_filters, input_channels, filter_size, filter_size) * np.sqrt(2 / (input_channels * filter_size * filter_size))
```

**Mathematical Justification:**

For ReLU activations, the optimal variance is:
```
Var(W) = 2 / n_in

Where n_in = input_channels × filter_height × filter_width
```

**Why this works:**
1. **Variance Preservation**: Maintains activation variance across layers
2. **Gradient Flow**: Prevents vanishing/exploding gradients
3. **ReLU Optimization**: Accounts for ReLU zeroing half the activations

#### **im2col Implementation Deep Dive**

The `_im2col` method is the heart of efficient convolution:

```python
def _im2col(self, x, filter_h, filter_w, stride):
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
```

**Step-by-Step Breakdown:**

1. **Memory Allocation**: Create 6D tensor for all patches
2. **Patch Extraction**: Extract overlapping patches with stride
3. **Reshape**: Convert to 2D matrix for GEMM operation

**Visual Example:**
```
Input (1×1×4×4):           After im2col (3×3 kernel):
[1  2  3  4]              [1 2 3 5 6 7 9 10 11]
[5  6  7  8]       →      [2 3 4 6 7 8 10 11 12]
[9  10 11 12]             [5 6 7 9 10 11 13 14 15]
[13 14 15 16]             [6 7 8 10 11 12 14 15 16]

Each row = one 3×3 patch flattened
```

#### **Forward Pass Mathematics**

```python
# Perform convolution via matrix multiplication
out = w_col @ x_col.T + self.biases
```

**Mathematical Representation:**
```
Y = W × X + B

Where:
- W: (F, C×Kh×Kw) - Reshaped filters
- X: (C×Kh×Kw, N×H'×W') - im2col patches
- Y: (F, N×H'×W') - Output feature maps
- B: (F, 1) - Bias terms
```

**Computational Complexity:**
- **Traditional**: O(N × F × C × Kh × Kw × H' × W')
- **im2col + GEMM**: O(C × Kh × Kw × N × H' × W') + O(F × C × Kh × Kw × N × H' × W')
- **Practical Speedup**: 10-100x due to optimized BLAS libraries

#### **Backward Pass Implementation**

```python
def backward(self, grad_output):
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
```

**Mathematical Derivations:**

**1. Gradient w.r.t. Bias:**
```
∂L/∂b_f = Σ_n Σ_h Σ_w ∂L/∂y[n,f,h,w]
```
Sum all gradients for each filter's bias.

**2. Gradient w.r.t. Weights:**
```
∂L/∂W[f,c,i,j] = Σ_n Σ_h Σ_w ∂L/∂y[n,f,h,w] × x[n,c,h×s+i,w×s+j]
```
Cross-correlation between input patches and output gradients.

**3. Gradient w.r.t. Input:**
```
∂L/∂x[n,c,i,j] = Σ_f Σ_p Σ_q ∂L/∂y[n,f,p,q] × W[f,c,i-p×s,j-q×s]
```
Full convolution with flipped filters.

### 2. Max Pooling Layer Analysis

#### **Forward Pass with Index Tracking**

```python
def forward(self, x):
    # Use im2col for efficient pooling
    x_col = self._pool_im2col(x_reshaped, self.pool_size, self.pool_size, self.stride)
    
    # Find max values and indices
    max_indices = np.argmax(x_col, axis=1)
    out = np.max(x_col, axis=1)
    
    # Store for backward pass
    self.max_indices = max_indices
```

**Key Insight: Index Preservation**
Unlike convolution, pooling is not differentiable everywhere. We must track which elements were selected during the forward pass.

#### **Backward Pass: Gradient Routing**

```python
def backward(self, grad_output):
    # Create gradient matrix for pooling regions
    dx_col = np.zeros(self.x_col_shape)
    grad_flat = grad_output.reshape(-1)
    
    # Distribute gradients to max positions
    dx_col[np.arange(len(grad_flat)), self.max_indices] = grad_flat
```

**Mathematical Principle:**
```
∂L/∂x_i = {
    ∂L/∂y  if x_i was the maximum in its pooling window
    0       otherwise
}
```

This implements the **subgradient** of the max function.

### 3. Batch Normalization Implementation

#### **Forward Pass Statistics**

```python
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
```

**Mathematical Operations:**

**Step 1: Batch Statistics**
```
μ_B = (1/m) × Σᵢ xᵢ
σ²_B = (1/m) × Σᵢ (xᵢ - μ_B)²
```

**Step 2: Exponential Moving Average**
```
μ_running = α × μ_running + (1-α) × μ_batch
σ²_running = α × σ²_running + (1-α) × σ²_batch
```

**Step 3: Normalization**
```
x̂ᵢ = (xᵢ - μ_B) / √(σ²_B + ε)
yᵢ = γ × x̂ᵢ + β
```

#### **Complex Backward Pass**

```python
def backward(self, grad_output):
    N = grad_output.shape[0]
    
    # Gradients w.r.t. learnable parameters
    self.grad_gamma = np.sum(grad_output * self.x_norm, axis=0, keepdims=True)
    self.grad_beta = np.sum(grad_output, axis=0, keepdims=True)
    
    # Complex gradient w.r.t. input
    dx_norm = grad_output * self.gamma
    dvar = np.sum(dx_norm * self.x_centered * -0.5 * (self.batch_var + self.eps)**(-1.5), axis=0, keepdims=True)
    dmean = np.sum(dx_norm * -1 / np.sqrt(self.batch_var + self.eps), axis=0, keepdims=True) + dvar * np.sum(-2 * self.x_centered, axis=0, keepdims=True) / N
    dx = dx_norm / np.sqrt(self.batch_var + self.eps) + dvar * 2 * self.x_centered / N + dmean / N
```

**Gradient Derivation:**

The gradient computation involves the chain rule applied to the normalization:

```
∂L/∂xᵢ = ∂L/∂ŷᵢ × ∂ŷᵢ/∂x̂ᵢ × ∂x̂ᵢ/∂xᵢ + [terms from μ and σ² dependencies]
```

Breaking this down:
1. **Direct term**: `∂L/∂x̂ᵢ × ∂x̂ᵢ/∂xᵢ`
2. **Variance term**: Effects through σ²
3. **Mean term**: Effects through μ

### 4. Dropout Implementation

#### **Training vs Inference Modes**

```python
def forward(self, x):
    if self.training:
        self.mask = np.random.rand(*x.shape) > self.dropout_rate
        return x * self.mask / (1 - self.dropout_rate)
    else:
        return x
```

**Mathematical Insight:**

**Training Phase:**
```
y = {
    x / (1-p)  with probability (1-p)
    0          with probability p
}
```

**Expected Value:**
```
E[y] = (1-p) × x/(1-p) + p × 0 = x
```

This **inverted dropout** ensures consistent expected values between training and inference.

#### **Regularization Effect**

Dropout prevents overfitting through:
1. **Co-adaptation Prevention**: Neurons can't rely on specific others
2. **Ensemble Learning**: Training 2^n sub-networks simultaneously
3. **Noise Injection**: Forces learning of robust features

### 5. Dense Layer Mathematics

#### **Forward and Backward Pass**

```python
def forward(self, x):
    self.input = x
    return x @ self.weights + self.biases

def backward(self, grad_output):
    self.grad_weights = self.input.T @ grad_output
    self.grad_biases = np.sum(grad_output, axis=0, keepdims=True)
    return grad_output @ self.weights.T
```

**Mathematical Operations:**

**Forward:**
```
Y = XW + B
Shape: (N, din) @ (din, dout) + (1, dout) = (N, dout)
```

**Backward:**
```
∂L/∂W = X^T × ∂L/∂Y    Shape: (din, N) @ (N, dout) = (din, dout)
∂L/∂B = Σ_batch ∂L/∂Y  Shape: sum over batch dimension
∂L/∂X = ∂L/∂Y × W^T    Shape: (N, dout) @ (dout, din) = (N, din)
```

---

## 🧠 Loss Functions and Optimization

### Cross-Entropy Loss Implementation

#### **Numerically Stable Softmax**

```python
def softmax(x):
    """Numerically stable softmax"""
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
```

**Numerical Stability:**
```
# Unstable: exp(x) can overflow
softmax(x) = exp(x) / Σ exp(x)

# Stable: subtract max to prevent overflow
softmax(x) = exp(x - max(x)) / Σ exp(x - max(x))
```

**Mathematical Equivalence:**
```
exp(xᵢ - M) / Σⱼ exp(xⱼ - M) = [exp(-M) × exp(xᵢ)] / [exp(-M) × Σⱼ exp(xⱼ)]
                                = exp(xᵢ) / Σⱼ exp(xⱼ)
```

#### **Cross-Entropy Forward Pass**

```python
def forward(self, predictions, targets):
    self.probs = softmax(predictions)
    
    if targets.ndim == 1:
        # Class indices format
        log_probs = -np.log(self.probs[np.arange(batch_size), targets] + 1e-10)
    else:
        # One-hot format
        log_probs = -np.sum(targets * np.log(self.probs + 1e-10), axis=1)
    
    return np.mean(log_probs)
```

**Mathematical Formulation:**

**For Class Indices:**
```
L = -(1/N) × Σᵢ log(pᵧᵢ)
Where pᵧᵢ is the predicted probability for the correct class yᵢ
```

**For One-Hot Vectors:**
```
L = -(1/N) × Σᵢ Σⱼ yᵢⱼ × log(pᵢⱼ)
```

#### **Elegant Gradient Computation**

```python
def backward(self):
    grad = self.probs.copy()
    
    if self.targets.ndim == 1:
        grad[np.arange(batch_size), self.targets] -= 1
    else:
        grad -= self.targets
    
    return grad / batch_size
```

**Mathematical Beauty:**

The gradient of cross-entropy loss combined with softmax is elegantly simple:
```
∂L/∂zᵢ = pᵢ - yᵢ

Where:
- pᵢ = softmax(zᵢ) (predicted probability)
- yᵢ = target (0 or 1)
- zᵢ = logit (pre-softmax value)
```

This simplicity makes backpropagation computationally efficient!

---

## 🏗️ Network Architecture and Training

### CNN Class Design

#### **Modular Architecture**

```python
class CNN:
    def __init__(self):
        self.layers = []
        self.loss_fn = CrossEntropyLoss()
    
    def add_layer(self, layer):
        self.layers.append(layer)
        return self  # Method chaining
```

**Design Benefits:**
- **Flexibility**: Easy to experiment with different architectures
- **Modularity**: Each layer is independent and testable
- **Extensibility**: New layer types can be added seamlessly

#### **Training Loop Implementation**

```python
def train_step(self, x, y, learning_rate=0.01):
    # Forward pass
    predictions = self.forward(x)
    loss = self.loss_fn.forward(predictions, y)
    
    # Backward pass
    grad = self.loss_fn.backward()
    self.backward(grad)
    
    # Update weights
    self.update_weights(learning_rate)
    
    return loss
```

**Training Pipeline:**
1. **Forward Propagation**: Compute predictions
2. **Loss Calculation**: Measure prediction quality
3. **Backward Propagation**: Compute gradients
4. **Weight Update**: Apply gradient descent

#### **Batch Training with Shuffling**

```python
def train(self, X, y, epochs=10, batch_size=32, learning_rate=0.01):
    for epoch in range(epochs):
        # Shuffle data each epoch
        indices = np.random.permutation(N)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Train in batches
        for i in range(0, N, batch_size):
            batch_end = min(i + batch_size, N)
            X_batch = X_shuffled[i:batch_end]
            y_batch = y_shuffled[i:batch_end]
            
            loss = self.train_step(X_batch, y_batch, learning_rate)
```

**Why Shuffle?**
- **Randomization**: Prevents overfitting to data order
- **Better Convergence**: Reduces correlation between consecutive batches
- **Generalization**: Improves model robustness

---

## 🏛️ Architecture Templates

### Simple CNN Architecture

```python
def create_simple_cnn(input_shape, num_classes):
    _, C, H, W = input_shape
    
    cnn = CNN()
    cnn.add_layer(ConvLayer(32, 3, C, padding=1))      # 32 filters, 3x3
    cnn.add_layer(ReLU())
    cnn.add_layer(MaxPool2D(2, 2))                     # 2x2 pooling
    
    cnn.add_layer(ConvLayer(64, 3, 32, padding=1))     # 64 filters, 3x3
    cnn.add_layer(ReLU())
    cnn.add_layer(MaxPool2D(2, 2))
    
    cnn.add_layer(Flatten())
    cnn.add_layer(Dense(64 * (H//4) * (W//4), 128))   # FC layer
    cnn.add_layer(ReLU())
    cnn.add_layer(Dropout(0.5))
    cnn.add_layer(Dense(128, num_classes))             # Output layer
    
    return cnn
```

**Architecture Analysis:**

**Feature Extraction:**
```
Input: (N, 3, 32, 32)
Conv1: (N, 32, 32, 32)  → 32 feature maps
Pool1: (N, 32, 16, 16)  → Spatial reduction
Conv2: (N, 64, 16, 16)  → 64 feature maps  
Pool2: (N, 64, 8, 8)    → Further reduction
```

**Classification:**
```
Flatten: (N, 64×8×8) = (N, 4096)
Dense1:  (N, 128)                   → Feature compression
Dense2:  (N, num_classes)           → Class scores
```

**Parameter Count:**
```
Conv1: 32 × (3×3×3 + 1) = 896
Conv2: 64 × (3×3×32 + 1) = 18,496
Dense1: 4096 × 128 + 128 = 524,416
Dense2: 128 × 10 + 10 = 1,290
Total: ~545K parameters
```

### Advanced CNN with Batch Normalization

```python
def create_advanced_cnn(input_shape, num_classes):
    cnn = CNN()
    # First block
    cnn.add_layer(ConvLayer(32, 3, C, padding=1))
    cnn.add_layer(BatchNormalization(32))
    cnn.add_layer(ReLU())
    cnn.add_layer(MaxPool2D(2, 2))
    
    # ... more blocks with BN
```

**Benefits of Batch Normalization:**
- **Faster Training**: 2-10x convergence speedup
- **Higher Learning Rates**: More stable training
- **Regularization Effect**: Reduces overfitting
- **Less Sensitivity**: To initialization and hyperparameters

---

## 📊 Performance Analysis and Optimization

### Memory Usage Calculation

For the simple CNN on CIFAR-10 (32×32×3):

```python
# Forward pass memory
Input:    32 × 3 × 32 × 32 × 4 bytes = 393 KB
Conv1:    32 × 32 × 32 × 32 × 4 bytes = 4.19 MB
Pool1:    32 × 32 × 16 × 16 × 4 bytes = 1.05 MB  
Conv2:    32 × 64 × 16 × 16 × 4 bytes = 2.10 MB
Pool2:    32 × 64 × 8 × 8 × 4 bytes = 524 KB
Dense1:   32 × 128 × 4 bytes = 16 KB
Dense2:   32 × 10 × 4 bytes = 1.3 KB

Total Forward: ~8 MB per batch
```

**Memory Optimization Strategies:**
1. **Reduce Batch Size**: Linear memory scaling
2. **Gradient Checkpointing**: Trade computation for memory
3. **Mixed Precision**: Use float16 for activations
4. **In-place Operations**: Reuse memory buffers

### Computational Complexity

#### **FLOPS Analysis**

For each layer type:

**Convolution:**
```
FLOPs = N × C_out × H_out × W_out × (C_in × K_h × K_w + 1)
```

**Dense Layer:**
```
FLOPs = N × output_size × (input_size + 1)
```

**Example for Simple CNN:**
```
Conv1: 32 × 32 × 32 × 32 × (3×3×3 + 1) = 28.3M FLOPs
Conv2: 32 × 64 × 16 × 16 × (3×3×32 + 1) = 151M FLOPs  
Dense1: 32 × 128 × (4096 + 1) = 16.8M FLOPs
Dense2: 32 × 10 × (128 + 1) = 41K FLOPs

Total: ~196M FLOPs per forward pass
```

### Optimization Techniques

#### **Vectorization Benefits**

```python
# Slow: Element-wise operations
for i in range(len(array)):
    result[i] = array[i] * 2

# Fast: Vectorized operations
result = array * 2  # Uses SIMD instructions
```

**Performance Gain:** 4-8x speedup on modern CPUs

#### **Memory Access Patterns**

```python
# Cache-unfriendly (column-major access)
for j in range(cols):
    for i in range(rows):
        process(data[i, j])

# Cache-friendly (row-major access)  
for i in range(rows):
    for j in range(cols):
        process(data[i, j])
```

**Impact:** Up to 10x performance difference due to cache locality

---

## 🎛️ Hyperparameter Tuning and Best Practices

### Learning Rate Strategies

#### **Learning Rate Schedules**

```python
# Step decay
def step_decay(epoch, initial_lr=0.01, drop_rate=0.5, epochs_drop=10):
    return initial_lr * (drop_rate ** (epoch // epochs_drop))

# Exponential decay
def exp_decay(epoch, initial_lr=0.01, decay_rate=0.95):
    return initial_lr * (decay_rate ** epoch)

# Cosine annealing
def cosine_decay(epoch, initial_lr=0.01, min_lr=0.001, max_epochs=100):
    return min_lr + (initial_lr - min_lr) * (1 + np.cos(np.pi * epoch / max_epochs)) / 2
```

#### **Adaptive Learning Rates**

```python
class AdamOptimizer:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0   # Time step
    
    def update(self, layer_id, weights, gradients):
        self.t += 1
        
        if layer_id not in self.m:
            self.m[layer_id] = np.zeros_like(weights)
            self.v[layer_id] = np.zeros_like(weights)
        
        # Update moments
        self.m[layer_id] = self.beta1 * self.m[layer_id] + (1 - self.beta1) * gradients
        self.v[layer_id] = self.beta2 * self.v[layer_id] + (1 - self.beta2) * gradients**2
        
        # Bias correction
        m_corrected = self.m[layer_id] / (1 - self.beta1**self.t)
        v_corrected = self.v[layer_id] / (1 - self.beta2**self.t)
        
        # Update weights
        return weights - self.lr * m_corrected / (np.sqrt(v_corrected) + self.eps)
```

### Data Augmentation Strategies

#### **Augmentation Pipeline**

```python
class DataAugmentation:
    def __init__(self):
        self.augmentations = [
            self.random_horizontal_flip,
            self.random_rotation,
            self.random_brightness,
            self.random_contrast,
            self.random_crop
        ]
    
    def random_horizontal_flip(self, x, p=0.5):
        if np.random.random() < p:
            return np.flip(x, axis=-1)  # Flip along width
        return x
    
    def random_rotation(self, x, max_angle=15):
        angle = np.random.uniform(-max_angle, max_angle)
        # Apply rotation (implementation would use scipy or similar)
        return self.rotate_image(x, angle)
    
    def apply(
