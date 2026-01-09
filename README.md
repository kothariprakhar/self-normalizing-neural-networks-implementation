# Self-Normalizing Neural Networks

Deep Learning has revolutionized vision via convolutional neural networks (CNNs) and natural language processing via recurrent neural networks (RNNs). However, success stories of Deep Learning with standard feed-forward neural networks (FNNs) are rare. FNNs that perform well are typically shallow and, therefore cannot exploit many levels of abstract representations. We introduce self-normalizing neural networks (SNNs) to enable high-level abstract representations. While batch normalization requires explicit normalization, neuron activations of SNNs automatically converge towards zero mean and unit variance. The activation function of SNNs are "scaled exponential linear units" (SELUs), which induce self-normalizing properties. Using the Banach fixed-point theorem, we prove that activations close to zero mean and unit variance that are propagated through many network layers will converge towards zero mean and unit variance -- even under the presence of noise and perturbations. This convergence property of SNNs allows to (1) train deep networks with many layers, (2) employ strong regularization, and (3) to make learning highly robust. Furthermore, for activations not close to unit variance, we prove an upper and lower bound on the variance, thus, vanishing and exploding gradients are impossible. We compared SNNs on (a) 121 tasks from the UCI machine learning repository, on (b) drug discovery benchmarks, and on (c) astronomy tasks with standard FNNs and other machine learning methods such as random forests and support vector machines. SNNs significantly outperformed all competing FNN methods at 121 UCI tasks, outperformed all competing methods at the Tox21 dataset, and set a new record at an astronomy data set. The winning SNN architectures are often very deep. Implementations are available at: github.com/bioinf-jku/SNNs.

## Implementation Details

# Deep Dive: Self-Normalizing Neural Networks (SNNs)

This implementation demonstrates the core innovation of the SNN paper: the ability to train **very deep** Feed-Forward Neural Networks (FNNs) without Batch Normalization, by ensuring neuron activations automatically converge to a stable distribution (mean 0, variance 1).

### 1. The Core Problem: Variance Shift

In standard deep FNNs (like the `StandardReLU` baseline in the code), as data propagates through layers:
1.  **Mean Shift:** The mean activation can drift away from 0.
2.  **Variance Perturbation:** The variance can either vanish (drop to 0) or explode (go to infinity).

When variance vanishes, neurons die (stop learning). When it explodes, gradients become unstable. Batch Normalization (BN) fixes this by manually forcing $(\mu=0, \sigma=1)$ at every step. However, BN is computationally expensive and depends on batch statistics.

### 2. The Solution: SELU + LeCun Init + Alpha Dropout

The SNN acts as a **Banach Fixed-Point**. The paper proves mathematically that if you use the specific SELU activation function and specific weight initialization, the mapping from one layer to the next attracts the variance towards 1 and the mean towards 0.

#### A. Scaled Exponential Linear Units (SELU)
Used in line: `self.act = nn.SELU()`

The function is defined as:
$$
f(x) = \lambda \begin{cases} x & \text{if } x > 0 \\ \alpha e^x - \alpha & \text{if } x \le 0 \end{cases}
$$

The magic numbers (derived in the paper's appendix) are:
*   $\alpha \approx 1.6733$
*   $\lambda \approx 1.0507$

The code relies on PyTorch's optimized implementation, but these constants are what ensure the slope is slightly greater than 1 for positive inputs (expanding variance) and the negative saturation dampens variance, creating a stable equilibrium.

#### B. LeCun Normal Initialization
Implemented in `SNN.reset_parameters`:
```python
nn.init.normal_(m.weight, mean=0, std=math.sqrt(1.0 / m.in_features))
```
Standard initialization (like He/Kaiming) is designed for ReLU. SNNs require specific Gaussian initialization where the standard deviation is $\sqrt{1/N_{in}}$. If you use He init with SELU, the "Self-Normalizing" property breaks immediately because the initial variance is too high or low for the fixed-point attractor to catch it.

#### C. Alpha Dropout
Used in line: `self.dropout = nn.AlphaDropout(p=dropout_rate)`

Standard Dropout randomly sets inputs to 0. If inputs are normalized to mean 0, setting them to 0 changes the mean (and variance). **Alpha Dropout** instead:
1.  Sets values to $\alpha'$ (the negative saturation limit of SELU) instead of 0.
2.  Affine transforms the entire vector to restore the original mean and variance.

This allows strong regularization without breaking the self-normalization property.

### 3. Code & Dataset Strategy

*   **Dataset:** We used **FashionMNIST**. It is more complex than digit MNIST but still tabular-friendly (28x28 grayscale vectors). It serves as a good proxy for the UCI benchmarks mentioned in the paper.
*   **Depth:** We configured the networks with `DEPTH = 20`. For a simple FNN without residual connections or Batch Norm, 20 layers is notoriously difficult to train using ReLU. 
*   **Instrumentation:** We verified the paper's claims using **Forward Hooks** (`get_activation_hook`). This intercepts the signal at the middle of the deep network (layer 10/20) during validation.

### 4. Interpretation of Results

When you run the code, observe the generated plots:
1.  **Loss:** The SNN usually converges faster and to a lower loss than the deep ReLU network (which may struggle to learn at all due to vanishing gradients).
2.  **Activation Stats:**
    *   **SNN:** The mean stays near 0 and Std Dev near 1 (the dashed black lines). This confirms the fixed-point attractor theory.
    *   **ReLU:** The mean often drifts positive, and variance often collapses or fluctuates wildly, leading to poor signal propagation.

This implementation proves that with the correct math (SELU) and initialization, FNNs can compete with more complex architectures on structured data tasks.

## Verification & Testing

The code provides a solid implementation of Self-Normalizing Neural Networks (SNNs) following the key principles of the Klambauer et al. paper. 

**Strengths:**
1. **Architecture components**: Correctly implements the trinity of SNNs: `nn.SELU` activation, `nn.AlphaDropout`, and `LeCun Normal` initialization (`std=sqrt(1/fan_in)`). 
2. **Structure**: The modular design allows for easy depth scaling.

**Minor Observations / Suggestions:**
1. **Input Normalization**: SNNs are sensitive to input statistics. The paper recommends inputs with mean 0 and variance 1. The code uses `transforms.Normalize((0.5,), (0.5,))`, mapping inputs to $[-1, 1]$. While centering is correct, the variance of uniform $[-1, 1]$ data is $1/3$, not 1. Using dataset-calculated standardization (z-score) is theoretically better for SNN initialization assumptions, though the network often adapts.
2. **Hook Placement**: The hook registers on the linear layer `model.layers[i]`. In the `forward` loop, the order is `Linear -> Act -> Dropout`. The hook captures the **pre-activation** (linear output). While stable pre-activations imply stable signals, strictly verifying the SNN "fixed point" (mean 0, var 1) is usually done on the **post-activation** values.
3. **Hardcoded Visualization**: The slicing logic in the plotting section (`[i:i+10]`) assumes a specific number of batches in the validation set. This might break if the batch size or dataset size changes significantly.