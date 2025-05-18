# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="_AZn4sN5KfH3"
# # Quantization Challenge: Understanding and Implementing Symmetric/Affine Quantization
#
# In this notebook, we'll explore quantization techniques for neural networks. You'll learn about:
#
# 1. What quantization is and why it's important
# 2. Different quantization types and data representations
# 3. How to implement symmetric linear quantization
# 4. Applying quantization using transformers
#
# Let's get started!

# %% id="6TfXwujeSusm"
from google.colab import drive
drive.mount('/content/drive')

# %% [markdown] id="hWkovgGzKfH4"
# ## 1. Introduction to Quantization
#
# ### What is Quantization?
#
# Quantization is a technique to reduce the computational and memory costs of running neural networks by representing weights and activations with lower-precision data types. Instead of using standard 32-bit floating point (float32), we might use 8-bit or 4-bit integers (int8 or int4) or 16-bit floating point (float16 or bfloat16).
#
# ### Benefits of Quantization
#
# - **Reduced memory usage**: Lower precision means smaller model size
# - **Lower latency**: Integer operations are faster than floating-point operations
# - **Lower power consumption**: Fewer bits mean less energy usage
# - **Deployment on edge devices**: Many embedded devices only support integer arithmetic
#
# ### Important Concepts
#
# - **Precision**: The number of bits used to represent a value
# - **Dynamic range**: The ratio between the largest and smallest representable values
# - **Quantization error**: The difference between the original value and its quantized representation
# - **Scale and zero-point**: Parameters used to map between floating-point and integer spaces

# %% [markdown] id="lADIaZNdKfH4"
# ## 2. Number Representation in Computers
#
# To understand quantization, it helps to know how computers represent numbers. Let's review different data types and their properties.

# %% id="LgIFpjC0KfH4" outputId="dccb9bef-4c70-4d97-ebdb-9e9fcc6f63f5"
# !pip3 install numpy torch matplotlib

# %% id="T4TQkN3eKfH4"
import numpy as np
import torch
import matplotlib.pyplot as plt
import struct

# Setup for displaying figures
# %matplotlib inline
plt.style.use('ggplot')

# %% [markdown] id="6Fr9sEQEKfH5"
# ### 2.1 Integer Data Types
#
# Integers can be represented as unsigned (only positive values) or signed (both positive and negative values).

# %% id="yS2S_shBKfH5" outputId="a03b0f5e-6abd-40e5-82de-e4bc3c6db5e9"
# Display the range of different integer data types
int_types = {
    'int8': (np.iinfo(np.int8).min, np.iinfo(np.int8).max, 8),
    'uint8': (np.iinfo(np.uint8).min, np.iinfo(np.uint8).max, 8),
    'int16': (np.iinfo(np.int16).min, np.iinfo(np.int16).max, 16),
    'uint16': (np.iinfo(np.uint16).min, np.iinfo(np.uint16).max, 16),
    'int32': (np.iinfo(np.int32).min, np.iinfo(np.int32).max, 32),
    'uint32': (np.iinfo(np.uint32).min, np.iinfo(np.uint32).max, 32)
}

for dtype, (min_val, max_val, bits) in int_types.items():
    print(f"{dtype}: Range [{min_val}, {max_val}], Bits: {bits}, Values: {2**bits}")

# %% [markdown] id="XtQNlTw8KfH5"
# ### 2.2 Floating-Point Data Types
#
# Floating-point numbers have a sign bit, exponent bits, and mantissa (fraction) bits:

# %% id="JzAmP2XxKfH5" outputId="4c960781-9360-4adf-f64a-33da14d0a09c"
# Display the properties of floating-point data types
float_types = {
    'float16': (np.finfo(np.float16).min, np.finfo(np.float16).max, 16),
    'float32': (np.finfo(np.float32).min, np.finfo(np.float32).max, 32),
    'float64': (np.finfo(np.float64).min, np.finfo(np.float64).max, 64)
}

for dtype, (min_val, max_val, bits) in float_types.items():
    print(f"{dtype}: Range [{min_val}, {max_val}], Bits: {bits}")

# For bfloat16 (not directly available in NumPy)
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
    x = torch.tensor([1.0, -1.0], dtype=torch.bfloat16)
    print(f"bfloat16 example values: {x}")
else:
    print("bfloat16: Range approximately [-3.4e+38, 3.4e+38], Bits: 16 (8 exponent, 7 mantissa)")


# %% [markdown] id="Ld4cbwcYKfH5"
# ### 2.3 Visual Comparison of Number Representations
#
# Let's visualize how different number formats represent values:

# %% id="MocXAlN1KfH5" outputId="073f05ad-6eaf-41d8-f6d5-49b6b9b91630"
# Create a simple visualization of the distribution of representable numbers
def visualize_number_line(dtype, num_points=1000):
    if dtype == 'int8':
        values = np.linspace(-128, 127, num_points)
        representable = np.arange(-128, 128)
    elif dtype == 'uint8':
        values = np.linspace(0, 255, num_points)
        representable = np.arange(0, 256)
    elif dtype == 'float16':
        values = np.linspace(-10, 10, num_points)
        # Generate some representable float16 values in different ranges to show distribution
        dense_near_zero = np.array([np.float16(x) for x in np.linspace(-0.1, 0.1, 40)])
        medium_range = np.array([np.float16(x) for x in np.linspace(-1, 1, 30)])
        sparse_range = np.array([np.float16(x) for x in np.linspace(-10, 10, 30)])
        representable = np.concatenate([dense_near_zero, medium_range, sparse_range])
        representable = np.unique(representable)  # Remove duplicates

    plt.figure(figsize=(10, 2))

    # For floating point, show the non-uniform distribution
    if dtype == 'float16':
        plt.scatter(representable, np.zeros_like(representable), color='blue', s=20,
                   label=f'Representable {dtype} values')
        # Add a second row of points with small positive y-value to show density
        point_density = np.ones_like(representable) * 0.1
        plt.scatter(representable, point_density, color='red', s=5, alpha=0.5,
                   label='Density visualization')
    else:
        plt.scatter(representable, np.zeros_like(representable), color='blue', s=20,
                   label=f'Representable {dtype} values')

    plt.xlim([min(values), max(values)])
    plt.title(f'Distribution of representable {dtype} values')
    plt.yticks([])
    plt.legend()
    plt.show()

# Visualize integer and floating-point distributions
visualize_number_line('int8')
visualize_number_line('uint8')
visualize_number_line('float16')


# %% [markdown] id="lukW1ujBKfH5"
# Notice how integers are evenly distributed, while floating-point numbers have higher density near zero and become sparser as the magnitude increases. This is because floating-point formats use some bits for the exponent, allowing them to represent a wider range of values but with varying precision.

# %% [markdown] id="gDFI5vbmKfH5"
# ## 3. Quantization Types
#
# Quantization is like compressing your model to make it smaller and faster. Let's explore the different ways we can do this:
#
# ### 3.1 Based on Precision (What format we're converting to)
#
# 1. Float quantization: Converting to smaller floating-point formats
#    - Float32 → Float16 (using half the bits - from 32 to 16 bits)
#    - Float32 → BFloat16 (a special 16-bit format that's better for training)
#    
#
# 2. Integer quantization: Converting floating-point numbers to integers
#    - Float32 → Int8 (using just 8 bits, or 1/4 of the original size)
#    - Float32 → Int4 (using just 4 bits, or 1/8 of the original size)
#    
#
# ### 3.2 Based on When/How We Determine the Conversion Rules
#
# 1. Post-training quantization (PTQ): Apply quantization after the model is fully trained
#    
#    This is like compressing a finished painting without changing how it was painted.
#
# 2. Quantization-aware training (QAT): Train the model while simulating quantization effects
#    
#    This is like painting while regularly checking how it would look if compressed, adjusting your technique as you go.
#
# ### 3.3 Based on How We Map Values (The Math Behind Conversion)
#
# 1. Affine quantization: `real_value = scale * (quantized_value - zero_point)`
#    - Uses two parameters: scale (S) and zero-point (Z)
#    - Zero-point ensures that the real value 0 maps exactly to an integer
#    - Like converting temperatures between Fahrenheit and Celsius (needs both multiplication and offset)
#
# 2. Symmetric quantization: `real_value = scale * quantized_value`
#    - Simpler approach using only scale (S), with zero-point fixed at 0
#    - Often maps to [-128, 127] range for int8
#    - Like converting between inches and centimeters (just multiplication)
#
# ### 3.4 Based on How Detailed the Conversion Rules Are
#
# 1. Per-tensor quantization: Use the same conversion rule for the entire tensor
#    - Simple and fast, but less precise
#
# 2. Per-channel quantization: Use different conversion rules for each channel in the tensor (or dimension)
#    - More accurate because different channels often have different value ranges
#    - Requires more storage for the extra parameters

# %% [markdown] id="d8evxUjlKfH5"
# ## 4. Linear Quantization: Theory
#
# Linear quantization maps a continuous range of values to a discrete set of quantized values. The mapping is defined by two parameters:
#
# - **Scale (S)**: The step size between adjacent representable values
# - **Zero-point (Z)**: The quantized value that represents the real value 0. This is crucial when your data isn't balanced around zero (e.g., mostly positive values). Zero-point shifts the quantization range to better match your actual data distribution, ensuring we don't waste precious bits representing values that never occur.
#
# ### Quantization Formulas
#
# **Forward quantization** (float → int):
# ```
# x_q = round(x / S + Z)
# ```
#
# **Dequantization** (int → float):
# ```
# x = S * (x_q - Z)
# ```
#
# ### Computing Scale and Zero-point
#
# For a given range [x_min, x_max] and bit-width n:
#
# **Affine quantization**:
# ```
# S = (x_max - x_min) / (2^n - 1)
# Z = round(-x_min / S)
# ```
#
# **Symmetric quantization**:
# ```
# x_abs_max = max(abs(x_min), abs(x_max))
# S = x_abs_max / (2^(n-1) - 1)  # Using n-1 to get range [-2^(n-1), 2^(n-1)-1]
# Z = 0
# ```
#
# Let's implement and visualize these concepts:

# %% id="dTJ0rgFMKfH5"
def compute_quantization_params(x_min, x_max, num_bits=8, symmetric=False):
    """Compute scale and zero-point for quantization"""
    if symmetric:
        x_abs_max = max(abs(x_min), abs(x_max))
        # Based on the formulas above, compute the scale
        # 1 line
        zero_point = 0
    else:
        scale = (x_max - x_min) / (2**num_bits - 1)
        # Based on the formulas above, compute the zeropoint
        # 1 line
        zero_point = max(0, min(2**num_bits - 1, zero_point))

    return scale, zero_point

def quantize(x, scale, zero_point, num_bits=8, symmetric=False):
    """Quantize floating-point values to integers"""
    if symmetric:
        # For symmetric, we quantize to [-2^(n-1), 2^(n-1)-1]
        qmin, qmax = -(2**(num_bits-1) - 1), 2**(num_bits-1) - 1
    else:
        # For affine, we quantize to [0, 2^n - 1]
        qmin, qmax = 0, 2**num_bits - 1

    q = np.round(x / scale + zero_point)
    return np.clip(q, qmin, qmax).astype(np.int32)

def dequantize(q, scale, zero_point):
    """Convert quantized values back to floating-point"""
    return scale * (q - zero_point)


# %% [markdown] id="3n5SylLLKfH5"
# Let's visualize how linear quantization maps floating-point values to integers:

# %% id="Xdn_eFdgKfH5" outputId="22fc3678-3dc6-4307-e252-60c98f38163e"
# Generate sample data with a normal distribution
np.random.seed(42)
data = np.random.normal(0, 5, 1000)
x_min, x_max = np.min(data), np.max(data)

# Compute quantization parameters for both methods
scale_affine, zp_affine = compute_quantization_params(x_min, x_max, num_bits=8, symmetric=False)
scale_sym, zp_sym = compute_quantization_params(x_min, x_max, num_bits=8, symmetric=True)

# Quantize and dequantize the data
q_affine = quantize(data, scale_affine, zp_affine, symmetric=False)
q_sym = quantize(data, scale_sym, zp_sym, symmetric=True)

dq_affine = dequantize(q_affine, scale_affine, zp_affine)
dq_sym = dequantize(q_sym, scale_sym, zp_sym)

# Plot the results
plt.figure(figsize=(15, 10))

# Original data distribution
plt.subplot(2, 2, 1)
plt.hist(data, bins=50, alpha=0.7)
plt.title(f'Original Data: Range [{x_min:.2f}, {x_max:.2f}]')
plt.xlabel('Value')
plt.ylabel('Frequency')

# Quantized values distribution
plt.subplot(2, 2, 2)
plt.hist(q_affine, bins=50, alpha=0.7, color='orange', label='Affine')
plt.hist(q_sym, bins=50, alpha=0.5, color='green', label='Symmetric')
plt.title('Quantized Values')
plt.xlabel('Quantized Value')
plt.ylabel('Frequency')
plt.legend()

# Dequantized values
plt.subplot(2, 2, 3)
plt.scatter(data, dq_affine, alpha=0.5, s=5, color='orange')
plt.plot([-20, 20], [-20, 20], 'k--', alpha=0.5)  # perfect reconstruction line
plt.title(f'Affine Quantization (S={scale_affine:.4f}, Z={zp_affine})')
plt.xlabel('Original Value')
plt.ylabel('Reconstructed Value')

plt.subplot(2, 2, 4)
plt.scatter(data, dq_sym, alpha=0.5, s=5, color='green')
plt.plot([-20, 20], [-20, 20], 'k--', alpha=0.5)  # perfect reconstruction line
plt.title(f'Symmetric Quantization (S={scale_sym:.4f}, Z={zp_sym})')
plt.xlabel('Original Value')
plt.ylabel('Reconstructed Value')

plt.tight_layout()
plt.show()

# Calculate quantization error
error_affine = np.abs(data - dq_affine)
error_sym = np.abs(data - dq_sym)

print(f"Affine Quantization: Mean Absolute Error = {np.mean(error_affine):.6f}")
print(f"Symmetric Quantization: Mean Absolute Error = {np.mean(error_sym):.6f}")

# %% [markdown] id="yr9666qDKfH5"
# Affine quantization error is generally lower than symmetric quantization error, and it's logical because we have more information to work with.

# %% [markdown] id="N4xxlit1KfH5"
# ## 5. Implementing Symmetric Linear Quantizer
#
# Now let's implement a simple symmetric linear quantizer function for neural network weights:

# %% id="Kzp-q26hKfH5"
import torch
import torch.nn as nn
import torch.nn.functional as F

class SymmetricQuantizer:
    """Implements symmetric linear quantization"""

    def __init__(self, num_bits=8):
        self.num_bits = num_bits
        self.qmin = -(2**(num_bits-1) - 1)
        self.qmax = 2**(num_bits-1) - 1
        self.scale = None

    def get_scale(self, x):
        """Calculate the scale factor for quantization"""
        x_abs_max = torch.max(torch.abs(x))
        scale = x_abs_max / self.qmax
        # Add a small epsilon to avoid division by zero
        scale = torch.max(scale, torch.tensor(1e-8))
        return scale

    def quantize(self, x):
        """Quantize the tensor"""
        # Based on the quantization code above implement this method
        # ~3 lines
        return x_q

    def dequantize(self, x_q):
        """Dequantize the tensor"""
        if self.scale is None:
            raise ValueError("Scale is not set. Quantize first or set scale manually.")
        return x_q * self.scale

    def quantize_dequantize(self, x):
        """Perform quantization followed by dequantization (simulates quantization effects)"""
        x_q = self.quantize(x)
        x_dq = self.dequantize(x_q)
        return x_dq

    def __repr__(self):
        return f"SymmetricQuantizer(bits={self.num_bits}, range=[{self.qmin}, {self.qmax}])"


# %% [markdown] id="rjvbhlmAKfH5"
# Let's test our quantizer on some random data:

# %% id="BbDf_YyaKfH5" outputId="d1f55252-ab80-4dcd-f850-cb8f6e4c5221"
# Generate random tensor
torch.manual_seed(42)
x = torch.randn(1000) * 5

# Test quantization with different bit widths
bit_widths = [8, 4, 2]
plt.figure(figsize=(15, 5))

for i, bits in enumerate(bit_widths):
    quantizer = SymmetricQuantizer(num_bits=bits)
    x_dq = quantizer.quantize_dequantize(x)

    plt.subplot(1, 3, i+1)
    plt.scatter(x.numpy(), x_dq.numpy(), alpha=0.5, s=5)
    plt.plot([-15, 15], [-15, 15], 'k--', alpha=0.5)  # perfect reconstruction line

    error = torch.abs(x - x_dq).mean().item()
    plt.title(f'{bits}-bit Quantization\nMAE: {error:.4f}')
    plt.xlabel('Original Value')
    plt.ylabel('Reconstructed Value')

    print(f"{bits}-bit: Scale = {quantizer.scale.item():.6f}, Mean Abs Error = {error:.6f}")

plt.tight_layout()
plt.show()

# %% [markdown] id="2TU-vN2RKfH5"
# You can see that the more bits we use, the more accurate the quantization is. In the 2-case bits for example we only have 4 values to represent the data, so the quantization is not very accurate.

# %% [markdown] id="wTdJN6A7KfH6"
# ## 6. Quantization using Transformers
#
# In transformers, we can use different quantization methods very easily using the `HFQuantizer` class under the hood, some of these methods are :
#
# - GPTQ
# - AWQ
# - BitsAndBytes
# - TorchAO
# - etc.

# %% [markdown] id="iDDXtj__KfH6"
# Let's see how we can quantize a model using `BitsAndBytes` for example, we need to install `bitsandbytes` and `accelerate`

# %% [markdown] id="aKtQpypoO6FA"
# Note : You need to have a gpu runtime for this section !

# %% colab={"base_uri": "https://localhost:8080/"} id="OXttnqMhKfH6" outputId="149dbb39-da31-4b21-de9b-07e293e4a7c9" executionInfo={"status": "ok", "timestamp": 1746831625362, "user_tz": -120, "elapsed": 116986, "user": {"displayName": "Mohamed Mekk", "userId": "08019817581514288946"}}
# !pip3 install accelerate bitsandbytes

# %% id="WwfZXAObKfH6" outputId="e3718036-0b5b-41be-892e-bc8a151c8236" colab={"base_uri": "https://localhost:8080/", "height": 461, "referenced_widgets": ["c21bc6670361490d958105f2bc54b0a5", "48a156d15f9946ce8ed7858cc57ae4f2", "4c4dd0a589924152825d645f67a7aa3d", "583dd5fc64074e889d569edc60164cc0", "acbe0808fb8840ebaba613b77cf8bb29", "d5753d0852714db98083971b488726fa", "9285925ede9e4a179d3f6227313a4e81", "0da58458e4dc43609d26d857decd3250", "b602a55c5bfd402282d1f88b2bbb115b", "f003bbea42f54f30b31fe41da355bd0c", "fad11c124c234601a492cfe9d77f9284", "f75ef345c2bb44f0bfb57d73c636ac7c", "2acc42f28eea47a7914ca83a825de593", "53db880a4cd640a0a3a75c0bc542bcdc", "f82031ef39a046eebb16b5db2b86b148", "663828fe9fad4a22a3d1182d86a33ae3", "ca6ae631f74349439e2ecc43057de588", "23888986fed24c599c88708aede7a63b", "2ab8983839c34ce6ae582a9cfe01bd8b", "ee757f4fdc824e9e9ddd4f25f09754a5", "51770e8aeb63421e9c5e29f3bcac3f98", "5054be650dbf4d948138cec4c187532a", "28524651b330401db95a8137ee9eb491", "feb952b5506741e29f2f50429c4193d3", "b2157d59cfa74a37a85f6b861fc217e3", "0384dde01f0d42daaec7f4db36d1d88e", "d141086ac0a84533a53c636dc3151b73", "d66bbcadb1bc4d6a9a2679aee90b3c83", "78410bd5e9d94a90ad7671cfcbe78069", "eb17b7ee65644c3da8f8c48d1cec8837", "c3517d996de44c2e8fca2db177882069", "636df01175a548d3bdef4c423c67d672", "b4e49749e0174f10a769e64c1d159c82", "54ed69390298417687a2116aac0695e6", "d77560599b194be5846ffcb90b2b3eda", "eace63cb30c8433e85747fd99bfaa7ce", "28e2104c4cf5402da2688064c80288cf", "0c523466587442f292d9def947e2df84", "bcf2203c52f546ddbbeebedcbdc97d3a", "defa7b0f2dcb44b9b6598f0a3f6b9496", "69375a0b4cff4da79613a45e5a36347c", "d60275f7639c40cbaa0d6a2fffe31653", "f4365ba3d43245bdbae5b374e31a0157", "4b6cfbcc6b994848baeb46dc3809668c", "3a678f758c3e49f4a0b5868068c89caa", "9c3ba30b83f844ada3c2bd02e4983819", "a740e80c61924376aa331a30040a0c48", "0b472bce52624e0682740d0c87cd5994", "80ff9a764ad84f90ad29e99557bd129d", "c3e57c7375ad4bc096af8e67caab4e0a", "a744d8eb68d143fdb305787ddfbb22a2", "fb8183a5151345ad9306f1641901b9fb", "56ddd1626df84c31bae865b48ad50227", "901e8211915f467db7fe21ed8ea61610", "1434c3ffa7b6475d80da6b154b49292c", "771a8ce9a2ab4701889dfb725d31174a", "96d8ac1d646842c18484136e3e63605d", "916eee87ae86442890ae0e076df50f7f", "f7a27c0afefd44bba2bd10912499d6ca", "1066575e702641839a4c2e618a5507d4", "51df3c2487644ee9abbf37eaf4f83902", "b3ebae862a124b0191a68506f8ec7801", "0844e4ff9e3b4f83b3e1ebed93dbca6a", "625efa2c6c1f453a830bdb4cedb40e8a", "215f1054a4a6468786d01e901f84c4fa", "3f91acd52267445984a9fb961738c83a"]} executionInfo={"status": "ok", "timestamp": 1746831725448, "user_tz": -120, "elapsed": 48040, "user": {"displayName": "Mohamed Mekk", "userId": "08019817581514288946"}}
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)

quantized_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Llama-3.2-1B",
    torch_dtype="auto",
    quantization_config=quantization_config
)

tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B")
input_text = "What are we having for dinner?"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

output = quantized_model.generate(**input_ids, max_new_tokens=10)
print(tokenizer.decode(output[0], skip_special_tokens=True))

# %% [markdown] id="cTrlK56gM0ZB"
# To load a quantized model it's even simpler, and everything is handled internally

# %% id="yCSsRBj2KfH6" colab={"base_uri": "https://localhost:8080/", "height": 318, "referenced_widgets": ["1a7246d01aa64135bb66ae3ed340897e", "733ab10689bd443b80440522665acc2f", "5d73a32d62c8469e8cd79725062acbda", "46819b67ea4f454d8346e4c386cf5af8", "7c39f399c6344080b8aa11621d9f5674", "8a7d136cd39449999fab3d2b87335e93", "598b8dec57824fc88f0edcf3fb0afd8c", "5e5a6ad651fe4fb6b9a9409092166a47", "67df1039b8ed47768bee8c903f54a302", "b5beefc6200740c6b0878d3606bb47c1", "f1c5f90160aa4cf38278b843dde074c4", "8175158826dc4eea9f95d15373d1ee91", "1292bbfcc9c148ac9028b7de74699b8a", "d19f2df79b71423d9babc70f12593244", "5887b074263745cf85d5fda8be0cd2b8", "a61f1116f78f4e27a38ddc9ee195eb30", "2c13dd89f74844f0a1d2304a9e0cfdcf", "1d923d93c4464c8eab8e60f59f056209", "6aa718d10f28414189a16533d52abb9e", "16f583a24acb4158beef0a2544a58f96", "1ff8b93736be4a22967a1a9334a2ae32", "13268bcc0772423bbf2e124c52907005", "2d4aeb4d0677429e89693d6261998b09", "cf73314a710243a5b17a35f3781104c3", "d8002b758aff47daaec4bcc892a224df", "f7445e58b92a46b8b72b32d6081bb06c", "dda241f6bc994a4eb86b91dbdd8407ce", "0e6cf1a63fcf4f95acf211f0ec3512ef", "6c163bedbd324363b643c4dac3cae480", "f953fa0fb1204abbb98ce11cdf7a1e09", "1a8c47edd2df45c7a6f8a8d1b8060146", "f31026d513064a4c9b68775606871ac1", "d7e4370e2eec4d41a7e8f7313459328e", "651c27d67ded42c98fbd6f0df2db0def", "58e8cacb627c41b092eb2f948951bd34", "0aeb41db998f421a867073e62dbd3e14", "d3886d137d8540648f309916e45db49f", "5253804ed5574ee387dac1c977ae1125", "dfffd4094a164911a4b6240c981f8a31", "7122e5f5cfca49df8e9829e7673cfa02", "372f1960ed5743d18e286f0668b78b05", "327e33042d2c4c26af938baecae47ab8", "8ca2d00baba840d0a96a946ee0255947", "008720fd059d47078674fd1d356979d9", "d0e66b47af3e496d8555416a67f1459a", "d270900ef1114d8bbcd443c9c330a77c", "7a70d168be634bfa8ef7558f727e4af7", "66a1c86f70d743ca83509ad4a7b40667", "d09531df2da1461a88eb57a391ffe7e0", "cdad632d1d3d4869948186a54fe97a5c", "c5f3673a21a74c0e8b037fc30407ee0a", "1c24f7ea491e48b1a3394a2cabc1ee1d", "68e20e47af46449c85a18a169394aa37", "357c1aaaef124dc1901bc0c88501d87e", "5c6c73354cbb4009a29d933df0cc79e0", "0b47281ce26845b3a0f7efe9c08913ff", "77ca1286f18344aabf005be68de43f33", "f95bcc3e74ef494cbd9b4b2df9accc72", "9da8e13beaa849b7a496b1e7d1f5c228", "f18fb6a58f7a4c7b99968e62be27f614", "18fc5a8d3cf6448abac1ff7ea0cbd5a5", "495f6410c0ff43898409087ee55853e0", "387e25b27ba74e97bfee94d7db057f84", "96fe47bfb6a34b0296b8a796700fad11", "6539bd448e1c44e189a4edb6287b0290", "136b8e2a338e430994141af4c43264e7"]} executionInfo={"status": "ok", "timestamp": 1746831744969, "user_tz": -120, "elapsed": 12222, "user": {"displayName": "Mohamed Mekk", "userId": "08019817581514288946"}} outputId="2a604c0e-68a9-46e2-d3d0-66df5bde559a"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


quantized_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/Llama-3.2-1B-bnb-4bit",
    torch_dtype="auto",
)

tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-bnb-4bit")
input_text = "What are we having for dinner?"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

output = quantized_model.generate(**input_ids, max_new_tokens=10)
print(tokenizer.decode(output[0], skip_special_tokens=True))

# %% [markdown] id="TmKJYT3VPf-E"
# Happy Hacking !
#
#
