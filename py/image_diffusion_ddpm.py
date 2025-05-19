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

# %% [markdown]
# # [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NouamaneTazi/hackai-challenges/blob/main/new_notebooks/image_diffusion_ddpm.ipynb)
#
# # Image Generation with Diffusion Models
#
# In this notebook, you'll learn about diffusion models and how they can generate images! We'll use a simple implementation to generate fashion items like clothes and shoes.
#
# **What you'll learn:**
# - What are diffusion models and how they work
# - How to generate images from noise
# - How to visualize the image generation process
#
# **Time to complete:** ~1 hour
#
# **Prerequisites:** Basic Python knowledge
#
# **Note:** This notebook uses a pre-trained model to save time. You can find the training code in the comments if you want to train your own model later!

# %% [markdown]
# # Part 1: Understanding Diffusion Models
#
# Diffusion models are a type of AI that can create images by learning to remove noise from pictures. Think of it like this:
#
# 1. **Adding Noise (Forward Process):**
#    - Start with a clear image
#    - Gradually add random noise until the image becomes completely random
#
# 2. **Removing Noise (Reverse Process):**
#    - Start with random noise
#    - Learn to remove the noise step by step
#    - End up with a clear image
#
# The cool thing is that once the model learns to remove noise, it can create new images by starting with random noise and cleaning it up!
#
# TODO: Add image showing the forward and reverse process

# %% [markdown]
# # Part 2: Let's Generate Some Images!
#
# We'll use a pre-trained model to generate fashion items. First, let's set up our environment and load the model.

# %% [markdown] id="51UOj5pi6-Nd"
# <h1>! Work in progress!</h1>
#
# **To Do:**
#
# - Theoretical part explaining :
#   - Image Generation task and various family of models.
#   - Denoising Diffusion Probabilistic Models (DDPMs) (Forward and reverse processes).
# - Add description of tasks to do.
# - Hide cells that won't be used (model definition, training loop, util functions...).
# - Setup Evaluation metric.

# %% [markdown] id="8IWhk_8y3c_-"
# # ---------------------------------------------------
# # Part 1: Understanding Diffusion Models
# # ---------------------------------------------------
#
# # Theory Section: Diffusion Models
#
# 1. What are diffusion models?
#
# Diffusion models are generative models that learn to generate data by reversing a gradual noising process.
# They work by slowly adding random noise to data in a forward process, and then learning to reverse this
# process to recover the original data distribution. Unlike GANs or VAEs, diffusion models use a sequence
# of denoising steps, making them more stable to train but potentially slower to sample from.
#
# 2. Forward diffusion process:
#
# The forward diffusion process gradually adds Gaussian noise to an image over T timesteps according to a
# predefined noise schedule. At each step t, some portion of the original signal is preserved while new noise
# is added. As t increases, the image becomes more noisy until at t=T, it approaches a pure Gaussian noise
# distribution. Mathematically, this is defined as q(x_t|x_{t-1}) where each step adds a small amount of
# Gaussian noise according to variance β_t.
#
# 3. Reverse diffusion process (sampling):
#
# The reverse diffusion process (sampling) starts with a pure noise sample x_T ~ N(0,1) and gradually denoises
# it over T steps to generate a sample from the data distribution. At each step, the model predicts the noise
# component in the current noisy image, then uses this prediction to compute a slightly less noisy image for
# the next step. This process continues until we reach x_0, which should resemble a sample from the original
# data distribution. In our case, we'll get fashion items like clothing and shoes.
#
# 4. What does the neural network in DDPM actually learn to predict?
#
# The neural network in DDPM learns to predict the noise ε that was added to the image at a particular timestep t

# %% [markdown] id="lAcpQgsTPZTI"
# # ---------------------------------------------------
# # Part 2: Implementing DDPMs
# # ---------------------------------------------------
#

# %% id="pGZ998FM69lV" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1746713916598, "user_tz": -120, "elapsed": 21105, "user": {"displayName": "Yassir", "userId": "14539807087988167223"}} outputId="8a76282c-cc5b-4459-9564-e136c3849271"
# DDPM Challenge: Generate Your Own Fashion Items

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import gdown  # For downloading pre-trained model

from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torchvision.utils import make_grid

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %% [markdown]
# Let's load a pre-trained model that can generate fashion items. This model was trained on the Fashion MNIST dataset, which contains images of clothes and accessories.

# %%
# Download pre-trained model
model_url = "https://drive.google.com/uc?id=YOUR_MODEL_ID"  # TODO: Add actual model URL
model_path = "ddpm_model.pth"
gdown.download(model_url, model_path, quiet=False)

# Load the model
model = UNet().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
print("Model loaded successfully!")

# %% [markdown]
# Now let's generate some fashion items! We'll create 16 new images and display them.

# %%
def generate_samples(model, n_samples=16):
    """Generate new samples from the trained model."""
    model.eval()
    with torch.no_grad():
        samples, intermediate_steps = model.sample(n_samples, (1, 28, 28), device, return_intermediate_steps=True)
    return samples, intermediate_steps

# Generate and display samples
samples, intermediate_steps = generate_samples(model)
show_images(samples, "Generated Fashion Items")

# %% [markdown]
# Let's see how the model generates these images step by step. Use the slider below to watch the image generation process!

# %%
create_batch_slider(intermediate_steps)

# %% [markdown]
# # Part 3: How Does It Work?
#
# Let's break down what's happening:
#
# 1. The model starts with random noise (like TV static)
# 2. It looks at the noise and tries to guess what kind of fashion item it could be
# 3. It gradually removes noise and adds details until we get a clear image
#
# The slider above shows this process - you can see how the image becomes clearer as we remove more noise!
#
# TODO: Add visualization of the noise removal process

# %% [markdown]
# # Part 4: Try It Yourself!
#
# Want to generate more images? Just run the cell below with a different number of samples!

# %%
# Generate more samples
more_samples, _ = generate_samples(model, n_samples=32)
show_images(more_samples, "More Generated Fashion Items")

# %% [markdown]
# # Part 5: Quiz
#
# Test your understanding of diffusion models:
#
# 1. What happens to an image during the forward process?
#    - A) It becomes clearer
#    - B) It becomes noisier
#    - C) It stays the same
#
# 2. How does the model generate new images?
#    - A) By copying existing images
#    - B) By removing noise from random patterns
#    - C) By drawing them from scratch
#
# 3. What's the main advantage of diffusion models?
#    - A) They're very fast
#    - B) They're stable to train
#    - C) They use less memory
#
# Answers: 1-B, 2-B, 3-B

# %% [markdown]
# # Congratulations! 🎉
#
# You've learned about diffusion models and generated your own fashion items! Here's what you accomplished:
#
# - Learned how diffusion models work
# - Generated new fashion items from noise
# - Visualized the image generation process
#
# Want to learn more? Check out these resources:
# - [Denoising Diffusion Probabilistic Models Paper](https://arxiv.org/abs/2006.11239)
# - [Hugging Face Diffusion Models Course](https://huggingface.co/course/chapter1/1)

# %% [markdown] id="YxEArOTA69lb"
def create_batch_slider(intermediate_steps, labels_list=None):
    """
    Create an interactive slider widget to browse through batches of images
    using the existing show_images function.

    Parameters:
    -----------
    intermediate_steps : list
        List of tensors, where each tensor contains a batch of images
    labels_list : list, optional
        List of label tensors corresponding to each batch of images

    Returns:
    --------
    None (displays the widget in the notebook)
    """
    import ipywidgets as widgets
    from IPython.display import display
    from IPython.display import clear_output

    # Create output area for displaying images
    output = widgets.Output()

    # Create slider widget for batch selection
    batch_slider = widgets.IntSlider(
        min=0,
        max=len(intermediate_steps)-1,
        step=1,
        value=len(intermediate_steps)-1,
        description='Denoising step:',
        continuous_update=False,
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='500px')
    )

    # Function to display a batch using the existing show_images function
    def display_batch(step):
        with output:
            clear_output(wait=True)

            # Get current batch of images
            images = intermediate_steps[step]

            # Get corresponding labels if available
            labels = None
            if labels_list is not None and step < len(labels_list):
                labels = labels_list[step]

            # Call the existing show_images function
            show_images(images, title=f"Denoising Step {step + 1}/{len(intermediate_steps)}", labels=labels)

    # Update display when slider value changes
    def on_batch_change(change):
        display_batch(change['new'])

    batch_slider.observe(on_batch_change, names='value')

    # Show initial batch
    display_batch(batch_slider.value)

    # Add play button for slideshow
    play = widgets.Play(
        min=0,
        max=len(intermediate_steps)-1,
        step=10,
        value=len(intermediate_steps)-1,
        interval=1000,  # milliseconds between each frame
        description="Play",
        disabled=False
    )

    # Link play button with slider
    widgets.jslink((play, 'value'), (batch_slider, 'value'))

    # Create horizontal box with play button and slider
    controls = widgets.HBox([play, batch_slider])

    # Display controls and output
    display(widgets.VBox([controls, output]))



# %% colab={"referenced_widgets": ["3f29a1c448b94910834597307a91b58f", "025704f10bee4422b24c47fd51f2d862", "720bbb356588407a82a9a350d1f6dc1d", "f44a4dab778b4c3c8420bac45c0062fe", "a5b08d7f8e6f4d58b343fd021f1efcdf", "1b1ec0dfc0e940f58c1ae7feaccd4a82", "69695b6c9d1d43a69b127c66b87dd53f", "366814b1e94b42738a9b999d5c11a356", "c3b6abc1533a42cfbf9ba627f10d16a8", "709e4d49c8e64ffd8b9b5d2eadaacf75", "1805474e8f7a4830986b2ce3e69776f4", "308106f1c5bb4b45a3d66a34e593cc3a"], "base_uri": "https://localhost:8080/", "height": 861} id="cDYHC40569lb" outputId="2620c1c4-ae09-4644-ef3a-f6828e4502f1" executionInfo={"status": "ok", "timestamp": 1746716988057, "user_tz": -120, "elapsed": 201, "user": {"displayName": "Yassir", "userId": "14539807087988167223"}}
create_batch_slider(intermediate_steps)

# %% [markdown] id="OFxCd0Bz0IFp"
# # ---------------------------------------------------
# # Part 9: Quiz
# # ---------------------------------------------------
#
# 1. What is the primary difference between GANs and Diffusion Models?
#
# SOLUTION: GANs use a generator and discriminator that are trained adversarially, where the generator tries to produce realistic samples to fool the discriminator. Diffusion models instead gradually add noise to data and then learn to reverse this process through denoising steps. GANs are often faster for sampling but can suffer from mode collapse and training instability, while diffusion models are more stable to train but typically require many steps for sampling.
#
# 2. In the forward diffusion process, what happens to the image as t increases?
#
# SOLUTION: As t increases during the forward diffusion process, the image becomes progressively noisier. The original content of the image gradually disappears and is replaced by random Gaussian noise. At t=T (the final step), the image is effectively pure noise with almost no trace of the original image information.
#
# 3. What does the UNet model learn to predict in a DDPM?
#
# SOLUTION: The UNet model in a DDPM learns to predict the noise (ε) that was added to the image at a particular timestep. By learning to predict this noise component, the model can then remove it during the reverse diffusion process to gradually recover the clean image. This "noise prediction" approach has been shown to be more stable than directly predicting the clean image.
#
# 4. Why do we use a schedule for beta values rather than a constant value?
#
# SOLUTION: We use a schedule for beta values (typically linearly or quadratically increasing) to control the noise addition process more precisely. A gradual schedule allows the model to learn denoising at different noise levels - starting with small noise perturbations and gradually handling larger noise. This makes training more stable and allows the model to effectively learn the entire denoising process from almost-clean to very noisy images.
#
# 5. How does the sampling (reverse diffusion) process generate a new image?
#
# SOLUTION: The sampling process starts with pure Gaussian noise (x_T) and iteratively applies the trained denoising model to remove noise step by step. At each step t, the model predicts the noise component in the current noisy image, then uses this prediction to compute a slightly less noisy image for step t-1. The process continues for T steps until we reach x_0, which should be a clean image sample from the learned data distribution.
#
# 6. What's the significance of the parameter T (number of timesteps) in diffusion models?
#
# SOLUTION: The parameter T determines the number of diffusion steps in both the forward and reverse processes. A larger T means smaller noise additions at each step, making the diffusion process more gradual and potentially easier for the model to learn. However, larger T values also increase the computational cost and time required for sampling. There's a trade-off between quality (larger T) and sampling speed (smaller T).
#

# %% [markdown] id="_LSEuDgR0A2O"
# <h1>Congratulations! You've completed the DDPM Challenge!</h1>

# %% [markdown]
# # Model Definition
#
# Below is the model architecture we'll use. Don't worry about understanding all the details - this is just to show you what's happening behind the scenes!

# %%
class UNet(nn.Module):
    """A simple UNet model for denoising diffusion."""
    def __init__(self, in_channels=1, time_dim=256):
        super().__init__()
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Initial convolution
        self.conv0 = nn.Conv2d(in_channels, 64, 3, padding=1)
        
        # Downsampling
        self.conv1 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        
        # Middle
        self.conv3 = nn.Conv2d(256, 512, 3, padding=1)
        
        # Upsampling
        self.up1 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        
        # Output
        self.output = nn.Conv2d(128, in_channels, 3, padding=1)
        
    def forward(self, x, t):
        # Time embedding
        t = self.time_mlp(t.unsqueeze(-1))
        
        # Initial
        x = self.conv0(x)
        
        # Down
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        
        # Middle
        x = self.conv3(x2)
        
        # Up
        x = self.up1(x)
        x = self.up2(x)
        
        # Output
        return self.output(x)
    
    def sample(self, n_samples, shape, device, return_intermediate_steps=False):
        """Generate samples from the model."""
        model.eval()
        with torch.no_grad():
            x = torch.randn(n_samples, *shape).to(device)
            intermediate_steps = [x.cpu()]
            
            for t in reversed(range(1000)):
                t_batch = torch.ones(n_samples, device=device) * t
                predicted_noise = self(x, t_batch)
                alpha = 1 - 0.02  # Simplified noise schedule
                x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha)) * predicted_noise)
                if t % 100 == 0:
                    intermediate_steps.append(x.cpu())
            
            if return_intermediate_steps:
                return x, intermediate_steps
            return x

# %% [markdown]
# # Utility Functions
#
# These functions help us display and visualize our generated images.

# %%
def show_images(images, title=""):
    """Display a grid of images."""
    if isinstance(images, torch.Tensor):
        images = images.cpu()
    
    # Make sure images are in the range [0, 1]
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)
    
    # Create grid
    grid = make_grid(images, nrow=4, padding=2)
    
    # Display
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.title(title)
    plt.axis('off')
    plt.show()
