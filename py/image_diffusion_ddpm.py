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

from torchvision.datasets import FashionMNIST
from torchvision import transforms
from torchvision.utils import make_grid

# Set random seed for reproducibility
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %% id="zYMTKbm569lY" outputId="6dd97d5e-ddc5-46ce-c232-c14ca044d7c1" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1746713924398, "user_tz": -120, "elapsed": 7798, "user": {"displayName": "Yassir", "userId": "14539807087988167223"}}
# ---------------------------------------------------
# Part 2: Load and Visualize the MNIST Dataset
# ---------------------------------------------------

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),                # Convert images to PyTorch tensors
    transforms.Normalize((0.5,), (1.0,))  # Normalize to [-1, 1]
])

# Load Fashion MNIST dataset
train_dataset = FashionMNIST('./data', train=True, download=True, transform=transform)

# Fashion MNIST classes
fashion_mnist_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# %% id="dJvpKLQn69lZ" outputId="e8518bc8-f064-43c9-9930-29002f136be0" colab={"base_uri": "https://localhost:8080/", "height": 829} executionInfo={"status": "ok", "timestamp": 1746713925142, "user_tz": -120, "elapsed": 735, "user": {"displayName": "Yassir", "userId": "14539807087988167223"}}

# Visualize some examples
def show_images(images, title="", labels=None):
    """Display a batch of images."""
    plt.figure(figsize=(10, 10))

    # Unnormalize the images
    images = (images + 1) / 2

    # Create a grid of images
    grid = make_grid(images[:16], nrow=4).permute(1, 2, 0)
    plt.imshow(grid.cpu().numpy(), cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
# Display some training examples
dataiter = iter(train_loader)
images, labels = next(dataiter)
show_images(images, "Fashion MNIST Training Examples", labels)


# %% id="bvg8W0RW69lZ"
# ---------------------------------------------------
# Part 3: Define the DDPM Noise Schedule
# ---------------------------------------------------

def ddpm_schedules(beta1: float, beta2: float, T: int, device: str = 'cpu') -> dict:
    """
    Returns pre-computed schedules for DDPM sampling, training process.

    Args:
        beta1: Start value for the noise schedule
        beta2: End value for the noise schedule
        T: Number of diffusion steps

    Returns:
        A dictionary of useful values for the diffusion process
    """
    # Noise schedule
    beta_t = (beta2 - beta1) * torch.arange(0, T + 1, dtype=torch.float32, device=device) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)

    # Alpha schedule
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    # Other useful quantities
    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab = (1 - alpha_t) / sqrtmab

    return {
        "beta_t": beta_t,  # β_t
        "alpha_t": alpha_t,  # α_t
        "oneover_sqrta": oneover_sqrta,  # 1/√α_t
        "sqrt_beta_t": sqrt_beta_t,  # √β_t
        "alphabar_t": alphabar_t,  # ᾱ_t
        "sqrtab": sqrtab,  # √ᾱ_t
        "sqrtmab": sqrtmab,  # √(1-ᾱ_t)
        "mab_over_sqrtmab": mab_over_sqrtmab,  # (1-α_t)/√(1-ᾱ_t)
    }


# %% id="ZioKPbH669la" outputId="ac52d459-c858-40f9-d040-c3aa9d8609ce" colab={"base_uri": "https://localhost:8080/", "height": 505} executionInfo={"status": "ok", "timestamp": 1746713925827, "user_tz": -120, "elapsed": 662, "user": {"displayName": "Yassir", "userId": "14539807087988167223"}}
# Test our implementation
test_schedules = ddpm_schedules(1e-4, 0.02, 1000, device=device)
print("Noise schedule computed successfully!")

# Plot the noise schedule to visualize it
plt.figure(figsize=(10, 5))
plt.plot(test_schedules["alphabar_t"].cpu().numpy(), label=r'$\bar{\alpha}_t$')
plt.plot(test_schedules["sqrtmab"].cpu().numpy(), label=r'$\sqrt{1-\bar{\alpha}_t}$')
plt.plot(test_schedules["beta_t"].cpu().numpy() * 100, label=r'$\beta_t \times 100$')
plt.legend()
plt.title("DDPM Noise Schedule")
plt.xlabel("Diffusion Step t")
plt.ylabel("Value")
plt.show()


# %% id="nlAz_LFf69la" outputId="a4c12392-d197-4ff8-ea95-9fa3591f7adf" colab={"base_uri": "https://localhost:8080/", "height": 408} executionInfo={"status": "ok", "timestamp": 1746713926198, "user_tz": -120, "elapsed": 372, "user": {"displayName": "Yassir", "userId": "14539807087988167223"}}
# ---------------------------------------------------
# Part 4: Implement the Forward Diffusion Process
# ---------------------------------------------------

def forward_diffusion(x0, t, schedules):
    """
    Forward diffusion process: q(x_t | x_0)
    Gradually adds noise to an image according to the noise schedule.

    Args:
        x0: Initial image (clean)
        t: Timestep(s) for which to add noise
        schedules: Dictionary of pre-computed diffusion values

    Returns:
        x_t: Noisy image at timestep t
        noise: The noise that was added
    """
    # Generate random noise
    noise = torch.randn_like(x0)

    # Add noise according to the schedule
    # x_t = √(ᾱ_t) * x_0 + √(1-ᾱ_t) * ε
    x_t = schedules["sqrtab"][t, None, None, None] * x0 + schedules["sqrtmab"][t, None, None, None] * noise

    return x_t, noise

# Test the forward diffusion process
x0 = images[:4].to(device)  # Use 4 images for testing
timesteps = [0, 50, 100, 500, 999]  # Test at different diffusion steps

plt.figure(figsize=(15, 4))
for i, t in enumerate(timesteps):
    t_tensor = torch.tensor([t], device=device).repeat(x0.shape[0])
    x_t, _ = forward_diffusion(x0, t_tensor, test_schedules)

    # Display the noisy image
    plt.subplot(1, len(timesteps), i+1)
    grid = make_grid((x_t + 1)/2, nrow=2).permute(1, 2, 0)
    plt.imshow(grid.cpu().numpy(), cmap='gray')
    plt.title(f"t = {t}")
    plt.axis('off')
plt.suptitle("Forward Diffusion Process")
plt.show()


# %% id="na8aQmM_69la" outputId="eb6340c4-109c-4961-a991-c762291d7530" colab={"base_uri": "https://localhost:8080/"} executionInfo={"status": "ok", "timestamp": 1746714365860, "user_tz": -120, "elapsed": 11, "user": {"displayName": "Yassir", "userId": "14539807087988167223"}}
# ---------------------------------------------------
# Part 5: Define the Epsilon Prediction Model
# ---------------------------------------------------

# Define a simple convolutional block
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU()
    )

# Create a simpler model for faster training
class UNet(nn.Module):
    """
    Simplified model that predicts the noise added at each step.
    """
    def __init__(self, n_channels=1):
        super(UNet, self).__init__()

        # Simple convolutional neural network
        self.conv = nn.Sequential(
            conv_block(n_channels, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 128),
            conv_block(128, 64),
            nn.Conv2d(64, n_channels, 3, padding=1),
        )

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 128),
        )

    def forward(self, x, t):
        # Process the input
        features = self.conv(x)

        # Embed time and broadcast
        t_emb = self.time_embed(t.unsqueeze(-1))
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)

        # Combine time embedding with features
        # Just adding a small influence from the time embedding
        return features + 0.1 * t_emb.mean(dim=1, keepdim=True)

# Initialize the model
model = UNet().to(device)
print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")


# %% id="Fy1yeEwy69lb"
# ---------------------------------------------------
# Part 6: Implement the DDPM Training and Sampling
# ---------------------------------------------------

class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model.
    """
    def __init__(self, eps_model, betas, n_T, criterion=nn.MSELoss()):
        super(DDPM, self).__init__()
        self.eps_model = eps_model
        self.n_T = n_T
        self.criterion = criterion

        # Register noise schedule
        for k, v in ddpm_schedules(betas[0], betas[1], n_T).items():
            self.register_buffer(k, v)

    def forward(self, x):
        """
        Training forward pass. Equivalent to one step of Algorithm 1 in the DDPM paper.
        """
        # Sample random timesteps (batch_size,)
        t = torch.randint(1, self.n_T + 1, (x.shape[0],), device=x.device)

        # Add noise to the input
        noise = torch.randn_like(x)
        x_t = self.sqrtab[t, None, None, None] * x + self.sqrtmab[t, None, None, None] * noise

        # Predict the noise
        predicted_noise = self.eps_model(x_t, t / self.n_T)

        # Calculate loss
        loss = self.criterion(noise, predicted_noise)

        return loss

    def sample(self, n_sample, size, device, return_intermediate_steps=False):
        """
        Sampling function. Equivalent to Algorithm 2 in the DDPM paper.
        """
        # Start with random noise
        x = torch.randn(n_sample, *size).to(device)
        intermediate_steps = []
        # Iteratively denoise
        for i in tqdm(range(self.n_T, 0, -1), desc="Sampling"):
            # Get the current timestep
            t_norm = i / self.n_T

            # Predict the noise
            predicted_noise = self.eps_model(x, torch.ones(n_sample, device=device) * t_norm)

            # If not the last step, add some noise (stochasticity)
            z = torch.randn_like(x) if i > 1 else 0

            # Update x using the predicted noise and schedules
            x = (
                self.oneover_sqrta[i] * (x - predicted_noise * self.mab_over_sqrtmab[i])
                + self.sqrt_beta_t[i] * z
            )
            if return_intermediate_steps:
                intermediate_steps.append(x.clone())
        if return_intermediate_steps:
            return x, intermediate_steps
        return x


# %% id="SHD89aHD69lb" colab={"base_uri": "https://localhost:8080/", "height": 1000, "referenced_widgets": ["996f0b482cb0401cbd2303d95271f00b", "089175c407594d21a0d8aa52fe528f79", "ce0234372d9e452685290afcf93a4e12", "acaea88d705e41aca6646c53655cae3b", "de53d8d3c31144f1a63a6af8173adc21", "88144e9e1c8d4733a6a799a01b36548f", "bf74eca822e24e6f8c405e230fab653d", "4999be83a8ef4224b51d3ce5be96e7c2", "282d2b4067784c14a3514d7e7621dd2e", "f224efef75454d41a04dbb4bc40a4242", "b8df623894fd4c93baf518b6dd19dc87", "7f0e6eb0ef22413b8ededa45d8c3badc", "7f33fe16932d40ed86a01a7083ac638b", "bcd1dbc76834422aa1f31fcca0d18872", "297aa76a45fa4c5bbd87b97c3a2ac011", "dbf0a57cf44543918aaf3a556453ae02", "a4682f4515394449a5d2e436a40f2833", "d74e9859dc834c099034abedcd7e9969", "36b4cfc2674c4d918b9c780ed925ef91", "924bca122a6b4ae09407147e20bd23cf", "0a46be727e1048e3bcf283beb30007b8", "b7de1830c00b4e37b8bfb399f047a621", "411a7ee7491f45289f6b5731250ca817", "6f56f4c7b14b4fedb3a685c7679be431", "98a94e456c5940ab8bb88680e9461af0", "a78910ce1f0549beb53eb8e29c06af64", "1c82dabc876a4e4ebed75b9631ab08d0", "a81b63e99e7c4efb8a76c48123cb2216", "26dccadb01ee49ab90a5c2de37e8fc24", "fac9895acbd340fcbfd5fb31cbb00e43", "bd264109962942cc9cb841f85edad3d8", "1580c217825541978f9691f5642fb978", "e0bb7a93d5624410a344461771e46390", "304188e279b0420caf5155df6a56ac91", "2454d53de8a34bf280a355391c9ebd66", "d960b9ce7a5e4f74a2e33953b0aa1c1f", "d9809bde92d344b19409d3325343b55a", "c37e4f578970483f81d0b9b8a70f2b6b", "a5f7b4f642274abc9e02fb4fcdfbe4bb", "5bfcb17a4f3c4a07b3241fef7c0b73ca", "62e33309c12e4a698c454376a75b79bc", "4fb40a9ee00a45baa0ebc0fc4a0c9f67", "31b57af514a046b396aebe40fb8e7872", "c95c871300fa4f5fb84796aeb1b951d4", "958f3a777c7f43a38f103c9f419661ef", "96d23563615b4fd69ee7a2b904b126b9", "61a887d55c614f8585599c3020c84a34", "b1e94e49edaa4c9fb61d87d1a43456d7", "a91ceaa69b1046d281a23c4547933011", "c160ae7bcdaf478699ba3d20dbcb7951", "afcfdc95d2e04f90b211e3f8293e44cf", "9bb38a04a8444910a7e0da3c2ac79419", "019282487ac741f5afaf8e8e3970e8a7", "3cb6f6fecc504b0ca423c194546e7098", "f14f0b7215b94c96858e08cf38d33705", "e5bf4592ac5946c08fa9e23a02f02a98", "ca44844a3660472c9058011b8fd450ce", "6bf2842ebc144805ae39e215f1f32cca", "0a38d565408f4931982c24beaa5b4dcb", "afbc6b20d0464eae94570279ca584f6c", "85169f686320422893e2ff398901a6b1", "f767cbe0aa5d425cb73493c798bc6235", "ed54be41b997435997bbe610f9c0bbbd", "0040ec6bc8ca485686f5810479062b52", "80aa812fc3da4061a94191bf5e7c5007", "73c9495f284b4891a3bbb9aa14dcc8cb", "f7476a85b8634860850fcd5ed384d250", "38751dcb5114494aba96f28f96b596e8", "496092cd99a64d9d81c0533136dd31b6", "d6a8915c1811472aaf26462cfc8c91b2", "d88abe8f8abf41b3a3956ad8b3611340", "382c5fd6220f407b9855d70818fbb0ef", "04493ed62bc940a2b7d43ca0b3840596", "ba1725e256594c3086511c28b4faa056", "05a99c4157fc477983ba3bf73d8d9872", "acd1121c91534f68859f79501d2656a0", "f383d645a8394cc59d7e033ee5cbfc38", "6c46a650242447e9bc8294c084aebd80", "dbd7a1b0a4af45ca90c3b80a3c1e5d94", "f21e91309ecb41d8aa4c860bc2647f06", "1dafa737acfa4a05a71ae995343ede7c", "75717eb0349a415cb9d1cff3a5f2d457", "78a4b6b8654d4d95818fca0a6d27aae1", "3cec216cf75c462298ae080729bdc2e0", "b89ae779ec8b4c0d8d32635cf7fb2748", "98efa43d62ec4845bd07d4b1223aee34", "567cfbd5375c43599d2c59af5b3b1eb2", "01c803c0a3ab470da9281ab643e81a24", "dded576b63c34c2e9bfa557eff9ab84e", "e5ab2dc36baf4839b825c2f24d63fa00", "8aba5c16cef344418b09c52a544c3cf5", "cbb7988c5bc34ee082b8584f9d47c7af", "8e79fe808bce4e57a47df9481441024c", "91ad7a99cdf040279f0b0e94e795d2bc", "774a0e87dba64814b4394ff9a494f15f", "4a8d82b89a114a5aa780ac6180c66b93", "4418cb468d6749ef824c03825bc7216c", "eefb408fda6241dd86ef4a25e1b78978", "d6430d96b6084e0c9ea089c7e214c419", "527a412745a54fb98f791d3361195275", "9a838ad9e66e4180a2a5fddb60dc8bad", "4f229b251bf44d90800dc8b80f2bb304", "ba1fc10cb1d847279c7d0df87363ac4f", "cbb7dd2d910a4159a68e84240f349926", "fe25d90bec1d498fb0c3e497a8e1a6e8", "e10c4cede31e436780332c7ae0641e3c", "ebe34cd23f6c461bad4dc21438a315e8", "73932fc0fef14f8e98c343d40efd365d", "3ac839abfd054c20afa06e0ebd8cf0b2", "c6d7095868cd4471943087457386a295", "14e93b8441404614b0ae0b8963480c0d", "3b8273272cf74a12af08a9c196cda619", "268c988c1d13439396aa2d6fe72d657c", "5da3afe297314f4a9facf46fc9aad926", "d18b459a0bcf48f28910b0181ebefb2e", "13b7cba1c29741fdb4538573a509cd32", "2438db3cf4ca49de8a3427dfee1f4068", "a129effe15d3462f8e7b6b423e4a4d86", "44f8393682fd47aba6ad73a8b565dcbd", "e20b07da016c4a11aca70571c7cee14e", "8615c8f7e9c54b85a4ebe919640d8adf", "3d51f1fe4271488a98ee05a55bdce52f", "cc68e60e51ec4c559835b714bdd991df", "bde7edf8d4f344ed8576e3601a5bf2dc", "2c98261095fc4b2a912d1a5c29d80122", "b3a2488b76e54ad2ab8d11fa8a8d776b", "152c06ef1a9f4329bbe57b057971239b", "fa6df42ea91d4533bddb1875d4b3780b", "f05467b2853c424da4e79924de2a71de", "c1856a4fb58543d689627f8ae8a9b497", "eb38c660a5344cce86125a1d2b701316", "1ab993aa2a2c4f6285471bfe7e99b96d", "5baf5cee1d584c5098c6fad499cda9bf", "0884f597abcf449da910aa4b56f16b89", "9e487e9801834f2a81612c59a83b57f1", "f8183967026e4aefb5a5ec5464c626a5", "2ac1d53a8e364875aa6f315ed8408288", "fc105aac52be4919926ff9d9604da941", "ceeedb6ff2b64001b0f24af5f438985b", "526b1f42995e4233b0627c3183e4663d", "afc501801d3c4a62b63e1808491051bc", "4f489a4f4aac44fe9fd095934adc93bc", "fec0c865990e4cb7800dfb58292dfe11", "2a752c94b5e0453cb1c643aadc8ea5a6", "6e20c3b5d1f34cddbaef3894f25be666", "89b5a13235784dae8d87c545783474c3", "bf2b69377d7e4dc78108128f058a432a", "ab0356824e7a4df39c64e740b6fff223", "8d9adb06c3f34cbc9870aa001abe040b", "c366f095951244c2a59bbac1c1a80432", "6cec20f8523c47be85148fe7e174530a", "e0a11ac24745441ba62bf18b3efac7af", "9c903434a1dd4fa5ab68eb9714dd4443", "52fe7747bc7c48ccbdb6c8380e931712", "f1ad50d88cb143e4baa8030645eeaea9", "640fde1ba4da428c8c7a7ab257f365c7", "432f0d85823c44a59e7a51ee4c01abb9", "50234651e9e443099a44fa5060031acc", "716a829d4abf40838a1f59c4bdfe14ca", "be0bce48111141bdb0ba78947fe92594", "85eb434396d34a25b3a1a01279cba156", "c7c40502e8cb44b18f489166e87a2228", "0cec169c92ea42819b5dd4155968a7ea", "5d52332eddfd4f2fa0cb3fc77a4b3d9f", "1ae906e319794dddb0f6f83b1e073699", "9db9298a33c349368b14c30f8596e5f7", "2077e548f369499c965550c6a43b3b4e", "0b81445b33534f6c90c5b6bb48623938", "e830eb5476c640b6be0523e7a9bf67d6", "ff5f83687ef445f287f9eadec78796a8", "b78173120c76402f84e48feff8e09e81", "be1ffff64f484701b0922f966128bb4b", "1b688a39d18f4ce6877fcae25d5d04e2", "a3a64916c5524f0f9a6eada2a1eb2c43", "0684c74f96cc4b4a89c1fb455ed59452", "259fd2f7810244bea4cd03f3cf7498ea", "c90b44910dab4546a4cf88177f1bcf30", "05f04dc776774a2aa49b6dc135c2da8f", "06137a4b26ed4c2b9aa9542e9ab393a7", "04ce33798d84458e9400752cbb50cc2b", "86738ebdc21f459d8667d55b0df4e88a", "6f7986cf83844ba3bde769e0da54e293", "d088c79732fe45d1ad8a809d8e4b13c3", "8733e0cd52e94599a62f22193deba60b", "7e899b20d18f4333967c391150f85adb", "7cb7a67fef8040918d5582782bb8bde6", "cb19ff6c5d844382b41bcc39d5087281", "8fc36ceb129e49528fae0d18a4a98349", "045bcee9ec5c4393a952b786d49a3bde", "a6d222de83584bf8b2e380e11ab27ab3", "e8b96c31fbc04166b5153270456a7977", "6c31a7b4d1da40aba13ac1c712fecea1", "c987a6572a424ec397cddab98f48a413", "d3b4706a7d1149119be95fe85493834a", "07a2a46a6b6444d8b7e9e067c7942aaf", "99eb9a0601114150a39af9520922d165", "491bde5c2c594f7ea0815c2deca149b4", "67f0bd2eddd748b29c43ba7ea6b803e2", "ad07f3d3890c46dfbb097f8a9560475a", "f2ebb6eacd7e4abab133641216fc21a7", "907cf451289f4edea596fea3a5633c76", "b2df1390f1934c8b93a617d26d1e9291", "498d0c23601049f1aad7eea7e89772fa", "6bf157fb0f164a518ef8da9f9bef1033", "e26984b9f08f448a9e2027949f737313", "df5e3b10333b4c6a89b57ed18a92cd7a", "e358070ee3564530a7e82e2385a72568", "791e975a88d44b2e91a4b316c59b315d", "c3acfdf1d90b400b8bd870fd3ec70a1f", "0ad555d1e65745e788b3392e67acc85b", "9b2012a100fd4056a407566a39ff86bd", "8616b291e5584a6ca856f003212fe9d9", "f262ebbe4b7940ddbb162bc804c7e044", "99daf34b856843abb588d9f399aae9d4", "d1235af06cf3442296473819265f1a3b", "62a95ea062fb4eec9378b6129837f1d1", "caf006562ece4171b2855d42194d7fca", "ea501699186644daa64fd0c3c41eb3ec", "cd2c39bcee3b46809d012d42d39e7b23", "fdc0da55be8145a98cc9d8c63330ae49", "131febc0579d400caf432ee5d03f243d", "d465ce180bbb4308afa920c5d52e9cfd", "8ba5226934f34b37bd559bdea75b091b", "d28f25c3dfb746f08d5d92e8fa8eace8", "fd45be90bfdd435b9f0081e4eb60bf0a", "d0e2a5435d7c4d6f8a01253f8e8dacc8", "a93c045cae014efb90d91a07bc2197c0", "887a55f4260c41c8b7dbf04a85ad6445", "c5c5cefe962f45529cea2dfe239bfedc", "f3421c8171f6407ba92c96155803d5b4", "9593e58a13404730a35b1822144807fd", "c08c7890b12c4550a0a4f969ed2cf58e", "10b52f85f762442d802cd8cf8b7a2dc1", "4e907899a41442fcaa5a9e8ffda034b2", "ab4d6fa79a9147339db2a0e182d6a3c7", "5531514894c54431a98043838f44453a", "af60b1d9950845bdb7626e1e923042a6", "d8cd833225cf4063857cd80793468d35", "2950c5d5039448cdb9688577623a82b9", "74131e279d1c41219d584f8399b27117", "b7605caa795d47feb214efc3d65a62cc", "0faa0bc9422c41ee985e8faaf0a893ea", "43f5acd0f67b473e9a2d8f5a6453a44f", "22c9114a86be41e18f9d3b7fc2f8ecac", "bb977b325e4247da9984c8f4bcbee6a0", "d8fc52ed86454f06ab75335dfbb165e5", "f4e2d4a293cd43c9bf45e19d6d44b63e", "addee44c96e748f09ba0b9342709ce2f", "36d9a9a79f0949e5858176ec3699b1d9", "8e5d1d9073ae4e2e8fafe37e3b498de4", "df096616f0c14721a8b6829740f34816", "3d4807b93e064ce7941da01493befa26", "0ab407b2863646d28790fa19de441f72", "1c4bacedfb414dd8b4808e872440792c", "989d744c83fb4b7ab4d53852c11cca2b", "cc1599c58885490e8acbc90401da8166", "1fe9536b8fb04b3e938ee6fee5072633", "c261868d5b8b4dfb9dd25db1985d8edf", "b646cb21ff534310b2f183c5c4cd6911", "21697d2d4ea84eba8498aaab7a1d411e", "c6213942bec7490eb2953b0db085b8a2", "b777fff862dd4901890afb15a31d7f11", "990f6b6f5e924fb79f3d34ee61692f94", "817af0cadaf74637b454ff8f623380a5", "4cfd6456e5b84159a98e14c0aea7e5e5", "5e32a26ce298424cab76b0385c5d4363", "2f26dd4ab0474f209eb7347cd8ff9680", "d35305afd7464ce6bb4c793cd994c9ad", "e8d31944a8004031b362555a89765dbd", "e6d0e26c3f414e8ba28ef4e2d6a55037", "19a9c413c11f4d11abac261e9eec57cf", "9daa55e06a014fa18d419f09790cfbc2", "ab932beb2c6e46408f0af6933a5dfc6a", "582a143f59d84fcfb46000addb48d2b4", "649ece02c01844f89b01d80ac11bd8b0", "da07d523222049ea8e5d3ea24282b47d", "4abb28af580545dfa6135348fad299b1", "32fc7a07354440d98a8c5e93e11e356b", "7e1c324562a94f9ea0fd9610b5465aae", "9db4bc5499d849319e3cfb2e34147ffc", "6176a25727b74bf6b028e32e5041cb7c", "67d94cfb249d4a6b951748663fe02bd0", "481fc9e46d1547b9aeead5d195383aad", "59c814ece48c4b4dbe9d12c5028f724c", "6c8b2ae0418a4bc89ca15d8c577b90e1", "a107b3243d9a4e9c982ae6f126bc5929", "049620536acb45dfb94d35e29c7ec0d8", "0d331fc2432a43b3a9fdc4553c8b0a68", "983ba5131c5848388c65b3597d5ea75c", "ecf0a1713d20429193b035e6b1ad5b3e", "b420967603d949baa408ec313baea8ff", "a36cc1ba3f7344ea8a90c9237756167a", "1b3ebda6526b49f38f950da350eae4e5", "bb96b737c92a4d60ae4b3b88908c21ae", "8f892206e77644ab9301cdfbad8dd32b", "2f7ed53b4a494371ae709b355df6c66b", "52bfbf91cb59481da27be3f92dd79aa5", "e1a87930d2f7417e9840f84f96dd61d9", "09baa5cfc4f64785a5b063da1f403f1c", "805945b998b44f71a13ea9ba77b2b504", "c3305b6cf37242dd9f0375c3169b80c1", "9ef7619987ef4628a97ca3418b9b194e", "f05aecee686b4fa997cdae207dd761f7", "0a0e838f3fb94877a171afdc17c00837", "b619f8373b79439ea4ddbf03fcb5a686", "2cabf387ddb94944bc256437dd614667", "9e59224f3fe9411f8a0ce342dee9c10c", "89f534e22d134e8fa642f0df40555f77", "cea9ca9bee5f45809a9032d3cbe9128f", "28455cf546a749a387d0a0e6ad6490fc", "b14cbc16946048f986eaa3fbefe013f6", "c8bf9971eb214af9857b1a30571c0ee6", "1e207a55e0a241768a0405be1f465977", "528456e8546b4dea846033462661dfaa", "832ab0a8bca9491796e2f9b03db21f4b", "c086fe5240794282b1a9f42d587a988e", "055e29e0bd7e474e88ec94841a450113", "32ab7b7471924b3d8b10ca4894ebbcf3", "ae0a07b14e7e4dca88ccee18a15568d4", "17eabfd9fe90474b9f2936b2276ae9ef", "797f819174914762b2271c899f60b73b", "be9b37f6f08e477b83fc733a62dd4bd3", "a36399e5cc1a4ce0966bd308585c0c8d", "67fa0c58e7b64520b69cf651efd01ed9", "da37b6b6d2cc4c5d9f060c4e19b2bff9", "a4f4d1f316654e98bf8cf475345bcf4f", "2f8c8388e52842f48288d4ae3e86e1f4", "0b05c53c209641a18a5dca0e8b559f30", "2d552406a33f4afdab5e4624669ecd3d", "3f3c6aa7d5a94f76be67f2ec8298cc7a", "d506406154414b17ba9396239adce45b", "a5f5ecdc4f0f4286b58901fa1e788435", "170e42d39a1a4d83bf9a68612ba6bad9", "03f5722f605843f0bea792f0c4c936a5", "78b4f67e8d8e4a5695807e1c1715d38f", "cda2ba76577446c88c8030c6ef2d2eab", "ac701ef1b8b44b13b23cf2b32692f3dc", "1d1512c85dab4bc6835b03f4116e17e6", "c8af87084e4f4a33ae7c47a7895d144c", "e0ef3e7eb5c7431e958448bb8d8fc39c", "38151911728d42538d80de9901a71f7b", "ecc179763a7a42fa9b5ba8ca20b30c4a", "864611cbb90b42d9a9544d4eca19dbe1", "f74af747124b4fdf8f1991f8cbd5f0ca", "2cf64877b0b84a5d88964c7da21c3783", "faac5500b93c474da2bfb80c3be4c16a", "ac5e4b33474f44dc8f80060a1d9ee5fc", "3668e3ad601c47e1b6e61ec63aa69123", "09e41919afda4ce2b59358f90d120a36", "175cef0ca717463580d8c999a969d486", "b53dd69112ae4f778a0de24c8c80fe40", "f0a04465c99e4b77b0b9297a3e16ae81", "aa1cd53ac9f04315a695541e868bce89", "0a9065ae41334e80aa6ec06def4d2388", "362916412ba042e496e85bf3c5f31cb3", "88a5501293e24b88a6f9bf5ad8cd195b", "259211eb8b8b4ac39ba69660d9fb0221", "eb5ccb6b572c4677be68456f249bc23e", "69e216fcf5984affbfc3e7c4a7400f92", "ea965109dc214ec1bb350b7d812a029d", "48318a4cf97347e588e4fdab291f1eb9", "24386e52bbd54ac2b214446210f72210", "8f521a01daa54ea69b95473d3dfb5be5", "13f800aca471435fb5d6cd34d6660632", "a7c0bd11b6ae42be8203e3af44ae52cb", "69f8d81b046041a9a7adda77c1c90422", "39988862315b47eba83d8c2cefce82f9", "db042d255e9e48ddbd793bcc2c806996", "bce7fb77585e4f5c8585ce3a81250dbc", "f34c71194a3743d7a27bd9b0aefffdb0", "e95dbbf3da41483c8c77ddddf7bcfac5", "65281e7195de46dcb8adb94b8a529a76", "7137302c86bc41dc8d408a7045bb7b80", "ff8195ad8fb74daba6d6d72f4fcedebe", "900b9073a25a46f8acd35ba5237e483d", "2376a2ea583647e281c995d59d4b963b", "71447ea54f2f48f8a18008a5b6874fa5", "3fbc25bdb3ea423f936ca3df4e6e055c", "8a2f354d1474466bb9bab79fb20adc47", "37a6854f8e4e453a9faa9c8188550f5b", "d6767a5be6ba4f9f9fb24aa2862a0be2", "899a596b0833466cb515384790bf07dc", "92635724345f4648b28516f5a536fa43", "d11d71a44cb2481093d6187d847eb299", "74e05b058e9c48ed8a86a638c114db0d", "785f0cae572e4b19bdaaaf404aedf4b0", "53ab418e51ed437f968237d13cef1ad2", "05de063d980c45dcbd32cdcc0b0808a1", "a555ba9cee554dcc8675313b7d45e1e8", "f2f948170a4648aa87cb806bdb4d6bdc", "a02d7cc8bfd94431a3f41283fa7d511b", "4b9583a9adee4c029c54096231bfc1ca", "76adab4095774250b24d6d161b231a2d", "6d31db96bd774cd581616d8126a8c1bb", "0f5ad089bd904849bd433ed09d1741a5", "bb10e068d52247ed85098a4be06c00e3", "d7e4233d59104627ba9c9615349e11f0", "74fdeadac17048a08d6a60e348a8d60e", "de4c2161ae6b41ada78357e287087d61", "fb74afddb4f34fdab1314fef9ed0cf15", "9c52c9ce49554c859c25d2972dbf7689", "ee8f49a2dc084dc688b8677314ca6853", "96bd17460852437c9f821ab228f868b1", "f113a07d95994261a4fde4a9e179bf21", "9951f6cf083c49098497a97faf35c5cf", "b0cd13050628424c866d7d4cfe5b02f6", "63271d22e83441b3ba69b5876da502d8", "ee851c12048e4a7b90c267b2622c1858", "d92e48ced36c46a8bd100d4b7eacb6a6", "b606a6515eec4a01b0e9ced4cafcbfda", "070277a7cc35474fbb3cfc7b73b77f19", "3f651962432542f687aeaa809c968949", "f62836788488411ebffcc15a5bf25d91", "d75614aa288d4b20a4b1cf53df5ba4ea", "e0877a8b7744480e83c477646386b3c9", "40509b3741114601a6a66e4d739c0d16", "eb59ffcd68d949f69e4e30d7a3549cf7", "a899bef1032246b482549c8ccfda662e", "c6ffe10e25b945e1a2b1a02b54af6c5c", "75faedc5b54a46a4955167f0e08d3056", "f0af4693ed83472995ec85ec1583cb3d", "79f2f610916848f5a7af0d93f138a00e", "4897ac1aa94a4eee977c3240e033bccc", "07303579f26142d28a42cbe19a8badf5", "eee1fd1b316345e4b5938c6b4671adff", "86249328942f4d348f6f4b8a840f4a33", "dc07b383e1ae48198518e6006fea8e3c", "d25ff517e1c34bbca7ce5c7fe9bb8707", "ec9f603964c548d9bff638b32b7bb7da", "74b2261c219544aa85383cb193eab573", "59fdc1d737d2454aa71b9162fe731e4b", "3a2c3971c5ac491a88ab8b1754ad00e9", "2645d48e214b4580a32a602082ac87cb", "0a76ca73670b46b3b35dfd133694ea2a", "3db477d72de644d9808b552d09a9638a", "3116bb918e024938abbacc9207203451", "98ce6965c6b54430bccb2dd3f9bd3611", "35187fe94c8d47a181fd5f0617f2e010", "7130da4981724d89aa1079f358feed15", "ea2f4f5c4f9948c49f9a754cba1ea39e", "5b2d0863feaa4ff387f7943966bc2882", "ea1d21b9451849999f6302b746fec8a4", "fc899cfd0a584889a9bd8c53a37caab2", "f354327c3cb84b95b005784ec8051387", "c1a22495354646db8191f1476cd6d596", "eb31f5bb12544cdaaa862b3704246f8e", "a0fb914a3e8b4d57829a74b0c3d23418", "3009f826a2e44e7f96e859255380646f", "6a92d662ba3347ddb828893ca6539f02", "6bbe290bf1234adcafb8e840e916580c", "cb533168b1164b0ebbb436fd58a31b6c", "4e7958c4d2bb432482df0ae23d7d88a8", "508008502dc34a1280154e2e83f2c490", "e2543297de8447fe957e9380e55157ec", "4ea8f6cbbae040b89e47d0491a5daf4a", "ac62301b55c347d69d17730b0396c092", "669ce7711d49465abd003d1bef070f33", "9dc1e1621e7d480eb5fdcb2ee6e45d05", "293ce155a602491887344be0c646eafc", "5a319b4aae1745e7a04a82b77694f529", "0b87304e82d047b79ffeb0d57ba181a5", "2e1f4c87650f485d8dbff31e263d8eb6", "9bcd7406beef48a2a1893ba3617c2d6c", "0fe13edfbc9c4a4098fcc5920c35f5ab", "20db5d8c415b49f9bc97644692c19ee9", "339a48dc17304f82a6481b89276b6c64", "a17b30c9e72c44018045554dcfbc6deb", "a3b77827746f4e118a1131f6ede97312", "d692ce4641a44b0d97b369088a0dc998", "fe4ffaa17c394879a36d4db5c3a48278", "b1c2ce5ee44f43c28b1f2b699ea95fac", "a7c1d05f68d440ba8ee6ee46eafb08b8", "ba0b03b94cf74c6097f0dbafa4544820", "24141981acf741cca140f8fa543601b9", "d8d959dfbecc439e83c7eb27e8af179e", "0f4bcd96e28a4138b3a3c444fd7dcf36", "6b894580c7ca4d899cd02e82fa89e351", "a5b2ff0b61b04c06aec4c10a3dbce888", "fabe355ba3c442e6a0ffa9eeb8e114d8", "84e56facf8e342219a4f60a8c7b87d3d", "dbbe3c8587494baf8ba36683c0b308a1", "bebc2c679c57405c9640fc4013335580", "d6d2ea49ec474b6288b5a6b9e89b5098", "233d138e016e429f87f11e3e28212e5f", "abf61baa964a4c4c98a20d8534723f27", "61b32a436b50419ea02a20293747fa63", "310673f4b7594870a5cfb87ba4678741", "8fe214325e274199acb98b3346336724", "96c182c54224418eb3589def487e7a31", "dd45819001fc4e50a6c1751eed81c871", "372edb92c2894ae28b2d370ef89f81f3", "140fc28c720c4a11b763304b851a612f", "d8c6050854a64444b087239799c533f8", "d6b509ccfc7f490c914fed9e21112f47", "6c83dcff46c44c638b78a4dc6eded39f", "540d83f44d3d4a23ac1b80620bc4e32a", "0a2e4c2409bc431285dfb77ed17e2165", "d07d9142713148238f182ae882803e68", "b5cc8641421d441898df7e7c4b4e4d9b", "96c17dfee20d4f50a3803d15a794dcd2", "82d58201930444dd9769db48c61522b7", "c6ab7557f2cd4b219b2bfd4e78687ea3", "f8b70209c88c43df81fd4c343815f956", "27bc4ae992ab4d6d91bfa65c75a38a69", "d7b1a3a4519944608f7f8bd3719ae3bf", "b3d12d4c69f4479686208d30baa0fc28", "b159dc6b5741440b9b8a98a4b443709e", "b02f6b9c759946789fa5932a74f3d1ce", "370f2008e77a4e94bc64e4fa950a6379", "c2262fbc2ac2425c88731dfc274424d9", "ee52336f7054413e83a8bebcfbfb2619", "0ee92cc0983f4f61b27a6e14263c2bb1", "297b5df6ce1e4aa7ba45d17aabdf2e73", "bc54e1ca4def4065ae1dd1d71d3fe483", "c350ddedd03c428cb9b2b0996baf5748", "03d9bcfc46474b3986d7d64c88245722", "b4b5816de59e4c0f9a47f0a722354074", "d07f778e50154b83aba0ab261c2e672f", "f3b0fddc60a24e0cb14ad4e8d1cc2c46", "352b1e25483f4bc1a7cb69b5331918f6", "efea6042de3f4186bb385f97da90ac2b", "29d89b4ab03e43699e471dbad80c70df", "a3f1365d1a884f05ba5f9590c5d643ff", "08d8b11695fe467ea789a9ce6054b050", "e9cb0de9b0bd42319dca41dbfa1f473f", "1ddfc8305c66478eb86642fd95292d36", "40a4048c9a72493b90ce8cb782eb2a9e", "ffe8fdd0eca44523ad41c885a8a957c2", "51e7c534fe8d4521b817c48134ab34e9", "b6d34ece4bc24cc5a2a5a420d754090a", "d26c798e009d429d9cc5b4cc87a334cd", "23bb503108524eda8dd57c37ce5bdbb4", "d9d689a9843d4b4cb6cf6a88fd23e7aa", "43d42df4c4204dc986ae34f36446cd62", "ed14df3362134f528fd9d64f2920dd0a", "8b2ffafab3824a3d84b97e1cadfc02fa", "51c9bdbf29f843e49a5695dcb0c41f0a", "94504e8504894b58aad4f3ecc66bb048", "5a796f46151740eba5fb3ca76a6cc550", "a8c68186c9a24ae7b7c1c6804b641a64", "a31f0b413a6f4fd49be086452f4d6335", "f5bd05a1a1f3437b87805b957ee1fc3c", "f76154eab0864201a96f3f547cf3566e", "a88e71b60a3149858b7bf4f1f8b4dee2", "040ff419dade4d1da73e90334fdc7527", "7b89f65b4af94ff586703d8c05e440f6", "13d85b3216a14fcbb758fd2daebb4ea2", "f7116a8204bf4bbb9499e9b288fb0916", "a31a62c212c846cda5eb055b7222b6ab", "eccdc98e651e4e1caa9a578d22f504b2", "0fdd082e4d2446dd82197161eedf15c3", "eec30c4be7594f9e92e194fac78424bf", "02aaedea2fd241e69956a2f8d175187a", "b83a2a380a004bd6a81f226427c7eea6", "167cd399f28d4ad2ac166e44befea570", "71f1667aa2f24ae18af89371ac514dd0", "c8a294b65341495db66b526029bc92d0", "2ad570c919724c329c459969e18d3bfe", "5080338668494643b08ad2ee98642dd2", "c572c30fa47149279e4bd75ecd4b3aae", "0f99a95775df497bb32702c117f06c9c", "87c0669b2106498d999bf17a4f202a60", "93911a094cc049808b9d7ce08c5d2e44", "8da2b5ddaee14540946c1d70fd27b7cc", "3ef4b15d5cf94fb3bdcc660ae50348dc", "7c7b9391d6894e118fa21b439ffed4af", "b7ada8a820674d89bdbebe7f34d72991", "a0ad414eda9c47a289c1e9cc084fa4c8", "eaf9411f2f374dfdb15a247d6f830f38", "0a45afcdf7254e7693394a6a89746840", "7c49c1fc46ae44a7a0edba89202f0f10", "3af34b4b50174ea8aa4ba1a27ad8e9e3", "d4196fc2c09845f5a4955f8245e04a8a", "1fc77989b4744399973b7dbb73acd23c", "9b16a44f5f724d32bbc702b224d657bd", "4e0ffa9d016a48d1827a62f3b66b1040", "cd763213d1d1451497c2983bd5121e02", "4ed0692edb2143a49bc8b966b767e676", "4c411d278ca5478293aba3d87e9c3013", "4e9ddb06d8c64987acca7067a64b68a3", "3eb0ff17374c453c8da3a2e9cf759ff7", "2dc286333cc740ac9919471bd5acb06a", "ee101043aa4541c2b7439a5355aae028", "15708de659d8444fa142b6605b78c6db", "355039b3d6bb40b2977a15347b48474d", "8174b7f744374e4a853ff545c988c795", "52ed1207cc36420382cda1af4c5d0587", "c10def52476744348e5c0dbe673b70c8", "779fadc78f7d4188a9ba9a9f75981489", "02fec954fae14abbbab8bf7cd6e50034", "3e748dde422440a88673b2397432fc98", "f2787b28df584f5bb432db861ddc1e52", "17543fee09254661b5894fa0c987a2f0", "9cbde6ec1e27468ba6f440ea78714a0d", "fcba3064cfff4d888091d2dc7a427092", "c7fba05bd93f4b3c991a19ce3f04985d", "541a5657f5b542e38b9252d32f4357c4", "7d50fe11afd8455fbd2c6323349e5320", "47a9e7184c3749cca69c01afc900cded", "904223afe1a44352922ca68ab85ced74", "5f779346883c4feb911af65dbd9a5cba", "654c94f5c8534b4997f0afe3e864550e", "a5b6f7b373de41c1a38eb0899fc40043", "c278393455a445d38902a414f73d771a", "6e3b7ef0bad54d11bb233535f73a1842", "58778180f8d847f0aaeac43c39451bca", "fe7ee49d48ba4795820ba809a034fa6e", "5165b469d99744f8ab75e3036e043983", "2db689a17bd9433bb9a33742c8f0bd9a", "8ebbb0d663dc4dafabe69bc80e7c6f94", "724db66f1b7b407d944cc6236be77bd1", "351ab0b62c6c4dd295f57a053da4a3cb", "dcbbafbead724806a87ba22408038f7a", "cf412c4b980c4a84bc8ab3c312fe7a6c", "20ce3d38974b4e319c68541ace2aa766", "de879d19114a4d29bd41658ade68c3f8", "8059192aeaad49779cb546cacf9877a9", "f3d34d435722487097db2aa628009dda", "117fdcf05052433da3083298c9ef7bd0", "30992f767711466f98401f942c96ca2e", "86be45818cb64dd8b956a35ced12a43a", "a1863bbb6a0147cc8901f98b0bc13456", "4d58d372bd98489f8ab037880e1aaa58", "92a96c9c800a4cc29d249d1b4ffcf5ae", "c2ce69b4fdc343a688d440671a60f44e", "5b314f2b85724b38bd7daed8c83d844e", "240d4ee55a81481798c85253580d20b5", "83f524f20f184198855ec90479b98b0e", "508f4c1ec4274b7b9030863020615bb9", "9fd3160778b743399a976da0259080e9", "246f9d80948a438fa10e3374bd2b7af4", "28df3d4f24d54f32acad9fc498b93102", "ac2865213f964d12ac386ce2387a1829", "4b7ca6fbdf054e59ab9c85ee8f66d2b7", "e5f703b4f8d244f4b1588baaebe19c31", "a292e603ef724bb7b5fbc65868c9385f", "6c71946e39604330a4af039528d57585", "f34156e1d179465096d06629ee478e40", "337e7d833f894d6a9cdfca0d628ed5df", "88bdb3b9af6044f2954de88115b3ea82", "c93f19f5876b429c95c0d4eb287a5362", "a139f640cf8a43b88b3ed6ddef82be14", "96fc9c3008f749a28207124347bb1c71", "b6133c6169ef426ea48b41ae65577538", "795b39b6cf4d4951ab92676b2d5c5529", "7fb46c972a9c4fbfa7153e3773106bd8", "17030fdc37c64eedb5de0a35f6dbbeaf", "694ae439555d4f168d6a8c32c4358084", "6692000f17e2412cbcfa121aa5102323", "eb78714838c649da9c00b8efed855182", "c3d776ac176c4d19b76261eb4c2d71ce", "672e1163103b41f89b0c8ca32e3ca5fc", "560bff8186ce44c6b96c44506960f0ae", "509b456382fa474b87614ad2a927544b", "cc14ecf78d32441eaf2fefd1cca9aa6d", "af790b71b83348be9d690fde8956492a", "49ac768f875049efacceb6e9919603c5", "286839d1898c4abab0a0cae547bc8bbe", "460b8ae63cc44b95a11b16fa968d0398", "6b736456513346d0ac32c9e7911c6165", "00b40ccaa8d949c88fdeb4f267593df4", "a8bee4c7185f45c9bbb531670173ba47", "bd356d9cd97c444ab6036cdb0c12a09c", "dd91baff73534595af4e22cdce1f3320", "f442f93cbe5f4ec8be154b0d2927a11b", "438751440dc24a8b8dcfb6bf7af15247", "2799578912ab465b859b8c79a473cb1c", "3640c991f92d498aa8f8374e27fa6653", "66154736fcaa496ba180973ac04bf8d1", "c111ec4bc54b44f79e498c778063d0ce", "5700a4e2fc9140edb0b453d0ded60ab8", "d7f7d15121c54bd8bc6bc33661955be8", "77232ee68005434383562fc912fc8f94", "805892aa2d8b42398934f8cc7d4cd62e", "7d22b7736b514f35ab1966d506cf3c18", "81c950b7ca244c8baaa55c3f50f5fe8e", "37035292fdbb4c0c9a421e14727896b0", "fa6f565723d946e38fcc036c257d103a", "827d72413eec4f529143a248b9bfa756", "1786c08f1b1f48e4b996e263747f6728", "57ce6bea0b6a42028be9caf18dde27f8", "f2245e0d6ad84fbfae6c51869e94d0ff", "a416c8f429684978864767354a5f30a6", "8fff47c1cdc44e19b9c55fb7bc2b1408", "8b01f6fb53c34d3eb5015375afdf2d90", "8e98b6dc18a5448cacfc22d732e7db03", "9b29a13e1cd647daa5a101257bac8c85", "8bb1beef471848e896898b91e444a9e7", "82eaa0c52c2f4676b53bc55460bd7bad", "a7de8830e9df47099134c65cfac8e1e0", "80b51dcae2de4acfb7302e5ba71e9eaa", "e568640841554665afa76f56438dc752", "4780e3ea4e854a2495d946fdad47b6f2", "cde285fdca8e47adb7d8b7dfe0a179bf", "5b83f2c366474bb885f1f108281b2810", "245dd570e8424d7b835c0a1b565d8432", "d878b7bb5d8e494b9609907995016adf", "fb14e6f977904ae4812e00e727bdfbec", "eb2d2bf7cd984e3e8b7fe698a08d84cf", "1fdb04e18a3048e5a07c57cbb0a7d6ac", "e64ffba6472846b1a6036d543a6669a4", "ed76608b3b66427195c6a97ed7583eec", "89e18726d5d542b381116feec158946d", "13fd9e38c61a4d08aef294800ec809cf", "52facb45c5244198863ccf955f55e6b4", "ca140007ca964e2dbc228b15a8b8176b", "773e29d470f049bda1c7ed72c74a0a61", "52b5e55727764f6cab53be1aa6b9db93", "816bf0599fd440b784d182bb75aea7a2", "88d92452d63c44df82ab1795d4c5e34e", "e3cdba09f2aa482daa7eb47a3ad272e6", "825b9063103c4e41b7e328ab5e0b5e78", "56648772dc654b8aaaa6ecb5edab78ab", "27517b98451d4674ad32652e1e17289e", "af9598775e5a4b56bad4e96929694f32", "09227a3ba0124b85af2daa3a21127e4b", "5bd3368693a54879ae31bb7a6f987179", "fe9131a795cd409b901135c03b0516b8", "ace312ce6adb43fba61ad9472c05dc6e", "4e88b9ac081646e09a8e374599abd6d8", "29fb16e1b61647038992da82a3344025", "50d6e0a4e1734bad90517d3f18996903", "3800c43ce6624f9b91be32d625464082", "72f25279efbe4f98a025ff5ed7b4c831", "0397ddf08dbc48889e36355edf3ee1a9", "fcf5c8d127ea48dfa3db0123c8cce3c1"]} executionInfo={"status": "ok", "timestamp": 1746716655836, "user_tz": -120, "elapsed": 2277348, "user": {"displayName": "Yassir", "userId": "14539807087988167223"}} outputId="224f3a90-282b-470f-e771-6d736f07ed3f"
# ---------------------------------------------------
# Part 7: Train and Sample from the DDPM
# ---------------------------------------------------
# Initialize the DDPM model
ddpm = DDPM(
    eps_model=model,
    betas=(1e-4, 0.02),  # Start and end values for the noise schedule
    n_T=1000,           # Number of diffusion steps
).to(device)

# Optimizer
optimizer = torch.optim.Adam(ddpm.parameters(), lr=1e-4)

# Train the model (only a few epochs for the challenge)
def train_ddpm(model, dataloader, optimizer, n_epochs=10, plot_every=-1):
    """Train the DDPM model."""
    model.train()
    for epoch in range(n_epochs):
        # Use tqdm for a progress bar
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")
        epoch_loss = 0

        for batch, _ in pbar:
            batch = batch.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            loss = model(batch)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update progress bar
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        # Print epoch loss
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss/len(dataloader):.6f}")

        # Generate samples
        if plot_every > 0 and (epoch+1) % plot_every == 0:  # Generate samples every 2 epochs
            model.eval()
            with torch.no_grad():
                samples = model.sample(16, (1, 28, 28), device)
                show_images(samples, f"Samples after Epoch {epoch+1}")
            model.train()
train_ddpm(ddpm, train_loader, optimizer, n_epochs=60, plot_every=10)


# %% colab={"referenced_widgets": ["ed1aa10bb9b947899f650f4a0286f0f8", "b8dd61ec04f8412b962b5a4c7d4dc379", "8971d5edfc124c4eb06cadd73bca53b4", "379e41cbfcea4a288798da86a8e4e7b7", "a02f21d60cc8402a85ba9cfd76671b86", "f9c5fc17c4cd495b86934c764fae90db", "8ce6eb137edc45f29016376f39a9e10b", "260386c696a44ea4b8e773531d8d43b5", "aed15c995f8f4408a9cf6b3ec37d5a63", "798c5ab9a1e24dfd830048a72dd13e66", "60bb2101ba214f1fad1b74c489fb9612"], "base_uri": "https://localhost:8080/", "height": 861} id="nhTxIDyZ69lb" outputId="188e15f8-bfc5-405b-a9f4-356fb0f711e9" executionInfo={"status": "ok", "timestamp": 1746716981924, "user_tz": -120, "elapsed": 3590, "user": {"displayName": "Yassir", "userId": "14539807087988167223"}}
# ---------------------------------------------------
# Part 8: Evaluate Your Model
# ---------------------------------------------------

def generate_samples(model, n_samples=16):
    """Generate new samples from the trained model."""
    model.eval()
    with torch.no_grad():
        samples, intermediate_steps = model.sample(n_samples, (1, 28, 28), device, return_intermediate_steps=True)
    return samples, intermediate_steps

samples, intermediate_steps = generate_samples(ddpm)
show_images(samples, "Generated Samples")


# %% id="YxEArOTA69lb"
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
