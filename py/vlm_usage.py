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

# %% [markdown] id="sGFa6jqPipRZ"
# ## 🏆 Vision Language Models
# ###📌 Description
#
# In this challenge, you will interact with pre-trained Vision-Language Models (VLMs) and explore how they perform on tasks (Visual Question Answering, OCR, etc.) that combine visual and textual inputs.  
# You’ll observe model behavior, analyze outputs, and reflect on how inputs from two modalities are fused to produce meaningful responses.
#
# We encourage you to **test the models with your own images and custom prompts**—feel free to experiment in **different languages**, especially **Darija** and **Arabic**, to better understand the model's multilingual capabilities.
#
# At the end of the challenge, feel free to **share your observations, insights, and feedback** with the mentors.
#
#

# %% [markdown] id="NAzjmCsYCCDu"
# ## Environment setup

# %% id="5IeGTYIR0KF1"
# !pip install transformers -q

# %% id="VZ7PZtWV6P4H"
# After you finish testing each VLM, we will use the following function to free up GPU memory and remove variables from the global scope
import gc
import time


def clear_memory():
    # Delete variables if they exist in the current global scope
    if "inputs" in globals():
        del globals()["inputs"]
    if "model" in globals():
        del globals()["model"]
    if "processor" in globals():
        del globals()["processor"]
    time.sleep(2)

    # Garbage collection and clearing CUDA memory
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)

    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")


# %% [markdown] id="xeYCwfNtkHp6"
# ## 1. Visual Question Answering
# Objective: Answer specific questions posed about an image/video.
#
# Input: An image/video coupled with a natural language question.
#
# Output: A concise answer in natural language.
#
# We're going to use `Qwen2-VL-2B-Instruct` for this task. (Feel free to try other models as well.)

# %% colab={"base_uri": "https://localhost:8080/", "height": 470, "referenced_widgets": ["ffb8d417e87e43e6a8d2dc667d0103e7", "3553a187de454ebdbbc7ed5a0c338510", "c3fc4685d68b4f8b8ef1fae3bf65727b", "b66c6278bd6948108291de7a0a9da036", "f2972656b170469dba622a60eba7440c", "6a58974e5c634d2b8f32cfe33475806a", "e8e147ec44f14ddbbcc52f8cf5c260ca", "e7928b7777644ebf8b2ca9ab9ee0f99f", "7a4e73a3d8d54dd283dbeab13cda19a6", "f4c2bf860072466d9a06a0c8aa77a359", "f9debbe1aa6342cbb0d8dea45cc7d4fa", "0fb1d7a5bab64408bfbab2725aae25a8", "0dd7ae13ba6e473eaccbe21e664b234d", "7d85a4b0b51d44b7b97d91ddc58322f9", "3a2f14237448413f9d99fe1013815c19", "fbc401ec61ea4994a379b3983fdffb39", "bd2722fb1b2848ec873379aedfd32bbd", "9cddb880b2a848819cbe916f4dfd400b", "1e1021ea569241b6bfa8aa6aec337588", "511fc0848eec419a95a63afefff02715", "bd176467d554444fb06f13bb385a0f53", "f8b900e0119d4dbebd27450403ec3d9a", "e97c77dd25764a60a1019663fa6dfb08", "4cfa9db4c8b44ef598dffb7b984c27df", "e3d9998705834c6d8da6d33588bc209e", "8b7f946ce3874518bd9727d9d6c90ead", "283314140d834776bdaeb5f8a5ee1102", "3c56322e4ad348369776fc9e6f44cb74", "3dcaf7dd237b4479b381c8a039132852", "04a284ffa8c549babfe7688c78149b8b", "e9ac597fd4834d428ebbd572c8e0aade", "fb33bb367a284d7ba65a84c8ded69c33", "b256856429a64ee5bb4eeb56ac5e45ad", "f186f43820f2437b830d14cc50369926", "0223753573014e4da1e0f25d8dd4ed8b", "f519d04d82a44bba991bfde18ce2eeda", "eddf13324870414396fcd06b68cb9542", "8ea47976c73146eba7548245d28d1c0b", "ce2a26fbf5924e8a9d7ae36b2881f48d", "e1d72f81505646a09d48319503ae87bf", "e78cda79c1f446cb92d647eece0eb3fb", "71f33f6f90de4adbbef4f813e7ce0718", "0fb0a6bee7604bcfac1c9caeb88a65a6", "017f3db47bdf4571a4083be7f4cdb5dc", "dad1e3476861416b9a8dda529c75ee72", "de3b3e2d59094659969510cb4cfee15d", "ed6f63e2922841c4affa5284fad94fa3", "e412cf6d5a8c494386b92971cee9f7f6", "38cecf9d5be94fd2adee864e44481fe6", "0589d890bf074341a6c12675a3faae8f", "a5ebb0ac4db8425c841c41606bf2e829", "344684bdc1b94a6f90b19b0c0711e0a9", "cafdfd822cb544c1acb9c38dd1daf2e9", "5f7ac93f09a24e519084a15f1b9c757d", "382190f2e3f248fab0a6f5a77ab6aefd", "b817d99a4b824c99b9507c64c5e115b5", "db2d1c59de7046e19850bbf5f374b23d", "b68d465e4c60422c86d035d35fdad70b", "d3f5d85ce8074f4ea40dec10055a5352", "fe97008ca50c4beeb839364403c3edd2", "4a3b9a9e87804e9caba4e6e1da3d5acc", "64e05b9ef7394e6e8cedee78daf95414", "d148f3a3064d40c5a119a4fa1f54a4b6", "14e5501d05484057af35982d6a0301c1", "64d8e7e11eef4cb5bc6c6799d6d8de24", "cbd0ae3a8e3f43d888d71f26d955c9fb", "e2869e53bb694fe4b26bc50a640f9ebe", "020e2d65347b43dfa59fee30668d027e", "72506adb3da34ae8bed7fcf61c056f3a", "d3a7531a3b504deaa57755634052c52a", "bb997ad5249a4642b643c93c4f3fb114", "a5450a18bfd848ff8a1c5dbd95345690", "f05aac77686748ed990acc9e9a6e24f6", "7c2a0fd7facb4a91a47635b02eb37ef6", "4ef2c9f23b2c4a9486fd20ef53c6edc0", "87bedb19268e434493014063a267d82d", "78cb5bef8e45413ea0316741c44f6a44", "ea9d81349ee544bd86f85393948c8fdc", "4228aae68cdf4d44bade2ce9abb769b4", "269a36142275453f9436c43fe6ee33a7", "e5012ed20ce249f5a86921a231f67cbd", "08e8c3c0e0b84b52840fb622e1b8d96b", "f9d83058ce8045c1850df49678723fa7", "47ce4d16bed24fe5b0241f00226205d1", "645b7b1f4f3c4003b3e486816107163a", "e91ee713b90e414dac9469d77c619d8f", "f8b58493fd6d41279ecbc0a128aa90e1", "a193e7453efd489b9fb3585ae73f3237", "07142f8c07964c38ba7487b2b1fc78d7", "e3d467c6d0c247a5927a890f9959ad0a", "3931b52a1f5e47e6b5d870ca5c87b6e7", "3279a8bf3dd4482e830716ed7b43a83b", "c1f5d38ab09e4067999ec61577fdad89", "7f697da03fff4f0fbbd6b8c8cf2ebe69", "38f1a2164ea941d1ab50a60ca32e5494", "45287a8e3a004099bc46cb6a16e77a5a", "b5b21d1818bd4dfaaa717ca7e3fbd371", "fcfbd0b9ab9f419499487e33953d92db", "826e1d4dc108449cb6468851fffd8240", "6bd5af71af594707b68e6281c87604bb", "8bbb7662295440cf96f9efae07f7f98b", "fe8a886b8acd42c796b166474ae28311", "67529a0393504b0faa9becfb905f2dca", "75f9e5901a22403e9eb48c8723d2a2ff", "4bd0695b66ca4d838a168a4ce58f55f2", "c128e717b0224167a12ffb4df04fbd25", "c0b2361ef0e842ac964298a6c8121d8f", "5ea33f184e8c44019497ab457d1ddaec", "49e6cdca427b49a9be84b3608e55efd2", "608630add28b471f8d73ffdcf91476db", "39e31d73f2d64ff6a9d02afc711cfd2b", "d20ed9b17fba4a26b96313f5f8069603", "2e85b870b3c3400caa4a3bf9b1dccc85", "bf36032b301a458b818b96eed808934f", "634c702e7ad545279a1f11ad68bae126", "f56d6dc1d01b49ffa1d1656e49fb8bc5", "f9aef45fd14b47ee83d06fa76de788a6", "60172503f19e42e19b5ed07099192f15", "793f028249574ddd86e93e1c131b2c39", "5b34945e754648ec997315c38015d2dd", "1e028396edf84c9a87cfed6610bb4962", "c0df82ed929f4fb0b7bb14c1034e56b3", "e53b8ea2241e416eb1b68259390655f4", "682cf3de60d54d2c937b6cb065a9090e", "1c0dd1daacae43e8b4ba69b63ac0f7fe", "4fa28c8e580d44a9bc23274bff95749d", "9a9405a5821341aa9e9b4c4a2d4314d1", "3dff9cba7c934f7c90e4219ec0a6a8d5", "8a652564e0704e58a6cb90eefb6f3f90", "9b05f6485d4f4faeb55c87b8a9d9f304", "0e4b986404714835af1db563d8947b2f", "d7bb2f7923d34af287f9049f7e73f391", "031f5201327c41f8a82349fa3da3db89", "868118b780204b24a25f79951b62c894", "0aa925a5911e411c9d7c2edc1d2a1552", "6244891b940841dfb16ca4692945f01d", "7786dfe885bb4c6495bec6892ed7a651", "516f93ebcdef4d42a1bb0e979b2c98e0", "891f07f825cc4b28a67f9960c96aaabd", "3447be7c70c74b5b9e80f1654e147da5", "21d24eaa95584580b0f585486fde9a68", "a1178e63a25e415a84c4cc13ceac9b8b", "f047c927375d4ec692bb455efb34228f"]} id="sgS1Ps3TmkkQ" executionInfo={"status": "ok", "timestamp": 1746557411006, "user_tz": -60, "elapsed": 75853, "user": {"displayName": "khadija bayoud", "userId": "02459665600863003406"}} outputId="75a23a2d-8f0a-4b2d-ec3a-3d1491b0c1c4"
from PIL import Image
import requests
import torch
from torchvision import io
from typing import Dict
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor

# First, let's load the model and its processor on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# %% [markdown] id="3KkKGxmc798C"
# #### Let's test with images

# %% colab={"base_uri": "https://localhost:8080/", "height": 637} id="m2pBfKEI1g3W" executionInfo={"status": "ok", "timestamp": 1746557503225, "user_tz": -60, "elapsed": 12198, "user": {"displayName": "khadija bayoud", "userId": "02459665600863003406"}} outputId="41fb35f9-9283-4107-d3a4-81a8bb8d9567"
# Image (Try other images)
url = "https://legarconboucher.com/img/cms/Recette/tajine-maroc.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# Prompt (Try with other questions; Darija maybe?)
text_query = "What do you see in the image?"

# Define model chat template
conversation = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
            },
            {"type": "text", "text": text_query},
        ],
    }
]


# Preprocess the inputs
text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
# Excepted output: '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'

inputs = processor(
    text=[text_prompt], images=[image], padding=True, return_tensors="pt"
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
output_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids = [
    output_ids[len(input_ids) :]
    for input_ids, output_ids in zip(inputs.input_ids, output_ids)
]
output_text = processor.batch_decode(
    generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
)

import textwrap
print(textwrap.fill(output_text[0], width=100))

image


# %% [markdown] id="hrEye1RX8HVN"
# #### Let's test with videos

# %% colab={"base_uri": "https://localhost:8080/"} id="XMob6WVk8d6I" executionInfo={"status": "ok", "timestamp": 1746475944562, "user_tz": -60, "elapsed": 2338, "user": {"displayName": "khadija bayoud", "userId": "02459665600863003406"}} outputId="8f093644-b864-4b9e-f14e-eed3fb0d4a29"
# !pip install pyav yt-dlp qwen-vl-utils

# %% id="s_IvggdFA24I"
def download_video(video_url: str):
  import yt_dlp

  download_folder = "/content/"

  ydl_opts = {
      'outtmpl': f'{download_folder}/video.mp4',
      'format': 'best',
  }

  with yt_dlp.YoutubeDL(ydl_opts) as ydl:
      ydl.download([video_url])


# %% colab={"base_uri": "https://localhost:8080/"} id="0lSzCPmuBVLP" executionInfo={"status": "ok", "timestamp": 1746475957662, "user_tz": -60, "elapsed": 8525, "user": {"displayName": "khadija bayoud", "userId": "02459665600863003406"}} outputId="4b6a16df-923d-4a69-d476-aaf5c0456677"
download_video("https://www.youtube.com/shorts/po8D2FUCtu0")

# %% colab={"base_uri": "https://localhost:8080/", "height": 70} id="K9SaNwWH8Gmd" executionInfo={"status": "ok", "timestamp": 1746475967172, "user_tz": -60, "elapsed": 6445, "user": {"displayName": "khadija bayoud", "userId": "02459665600863003406"}} outputId="ced3f8e8-87fc-42cc-f5af-37f88f9cde37"
from qwen_vl_utils import process_vision_info


# Video
video = "/content/video.mp4"
text_query = "What do you see in the video?"

# Messages containing a video and a text query
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video,
                "max_pixels": 360 * 420,
                "fps": 1.0,
            },
            {"type": "text", "text": text_query},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
output_text[0]


# %% colab={"base_uri": "https://localhost:8080/"} id="OiVD1lKF4KL0" executionInfo={"status": "ok", "timestamp": 1746476005491, "user_tz": -60, "elapsed": 8637, "user": {"displayName": "khadija bayoud", "userId": "02459665600863003406"}} outputId="b91260bb-ef77-4356-d229-9b70c990a2e4"
# Ensure that you have finished testing the VLM before calling this function.
clear_memory()

# %% [markdown] id="PAdkEfW6molR"
# ## 2. Image Captioning
#
# Objective: Generate a descriptive sentence that encapsulates the overall content of an image.
#
# Input: An image.
#
# Output: A natural language sentence describing the image.
#
# The difference between VQA and image captioning is that VQA involves answering specific questions about an image based on its content, while image captioning generates a general description of the entire image without any specific query.
#
# We're going to use `blip-image-captioning-base` for this task. (Feel free to try other models as well.)

# %% colab={"base_uri": "https://localhost:8080/", "height": 310, "referenced_widgets": ["8010d35ad5394c1782268fa00ab8dcb6", "f4ff10546ef54aa6ac657439914d0bb4", "2a5ef293359e4db5bff46dc87bd57b25", "7a43abbeb6fa4a2ea3ec7be37387682b", "e1acd63b1ff34e5e81b7e2d07c881952", "5416ccd44acd45cfa2add9dc5267a0e1", "4a6c9988ddb94e7c9b07da5f45fe775e", "c73da208ed954ad086c295e0cfe552db", "b0519257601c40d0823a28d2b012733a", "193d3aebd18d46f68182e596eb10e7b3", "e69fd3da5ae8473480b276a9f852d740", "49d718251b334dbc95f63f54b9f5c5e3", "1d6d199de32b48fd95b28bb2b91703c0", "a29c187b2e1b4bd280a02dd3fbb76661", "83953c69843b415691a68058b2bb9011", "2bb066c3745249e7a0e77141ead9a579", "fbce67d4db8a4b38ac33b2c29609f825", "68b29d91fadb40c695a303aa9378ecb9", "bfb018946e51461cb91fa6ad94b17b8c", "418d32eb9f424aaeacacafb42abc8825", "729bb7e7cd2341a7bd76914b614077d1", "d836c4f51c3347479410a65323673a61", "537da598f4b14dba900ec07b0618dba3", "bd4ea49926914399804bb940ff348d10", "e1ace0fb1bd44ad7af00290ef82df58a", "ba11c7d1365f4febb234b6adb9bd7bfe", "96df429f2cd248afb7d1a3a7bbb8aabc", "0279508e4263481eaea93ec242f4e71f", "a9a4a5a37bd845c98d087922034a58b4", "77ba8e47802f4673a3370aa25dca30c2", "b98dcf3338f04afa856256dead987e7c", "3a1ec407e8d543fcb97f3674149b624f", "48418b64eefe48f39fb688fa5059081c", "84a4f6abc1444bc1a8734359545a5ebc", "47f82374f6bd479ba49d2ff13ed0e296", "4b05c496b2374f06aece2ae9ff18a78f", "c9482b28379c466c938e2f598fd49601", "dd4950c43b0b4a03badd69d4f90dd806", "3e25c873841d419696543e86d408be2d", "670f9c84bf384dc38ee3e2202554cd15", "3ef3c35c86bd4ec6975e36dae99dbbf5", "6f391d861e5d429383e00ed67468cec2", "7009041390b24ff4999607043420e7c4", "812e7b883619471fb12d2323578e19f3", "9c68a077571b488f9123fce854991ac2", "1b85c4afa4974559a1fb180756ad2f60", "106cf860bc404313b230815335d1dfae", "456f4b0792ad4fcdb808e1d3e06c3fd6", "735933768c1a4903973ed16ba640dbc9", "43ae46b89a914dbd8aacb524b5001821", "6e2b497bed3c45b18720b50a215e35c1", "485a6b5e7ee444209907bcbc496ff1b7", "c789a78e7a8741469f8b080af6ee1efd", "b8f23a33588d4b1d8e1d30ec927c9182", "6b72d733b0cf4b8885d135b8573c80c2", "4efce111f2ee4b46be6d2c86f9438491", "b1a743e646cd4f508f23419bf8632415", "d6e3cc3dc52c40bb981e419602e45c50", "772ff7a6fad9434fa4d2d52f6c4ac0de", "5a150518cd4d40d6918a693a8ffdea52", "f4c792e8a14a4523a70fcb4cac0ab5c1", "f00b3240305548ba82204db690c3050e", "16fccf7f9bdd43d688b143ff19be317c", "640c259b43e640408b0ea15c5120661e", "2614639a92294f3e9ef3fda8b6e753a0", "9940b0ac7b3d45bd8645f30da0e90f71", "c9b87dfe1ae44ee3866d7e40672fd8d2", "648c70fc01cb46b8986af4ad33f5eb22", "a94044910f7444919ece89c58542ac07", "95b1d95a82b649b3980d8091d683c5a5", "b85a06cd12b4403e9fac715873b31835", "d67dd392354c40eeabfd07eef83a898d", "b09977a572ba47a180b17dbb580d7ba3", "af597d8db0d74b5d84e933985a637c07", "eba56054d69947d49748dd30935edf1a", "cc9aeb6b163447928773f46e85a608ab", "7ec4387e3da84785b5c803031a7476bd", "162303b1408e476990f8017d21fa1097", "ea3d20a527fa40829a48da6df0636a90", "33b66676058344d082f56a84656a004e", "d979a8cae73242389dcc13e4ab4555e2", "f5453bf0cfce44b1bd7472429fc7f6e8", "bb2b4c7a3f93407b9d7c94e8a3a64341", "75b058c1da3a4347b2d4ad4557bbafc7", "9615377a15c74850972b7ac6419e7409", "7fc96aefef874c1db790ff3261db6725", "1400ce456aff4c2594a2ba15dba2d8ad", "484df2a685604ceaa8a8ca6c3532d1f5"]} id="dxdbe_UuFlBq" executionInfo={"status": "ok", "timestamp": 1746476316897, "user_tz": -60, "elapsed": 13152, "user": {"displayName": "khadija bayoud", "userId": "02459665600863003406"}} outputId="807a155d-7dc4-4f00-870f-0a615f312d26"
import torch
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# First, let's load the model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16).to("cuda")

# %% colab={"base_uri": "https://localhost:8080/", "height": 658, "output_embedded_package_id": "1xIiOC4aJuKSCSoX4FxioP7QPonXt3tFy"} id="z5LUPYAHGatO" executionInfo={"status": "ok", "timestamp": 1746477105633, "user_tz": -60, "elapsed": 9511, "user": {"displayName": "khadija bayoud", "userId": "02459665600863003406"}} outputId="0ed9b119-e6ff-4741-8fe3-7983b8d51eae"
img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

# conditional image captioning (generates a caption based on an additional condition or query that guides the captioning process)
text = "a photography of"
inputs = processor(raw_image, text, return_tensors="pt").to("cuda", torch.float16)

out = model.generate(**inputs)
print("Conditional image captioning: ", processor.decode(out[0], skip_special_tokens=True))

# unconditional image captioning (generates a caption for an image without any specific context other than the image itself.)
inputs = processor(raw_image, return_tensors="pt").to("cuda", torch.float16)

out = model.generate(**inputs)
print("unconditional image captioning: ", processor.decode(out[0], skip_special_tokens=True))

raw_image

# %% colab={"base_uri": "https://localhost:8080/"} id="7D4tTZrWKatJ" executionInfo={"status": "ok", "timestamp": 1746477373569, "user_tz": -60, "elapsed": 8801, "user": {"displayName": "khadija bayoud", "userId": "02459665600863003406"}} outputId="13b52f5e-543f-43bf-94cc-736b152ffd1f"
clear_memory()

# %% [markdown] id="OpcMjvxRl4DV"
# ## 3. OCR (Optical Character Recognition)
# Objective: Extract textual information from images, encompassing printed, handwritten, or scene text, and convert it into machine-readable formats.
#
# Input: Scanned documents, or any visuals containing text coupled with an optional prompt to guide the model's focus, such as "Extract the invoice number" or "What is the expiration date on the ID card?"
#
# Ouptut: Machine-readable text transcribed from the image.
#
# We're going to use `Qwen2-VL-OCR-2B-Instruct` for this task. (Feel free to try other models as well.)

# %% id="DU6pq5-LKlEi"
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

# First, let's load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "prithivMLmods/Qwen2-VL-OCR-2B-Instruct", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("prithivMLmods/Qwen2-VL-OCR-2B-Instruct")

# %% colab={"base_uri": "https://localhost:8080/", "height": 894} id="-D4OhXpEL6FG" executionInfo={"status": "ok", "timestamp": 1746478907031, "user_tz": -60, "elapsed": 4504, "user": {"displayName": "khadija bayoud", "userId": "02459665600863003406"}} outputId="245395d1-4804-43c6-8f4d-abc570873b4a"
# Image
url = "https://trulysmall.com/wp-content/uploads/2023/04/Simple-Invoice-Template.png"
image = Image.open(requests.get(url, stream=True).raw)

text_query = "What is the name of the invoice' sender?"

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image,
            },
            {"type": "text", "text": text_query},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

import textwrap
print(textwrap.fill(output_text[0], width=50))

image

# %% colab={"base_uri": "https://localhost:8080/"} id="60JyzdG_Mo-X" executionInfo={"status": "ok", "timestamp": 1746478626583, "user_tz": -60, "elapsed": 8724, "user": {"displayName": "khadija bayoud", "userId": "02459665600863003406"}} outputId="7863d1a2-5970-42d3-b8fa-88819475d045"
clear_memory()

# %% [markdown] id="ExP6vLAop8pQ"
# ## Visual Grounding
# Objective: Identify and localize specific regions or objects within an image that correspond to a given natural language query.
#
# Input: An image or video containing various objects or scenes and an instruction referring to a specific object or region within the image/video.
#
# Outputs:
# - Bounding Box: Coordinates (typically in the form of [x_min, y_min, x_max, y_max]) that define the rectangular area enclosing the object or region identified by the query.
# or
# - Segmentation Mask : A pixel-wise mask delineating the exact shape of the identified object or region.
# or
# - Object Label : A classification label indicating the type of object identified (e.g., "cat," "car," "tree").
#
#
# We're going to use `microsoft/kosmos-2-patch14-224` for this task. (Feel free to try other models as well.)

# %% id="Zff7lYJOkGE9"
import requests

from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

# First, let's load the model and its processor
model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224", device_map="auto")
processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")


# %% id="zmEqF7sMxFie"
# let's define a function to run a prompt.

def run_example(prompt, image):

    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = inputs.to("cuda")
    generated_ids = model.generate(
      pixel_values=inputs["pixel_values"],
      input_ids=inputs["input_ids"],
      attention_mask=inputs["attention_mask"],
      image_embeds=None,
      image_embeds_position_mask=inputs["image_embeds_position_mask"],
      use_cache=True,
      max_new_tokens=128,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    _processed_text = processor.post_process_generation(generated_text, cleanup_and_extract=False)
    processed_text, entities = processor.post_process_generation(generated_text)

    return processed_text, entities


# %% id="Pl1vraaH0CRW"
# Let's define a function to draw the bounding boxes returned by the VLM.
# REMARK >> (IT'S A QUITE LONG FUNCTION NO NEED TO UNDERSTAND EVERY DETAIL, Haha!)

import cv2
import numpy as np
import os
import requests
import torch
import torchvision.transforms as T

from PIL import Image


def is_overlapping(rect1, rect2):
    x1, y1, x2, y2 = rect1
    x3, y3, x4, y4 = rect2
    return not (x2 < x3 or x1 > x4 or y2 < y3 or y1 > y4)


def draw_entity_boxes_on_image(image, entities, save_path=None):
    """_summary_
    Args:
        image (_type_): image or image path
        collect_entity_location (_type_): _description_
    """
    if isinstance(image, Image.Image):
        image_h = image.height
        image_w = image.width
        image = np.array(image)[:, :, [2, 1, 0]]
    elif isinstance(image, str):
        if os.path.exists(image):
            pil_img = Image.open(image).convert("RGB")
            image = np.array(pil_img)[:, :, [2, 1, 0]]
            image_h = pil_img.height
            image_w = pil_img.width
        else:
            raise ValueError(f"invaild image path, {image}")
    elif isinstance(image, torch.Tensor):
        image_tensor = image.cpu()
        reverse_norm_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])[:, None, None]
        reverse_norm_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])[:, None, None]
        image_tensor = image_tensor * reverse_norm_std + reverse_norm_mean
        pil_img = T.ToPILImage()(image_tensor)
        image_h = pil_img.height
        image_w = pil_img.width
        image = np.array(pil_img)[:, :, [2, 1, 0]]
    else:
        raise ValueError(f"invaild image format, {type(image)} for {image}")

    if len(entities) == 0:
        return image

    new_image = image.copy()
    previous_bboxes = []
    # size of text
    text_size = 1
    # thickness of text
    text_line = 1  # int(max(1 * min(image_h, image_w) / 512, 1))
    box_line = 3
    (c_width, text_height), _ = cv2.getTextSize("F", cv2.FONT_HERSHEY_COMPLEX, text_size, text_line)
    base_height = int(text_height * 0.675)
    text_offset_original = text_height - base_height
    text_spaces = 3

    for entity_name, (start, end), bboxes in entities:
        for (x1_norm, y1_norm, x2_norm, y2_norm) in bboxes:
            orig_x1, orig_y1, orig_x2, orig_y2 = int(x1_norm * image_w), int(y1_norm * image_h), int(x2_norm * image_w), int(y2_norm * image_h)
            # draw bbox
            # random color
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            new_image = cv2.rectangle(new_image, (orig_x1, orig_y1), (orig_x2, orig_y2), color, box_line)

            l_o, r_o = box_line // 2 + box_line % 2, box_line // 2 + box_line % 2 + 1

            x1 = orig_x1 - l_o
            y1 = orig_y1 - l_o

            if y1 < text_height + text_offset_original + 2 * text_spaces:
                y1 = orig_y1 + r_o + text_height + text_offset_original + 2 * text_spaces
                x1 = orig_x1 + r_o

            # add text background
            (text_width, text_height), _ = cv2.getTextSize(f"  {entity_name}", cv2.FONT_HERSHEY_COMPLEX, text_size, text_line)
            text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2 = x1, y1 - (text_height + text_offset_original + 2 * text_spaces), x1 + text_width, y1

            for prev_bbox in previous_bboxes:
                while is_overlapping((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2), prev_bbox):
                    text_bg_y1 += (text_height + text_offset_original + 2 * text_spaces)
                    text_bg_y2 += (text_height + text_offset_original + 2 * text_spaces)
                    y1 += (text_height + text_offset_original + 2 * text_spaces)

                    if text_bg_y2 >= image_h:
                        text_bg_y1 = max(0, image_h - (text_height + text_offset_original + 2 * text_spaces))
                        text_bg_y2 = image_h
                        y1 = image_h
                        break

            alpha = 0.5
            for i in range(text_bg_y1, text_bg_y2):
                for j in range(text_bg_x1, text_bg_x2):
                    if i < image_h and j < image_w:
                        if j < text_bg_x1 + 1.35 * c_width:
                            # original color
                            bg_color = color
                        else:
                            # white
                            bg_color = [255, 255, 255]
                        new_image[i, j] = (alpha * new_image[i, j] + (1 - alpha) * np.array(bg_color)).astype(np.uint8)

            cv2.putText(
                new_image, f"  {entity_name}", (x1, y1 - text_offset_original - 1 * text_spaces), cv2.FONT_HERSHEY_COMPLEX, text_size, (0, 0, 0), text_line, cv2.LINE_AA
            )
            # previous_locations.append((x1, y1))
            previous_bboxes.append((text_bg_x1, text_bg_y1, text_bg_x2, text_bg_y2))

    pil_image = Image.fromarray(new_image[:, :, [2, 1, 0]])
    if save_path:
        pil_image.save(save_path)

    return pil_image # new_image


# %% id="rQac6JdOvvaa"
# Test with other images

# url = "https://huggingface.co/microsoft/kosmos-2-patch14-224/resolve/main/snowman.png"
url = "https://cdn.nba.com/manage/2021/12/USATSI_15452777-scaled-e1639236310885-784x462.jpg"
image = Image.open(requests.get(url, stream=True).raw)

# %% id="HdT5S4kDxhxb"
# Define the prompt and call the model

prompt = "<grounding> Describe this image in detail:"
model_output, entities = run_example(prompt, image)

# %% colab={"base_uri": "https://localhost:8080/", "height": 534} id="mBfAcRwOxsMs" executionInfo={"status": "ok", "timestamp": 1746556168041, "user_tz": -60, "elapsed": 945, "user": {"displayName": "khadija bayoud", "userId": "02459665600863003406"}} outputId="6a1285bf-7f81-4fe7-91d8-04ca4108b732"
# Draw the bounding boxes on the image and print the model output
new_image = draw_entity_boxes_on_image(image, entities)
print(model_output, entities, sep="\n")
new_image

# %% [markdown] id="NyoudFBy3t_B"
# ## 🎉 Congratulations!
#
# You've reached the end of the notebook—great job! 👏  
# We hope this hands-on experience gave you a better understanding of how Vision-Language Models (VLMs) work.
#
# If you have any questions or want to share your results, don’t hesitate to reach out.
#
#
# ## 🤔 What’s Next?
#
# As you wrap up, take a moment to reflect:
#
# - **What VLM tasks impressed you the most?** answering questions, Captioning, grounding, …?
# - **If you were to fine-tune a VLM**, what task would you pick? What data would you need?
# - **What real-world problem could you solve** by combining vision and language?
#
#
# **Good luck with the rest of the challenge—and HAVE FUN! 🚀**

# %% id="zbPMXzrj2Fko"
