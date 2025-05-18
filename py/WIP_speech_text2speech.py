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
#     language: python
#     name: python3
# ---

# %% papermill={"duration": 120.229083, "end_time": "2025-05-17T20:27:51.695002", "exception": false, "start_time": "2025-05-17T20:25:51.465919", "status": "completed"}
## Install Coqui TTS
# ! pip install -U pip
# ! pip install coqui-tts==0.26.0

# %% [markdown] papermill={"duration": 0.024381, "end_time": "2025-05-17T20:27:51.744954", "exception": false, "start_time": "2025-05-17T20:27:51.720573", "status": "completed"}
# # Imports

# %% papermill={"duration": 47.363405, "end_time": "2025-05-17T20:28:39.132212", "exception": false, "start_time": "2025-05-17T20:27:51.768807", "status": "completed"}
import os

import torch
from trainer import Trainer, TrainerArgs

from TTS.bin.compute_embeddings import compute_embeddings
from TTS.config import load_config
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs, VitsAudioConfig
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.managers import save_file
from tqdm import tqdm
import json
import gdown
import tarfile

torch.set_num_threads(24)

# %% papermill={"duration": 0.03819, "end_time": "2025-05-17T20:28:39.365356", "exception": false, "start_time": "2025-05-17T20:28:39.327166", "status": "completed"}
OUT_PATH = "/kaggle/working/yourtts_tamazight"
LANG_NAME = "tamazight"
ISO = "zgh"

# Name of the run for the Trainer
RUN_NAME = f"YourTTS-{LANG_NAME.capitalize()}"

# Create output directory if it doesn't exist
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

# %% papermill={"duration": 0.07443, "end_time": "2025-05-17T20:28:39.469957", "exception": false, "start_time": "2025-05-17T20:28:39.395527", "status": "completed"}
# This paramter is useful to debug, it skips the training epochs and just do the evaluation  and produce the test sentences
SKIP_TRAIN_EPOCH = False

# Set here the batch size to be used in training and evaluation
BATCH_SIZE = 40

# Note: If you add new datasets, please make sure that the dataset sampling rate and this parameter are matching, otherwise resample your audios
SAMPLE_RATE = 22050

# Max audio length in seconds to be used in training
MAX_AUDIO_LEN_IN_SECONDS = 15
# Min audio length in seconds to be used in training
MIN_AUDIO_LEN_IN_SECONDS = 0.8

# %% papermill={"duration": 126.931338, "end_time": "2025-05-17T20:30:46.425791", "exception": false, "start_time": "2025-05-17T20:28:39.494453", "status": "completed"}
# If you want to do transfer learning and speedup your training you can set here the path to the CML-TTS available checkpoint that can be downloaded here:  https://drive.google.com/u/2/uc?id=1yDCSJ1pFZQTHhL09GMbOrdjcPULApa0p
RESTORE_PATH = os.path.join(OUT_PATH, "checkpoints_yourtts_cml_tts_dataset/best_model.pth")

URL = "https://drive.google.com/u/2/uc?id=1yDCSJ1pFZQTHhL09GMbOrdjcPULApa0p"
OUTPUT_CHECKPOINTS_FILEPATH = os.path.join(OUT_PATH, "checkpoints_yourtts_cml_tts_dataset.tar.bz")

# Download the CML-TTS checkpoint if it does not exist
if not os.path.exists(RESTORE_PATH):
    print(f"Downloading the CML-TTS checkpoint from {URL}")
    gdown.download(url=URL, output=OUTPUT_CHECKPOINTS_FILEPATH, quiet=False, fuzzy=True)
    with tarfile.open(OUTPUT_CHECKPOINTS_FILEPATH, "r:bz2") as tar:
        tar.extractall(OUT_PATH)
else:
    print(f"Checkpoint already exists at {RESTORE_PATH}")

# %% [markdown] papermill={"duration": 0.033911, "end_time": "2025-05-17T20:30:46.495696", "exception": false, "start_time": "2025-05-17T20:30:46.461785", "status": "completed"}
# # Prepare dataset

# %% papermill={"duration": 193.596359, "end_time": "2025-05-17T20:34:00.125933", "exception": false, "start_time": "2025-05-17T20:30:46.529574", "status": "completed"}
from datasets import load_dataset, Audio

ds = load_dataset("HackAI-2025/tts_dataset_30h", split="train")

# %% papermill={"duration": 0.048904, "end_time": "2025-05-17T20:34:00.212143", "exception": false, "start_time": "2025-05-17T20:34:00.163239", "status": "completed"}
ds[0]

# %% papermill={"duration": 0.064889, "end_time": "2025-05-17T20:34:00.312468", "exception": false, "start_time": "2025-05-17T20:34:00.247579", "status": "completed"}
# Listen to a random sample
import IPython
import random

random_index = random.randint(0, len(ds))
sample = ds[random_index]
print("Transcription: ", sample['text'])
print("Duration: ", sample['duration'])
IPython.display.Audio(sample['audio']['array'], rate=sample['audio']['sampling_rate'])

# %% papermill={"duration": 0.053667, "end_time": "2025-05-17T20:34:00.404203", "exception": false, "start_time": "2025-05-17T20:34:00.350536", "status": "completed"}
# Get statistics about the duration of audio files
durations = ds['duration']
print("Average duration: ", sum(durations) / len(durations))
print("Max duration: ", max(durations))
print("Min duration: ", min(durations))
print(f"Total duration in hours: {sum(durations) / 3600:.2}")

# %% papermill={"duration": 0.195712, "end_time": "2025-05-17T20:34:00.638401", "exception": false, "start_time": "2025-05-17T20:34:00.442689", "status": "completed"}
# Get character count frequencies
from collections import Counter

def get_char_counts(ds):
    all_text = ds['text']
    char_counts = Counter(''.join(all_text))
    # Sort
    char_counts = dict(sorted(char_counts.items(), key=lambda item: item[1], reverse=True))
    return char_counts

char_counts = get_char_counts(ds)
char_counts

# %% papermill={"duration": 0.045126, "end_time": "2025-05-17T20:34:00.723892", "exception": false, "start_time": "2025-05-17T20:34:00.678766", "status": "completed"}
PUNCT = set(' !,.:?')

CHARS = char_counts.keys()
CHARS = set(CHARS)

# Remove punctuation from character set
CHARS = CHARS - PUNCT
CHARS = sorted(list(CHARS))
PUNCT = sorted(list(PUNCT))

print("Punctuation: ", PUNCT)
print("Character set: ", CHARS)

# %% papermill={"duration": 0.043965, "end_time": "2025-05-17T20:34:00.805772", "exception": false, "start_time": "2025-05-17T20:34:00.761807", "status": "completed"}
# Create character config
from TTS.tts.configs.shared_configs import CharactersConfig

characters = CharactersConfig(
    characters_class="TTS.tts.models.vits.VitsCharacters",
    pad="_",
    eos="&",
    bos="*",
    blank=None,
    characters="".join(CHARS),
    punctuations="".join(PUNCT),
)

# %% papermill={"duration": 182.160534, "end_time": "2025-05-17T20:37:03.013445", "exception": false, "start_time": "2025-05-17T20:34:00.852911", "status": "completed"}
import pandas as pd
import librosa
import soundfile as sf
import os
from tqdm import tqdm

# Create the output directory if it doesn't exist
output_dir = "/kaggle/temp/zgh_tts_dataset"
wavs_dir = os.path.join(output_dir, "wavs")
os.makedirs(wavs_dir, exist_ok=True)

# Create the metadata file
metadata_file = os.path.join(output_dir, "metadata.csv")
metadata = []

# Iterate over the dataset
for i, sample in tqdm(enumerate(ds), total=len(ds)):
    # Get the audio data and text
    audio_array = sample['audio']['array']
    text = sample['text']

    # Create a unique file name
    file_name_no_ext = f"{i:05d}"
    file_name = f"{file_name_no_ext}.wav"
    file_path = os.path.join(wavs_dir, file_name)

    # Save the audio file at 22050 Hz
    sf.write(file_path, audio_array, 22050)

    # Append to the metadata
    metadata.append(f"wavs/{file_name}||{text}|yan|")

# Write the metadata to the csv file
with open(metadata_file, 'w') as f:
  for line in metadata:
    f.write(line + '\n')


# %% papermill={"duration": 0.293181, "end_time": "2025-05-17T20:37:03.404646", "exception": false, "start_time": "2025-05-17T20:37:03.111465", "status": "completed"}
# !ls -lh $output_dir

# %% papermill={"duration": 0.28247, "end_time": "2025-05-17T20:37:03.785931", "exception": false, "start_time": "2025-05-17T20:37:03.503461", "status": "completed"}
# !head $output_dir/metadata.csv

# %% papermill={"duration": 0.103963, "end_time": "2025-05-17T20:37:04.040071", "exception": false, "start_time": "2025-05-17T20:37:03.936108", "status": "completed"}
dataset_conf = BaseDatasetConfig(
    formatter="brspeech", meta_file_train="metadata.csv", path=output_dir, dataset_name="zgh_tts_dataset", language=ISO
)

# %% papermill={"duration": 369.427756, "end_time": "2025-05-17T20:43:13.565832", "exception": false, "start_time": "2025-05-17T20:37:04.138076", "status": "completed"}
### Extract speaker embeddings
SPEAKER_ENCODER_CHECKPOINT_PATH = (
    "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/model_se.pth.tar"
)
SPEAKER_ENCODER_CONFIG_PATH = "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/config_se.json"

D_VECTOR_FILES = []  # List of speaker embeddings/d-vectors to be used during the training

# Checks if the speakers embeddings are already computated, if not compute it
embeddings_file = os.path.join(dataset_conf.path, "speakers.pth")
if not os.path.isfile(embeddings_file):
    print(f">>> Computing the speaker embeddings for the {dataset_conf.dataset_name} dataset")
    compute_embeddings(
        SPEAKER_ENCODER_CHECKPOINT_PATH,
        SPEAKER_ENCODER_CONFIG_PATH,
        embeddings_file,
        formatter_name=dataset_conf.formatter,
        dataset_name=dataset_conf.dataset_name,
        dataset_path=dataset_conf.path,
        meta_file_train=dataset_conf.meta_file_train,
        meta_file_val=dataset_conf.meta_file_val,
    )
D_VECTOR_FILES.append(embeddings_file)

# %% papermill={"duration": 0.229047, "end_time": "2025-05-17T20:43:14.072500", "exception": false, "start_time": "2025-05-17T20:43:13.843453", "status": "completed"}
# Audio config used in training.
audio_config = VitsAudioConfig(
    sample_rate=SAMPLE_RATE,
    hop_length=256,
    win_length=1024,
    fft_size=1024,
    mel_fmin=0.0,
    mel_fmax=None,
    num_mels=80,
)

# %% papermill={"duration": 0.226236, "end_time": "2025-05-17T20:43:14.521071", "exception": false, "start_time": "2025-05-17T20:43:14.294835", "status": "completed"}
# Init VITSArgs setting the arguments that are needed for the YourTTS model
model_args = VitsArgs(
    spec_segment_size=62,
    hidden_channels=192,
    hidden_channels_ffn_text_encoder=768,
    num_heads_text_encoder=2,
    num_layers_text_encoder=10,
    kernel_size_text_encoder=3,
    dropout_p_text_encoder=0.1,
    d_vector_file=D_VECTOR_FILES,
    use_d_vector_file=True,
    d_vector_dim=512,
    speaker_encoder_model_path=SPEAKER_ENCODER_CHECKPOINT_PATH,
    speaker_encoder_config_path=SPEAKER_ENCODER_CONFIG_PATH,
    resblock_type_decoder="2",  # In the paper, we accidentally trained the YourTTS using ResNet blocks type 2, if you like you can use the ResNet blocks type 1 like the VITS model
    # Useful parameters to enable the Speaker Consistency Loss (SCL) described in the paper
    use_speaker_encoder_as_loss=False,
    # Useful parameters to enable multilingual training
    use_language_embedding=True,
    embedded_language_dim=4,
)

# %% papermill={"duration": 0.22602, "end_time": "2025-05-17T20:43:14.996870", "exception": false, "start_time": "2025-05-17T20:43:14.770850", "status": "completed"}
TEST_SENTENCES = [
    ["ⴰⵣⵓⵍ ⴰⵢⵜⵎⴰ ⴷ ⵉⵙⵙⵜⵎⴰ!", "yan", None, "zgh"],
]

# %% papermill={"duration": 0.233569, "end_time": "2025-05-17T20:43:15.501363", "exception": false, "start_time": "2025-05-17T20:43:15.267794", "status": "completed"}
# General training config, here you can change the batch size and others useful parameters
config = VitsConfig(
    epochs=25,
    output_path=OUT_PATH,
    model_args=model_args,
    run_name=RUN_NAME,
    project_name="YourTTS",
    run_description=f"""
            - YourTTS trained using the {LANG_NAME.capitalize()} dataset.
        """,
    dashboard_logger="tensorboard",
    logger_uri=None,
    audio=audio_config,
    batch_size=BATCH_SIZE,
    batch_group_size=4,
    eval_batch_size=BATCH_SIZE,
    num_loader_workers=8,
    # eval_split_max_size=256,
    print_step=25,
    plot_step=50,
    # log_model_step=1000,
    save_step=1000,
    save_n_checkpoints=5,
    save_checkpoints=True,
    target_loss="loss_1",
    print_eval=True,
    compute_input_seq_cache=True,
    add_blank=True,
    text_cleaner="no_cleaners",
    characters=characters,
    phoneme_cache_path=None,
    precompute_num_workers=12,
    start_by_longest=True,
    datasets=[dataset_conf],
    cudnn_benchmark=False,
    min_audio_len=int(SAMPLE_RATE * MIN_AUDIO_LEN_IN_SECONDS),
    max_audio_len=SAMPLE_RATE * MAX_AUDIO_LEN_IN_SECONDS,
    mixed_precision=True,
    test_sentences=TEST_SENTENCES,
    # Enable the weighted sampler
    # use_weighted_sampler=True,
    # Ensures that all speakers are seen in the training batch equally no matter how many samples each speaker has
    # weighted_sampler_attrs={"language": 1.0, "speaker_name": 1.0},
    # weighted_sampler_attrs={"language": 1.0},
    # weighted_sampler_multipliers={
    #     # "speaker_name": {
    #     # you can force the batching scheme to give a higher weight to a certain speaker and then this speaker will appears more frequently on the batch.
    #     # It will speedup the speaker adaptation process. Considering the CML train dataset and "new_speaker" as the speaker name of the speaker that you want to adapt.
    #     # The line above will make the balancer consider the "new_speaker" as 106 speakers so 1/4 of the number of speakers present on CML dataset.
    #     # 'new_speaker': 106, # (CML tot. train speaker)/4 = (424/4) = 106
    #     # }
    # },
    # It defines the Speaker Consistency Loss (SCL) α to 9 like the YourTTS paper
    speaker_encoder_loss_alpha=9.0,
)

# %% papermill={"duration": 0.372324, "end_time": "2025-05-17T20:43:16.094068", "exception": false, "start_time": "2025-05-17T20:43:15.721744", "status": "completed"}
# Load all the datasets samples and split traning and evaluation sets
train_samples, eval_samples = load_tts_samples(
    config.datasets,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)
print(f"Loaded {len(train_samples)} train samples")
print(f"Loaded {len(eval_samples)} eval samples")

# %% [markdown] papermill={"duration": 0.219303, "end_time": "2025-05-17T20:43:16.586457", "exception": false, "start_time": "2025-05-17T20:43:16.367154", "status": "completed"}
# # Train the model

# %% papermill={"duration": 15.291496, "end_time": "2025-05-17T20:43:32.100264", "exception": false, "start_time": "2025-05-17T20:43:16.808768", "status": "completed"}
# Init the model
model = Vits.init_from_config(config)

# %% papermill={"duration": 35874.749619, "end_time": "2025-05-18T06:41:27.075025", "exception": false, "start_time": "2025-05-17T20:43:32.325406", "status": "completed"}
# Init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(restore_path=RESTORE_PATH, skip_train_epoch=SKIP_TRAIN_EPOCH),
    config,
    output_path=OUT_PATH,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)
trainer.fit()

# %% papermill={"duration": 0.315977, "end_time": "2025-05-18T06:41:27.701666", "exception": false, "start_time": "2025-05-18T06:41:27.385689", "status": "completed"}
import glob, os
paths = sorted([f for f in glob.glob(OUT_PATH+"/YourTTS*")])
#ckpts = sorted([f for f in glob.glob(OUT_PATH+"/YourTTS*/best_model.pth")])
#configs = sorted([f for f in glob.glob(OUT_PATH+"/YourTTS*/config.json")])
paths

# %% papermill={"duration": 0.313593, "end_time": "2025-05-18T06:41:28.325933", "exception": false, "start_time": "2025-05-18T06:41:28.012340", "status": "completed"}
path = paths[-1]
test_ckpt = os.path.join(path, "best_model.pth")
test_config = os.path.join(path, "config.json")

# %% papermill={"duration": 52.155062, "end_time": "2025-05-18T06:42:20.789078", "exception": false, "start_time": "2025-05-18T06:41:28.634016", "status": "completed"}
# !tts --text "ⴰⵣⵓⵍ ⴰⵢⵜⵎⴰ ⴷ ⵉⵙⵙⵜⵎⴰ!" \
#       --model_path $test_ckpt \
#       --config_path $test_config \
#       --speaker_idx yan \
#       --out_path out.wav

# %% papermill={"duration": 0.433326, "end_time": "2025-05-18T06:42:21.541520", "exception": false, "start_time": "2025-05-18T06:42:21.108194", "status": "completed"}
import IPython
IPython.display.Audio("out.wav")

# %% papermill={"duration": 6.391965, "end_time": "2025-05-18T06:42:28.249849", "exception": false, "start_time": "2025-05-18T06:42:21.857884", "status": "completed"}
# Load the extension and start TensorBoard
# %load_ext tensorboard
# #%reload_ext tensorboard
# %tensorboard --logdir $OUT_PATH

# %% [markdown] papermill={"duration": 0.306777, "end_time": "2025-05-18T06:42:28.873516", "exception": false, "start_time": "2025-05-18T06:42:28.566739", "status": "completed"}
# !curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc \
#   | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null \
#   && echo "deb https://ngrok-agent.s3.amazonaws.com buster main" \
#   | sudo tee /etc/apt/sources.list.d/ngrok.list \
#   && sudo apt update \
#   && sudo apt install ngrok

# %% papermill={"duration": 0.317063, "end_time": "2025-05-18T06:42:29.501734", "exception": false, "start_time": "2025-05-18T06:42:29.184671", "status": "completed"}
# #!ngrok config add-authtoken 2wxfy0o2SaHiJQpWcTKawpLZ5jJ_4az7tDkNynTsk5LXe3nqv

# %% papermill={"duration": 0.316241, "end_time": "2025-05-18T06:42:30.129633", "exception": false, "start_time": "2025-05-18T06:42:29.813392", "status": "completed"}
# #!ngrok http 6006
