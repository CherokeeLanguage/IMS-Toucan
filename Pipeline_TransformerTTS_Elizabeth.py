"""
Train an autoregressive Transformer TTS model on the English single speaker dataset Elizabeth
"""
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import random
import warnings

import torch

from TransformerTTS.TransformerTTS import Transformer
from TransformerTTS.TransformerTTSDataset import TransformerTTSDataset
from TransformerTTS.transformer_tts_train_loop import train_loop
from Utility.path_to_transcript_dicts import build_path_to_transcript_dict_elizabeth

warnings.filterwarnings("ignore")

torch.manual_seed(13)
random.seed(13)

if __name__ == '__main__':
    print("Preparing")
    cache_dir = os.path.join("Corpora", "Elizabeth")
    save_dir = os.path.join("Models", "TransformerTTS_Elizabeth")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path_to_transcript_dict = build_path_to_transcript_dict_elizabeth()

    train_set = TransformerTTSDataset(path_to_transcript_dict,
                                      train=True,
                                      cache_dir=cache_dir,
                                      lang="en",
                                      min_len_in_seconds=1,
                                      max_len_in_seconds=10,
                                      rebuild_cache=False)
    valid_set = TransformerTTSDataset(path_to_transcript_dict,
                                      train=False,
                                      cache_dir=cache_dir,
                                      lang="en",
                                      min_len_in_seconds=1,
                                      max_len_in_seconds=10,
                                      rebuild_cache=False)

    model = Transformer(idim=133, odim=80, spk_embed_dim=None)

    print("Training model")
    train_loop(net=model,
               train_dataset=train_set,
               valid_dataset=valid_set,
               device=torch.device("cuda"),
               save_directory=save_dir,
               steps=400000,
               batch_size=64,
               gradient_accumulation=1,
               epochs_per_save=10,
               use_speaker_embedding=False,
               lang="en",
               lr=0.05,
               warmup_steps=8000)
