import os
import random

import torch

from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeech2 import FastSpeech2
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.fastspeech2_train_loop_ctc import train_loop
from Utility.path_to_transcript_dicts import build_path_to_transcript_dict_nancy as build_path_to_transcript_dict


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume):
    if gpu_id == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")

    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
        device = torch.device("cuda")

    torch.manual_seed(131714)
    random.seed(131714)
    torch.random.manual_seed(131714)

    print("Preparing")
    cache_dir = os.path.join("Corpora", "Nancy")
    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join("Models", "FastSpeech2_Nancy_CTC")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    path_to_transcript_dict = build_path_to_transcript_dict()
    save_dir_aligner = save_dir + "/aligner"
    os.makedirs(save_dir_aligner, exist_ok=True)

    """
    if not os.path.exists(os.path.join(cache_dir, "fast_train_cache.pt")):
        print("Training aligner")
        train_aligner(train_dataset=AlignerDataset(path_to_transcript_dict,
                                                   cache_dir=cache_dir,
                                                   lang="en"),
                      device=device,
                      save_directory=os.path.join(save_dir, "aligner"),
                      steps=10000,
                      batch_size=32,
                      path_to_checkpoint="Models/Aligner/aligner.pt",
                      fine_tune=True,
                      debug_img_path=save_dir_aligner,
                      resume=resume)

    acoustic_checkpoint_path = os.path.join(save_dir, "aligner", "aligner.pt")

    print("Preparing Dataset")
    train_set = FastSpeechDataset(path_to_transcript_dict,
                                  cache_dir=cache_dir,
                                  acoustic_checkpoint_path=acoustic_checkpoint_path,
                                  lang="en",
                                  device=device)
    """

    model = FastSpeech2()

    train_sents = list(path_to_transcript_dict.keys())

    print("Training model")
    train_loop(net=model,
               train_sentences=train_sents,
               device=device,
               save_directory=save_dir,
               aligner_checkpoint=os.path.join("Models", "Aligner", "aligner.pt"),
               steps=500000,
               batch_size=32,
               lang="en",
               lr=0.0001,
               warmup_steps=4000,
               path_to_checkpoint=resume_checkpoint,
               fine_tune=finetune,
               resume=resume)
