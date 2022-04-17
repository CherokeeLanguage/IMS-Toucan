import os
import random
from typing import Tuple

import torch
from torch.utils.data import ConcatDataset

from TrainingInterfaces.Text_to_Spectrogram.AutoAligner.autoaligner_train_loop import train_loop as train_aligner
from Utility.corpus_preparation import prepare_aligner_corpus
from Utility.path_to_transcript_dicts import *


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
    langs: List[str] = ["de", "en", "fr", "nl", "ru", "chr"]
    source_base: str = "/mount/resources/speech/corpora"
    # Non Cherokee before Cherokee to get better quality voice weights as the default for the model
    sources: List[str] = ["other-audio-data", "cherokee-audio-data", "cherokee-audio-data-private"]
    datasets = list()

    for source in sources:
        for lang in langs:
            toucan_file = os.path.join(source_base, source, f"ims-toucan-{lang}.txt")
            corpus_dir = os.path.join("Corpora", f"aligner-{source}-{lang}")
            path_to_transcript_dict: Dict[str, str] = dict()
            if not os.path.exists(toucan_file):
                continue
            with open(toucan_file, "r") as r:
                for line in r:
                    line = line.strip()
                    parts = line.split("|")
                    transcript: str = parts[1]
                    wav = os.path.join(source_base, source, parts[0])
                    wav = os.path.realpath(wav)
                    path_to_transcript_dict[wav] = transcript
            max_size: int = min(10_000, len(path_to_transcript_dict))
            items: List[Tuple[str, str]] = [*path_to_transcript_dict.items()]
            subset = dict(random.sample(items, max_size))
            datasets.append(prepare_aligner_corpus(transcript_dict=subset,
                                                   corpus_dir=corpus_dir,
                                                   lang=lang,
                                                   device=device,
                                                   loading_processes=1))

    train_set = ConcatDataset(datasets)
    save_dir = os.path.join("Models", "Aligner")
    os.makedirs(save_dir, exist_ok=True)
    save_dir_aligner = save_dir + "/aligner"
    os.makedirs(save_dir_aligner, exist_ok=True)

    train_aligner(train_dataset=train_set,
                  device=device,
                  save_directory=save_dir,
                  steps=500_000,
                  batch_size=32,
                  path_to_checkpoint=resume_checkpoint,
                  fine_tune=finetune,
                  debug_img_path=save_dir_aligner,
                  resume=resume)
