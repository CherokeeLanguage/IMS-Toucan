import os
import random
from typing import Dict
from typing import List
from typing import Tuple

import torch
from torch.utils.data import ConcatDataset

from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeech2 import FastSpeech2
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.meta_chr_loop import train_loop
from Utility.corpus_preparation import prepare_aligner_corpus


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume):
    if gpu_id == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")

    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"
        device = torch.device("cuda")

    torch.manual_seed(131714)
    random.seed(131714)
    torch.random.manual_seed(131714)

    print("Preparing")

    if model_dir is not None:
        save_dir = model_dir
    else:
        save_dir = os.path.join("Models", "FastSpeech2_Cherokee_West")
    os.makedirs(save_dir, exist_ok=True)

    langs: List[str] = ["de", "en", "fr", "nl", "ru", "chr"]
    source_base: str = "/mount/resources/speech/corpora"
    # Non Cherokee before Cherokee to get better quality voice weights as the default for the model
    sources: List[str] = ["other-audio-data", "cherokee-audio-data", "cherokee-audio-data-private"]
    datasets = list()

    for lang in langs:
        corpus_dir = os.path.join("Corpora", f"tts-{lang}")
        path_to_transcript_dict: Dict[str, str] = dict()
        for source in sources:
            toucan_file = os.path.join(source_base, source, f"ims-toucan-{lang}.txt")
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
    print("Training model")
    train_loop(net=FastSpeech2(lang_embs=100),
               device=torch.device("cuda"),
               datasets=datasets,
               batch_size=6,
               save_directory=save_dir,
               steps=500_000,
               steps_per_checkpoint=100,
               lr=0.001,
               path_to_checkpoint=resume_checkpoint,
               resume=resume)
