import random
from typing import Tuple

import torch
import torch.multiprocessing
from tqdm import tqdm

from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.FastSpeech2 import FastSpeech2
from TrainingInterfaces.Text_to_Spectrogram.FastSpeech2.meta_chr_loop import train_loop
from Utility.corpus_preparation import prepare_fastspeech_corpus
from Utility.path_to_transcript_dicts import *


def run(gpu_id, resume_checkpoint, finetune, model_dir, resume, remove_faulty_samples=False):
    torch.manual_seed(131714)
    random.seed(131714)
    torch.random.manual_seed(131714)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_id}"

    datasets = list()

    base_dir: str
    base_dir = os.path.join("Models", "FastSpeech2_Cherokee_West")
    if model_dir is not None:
        meta_save_dir = model_dir
    else:
        meta_save_dir = base_dir
    os.makedirs(meta_save_dir, exist_ok=True)

    print("Preparing")
    langs: List[str] = ["en", "fr", "nl", "ru", "de", "chr"]
    source_base: str = "/mount/resources/speech/corpora"
    sources: List[str] = ["other-audio-data", "cherokee-audio-data", "cherokee-audio-data-private"]
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
        max_size: int
        max_size = 8_000
        items: List[Tuple[str, str]] = [*path_to_transcript_dict.items()]
        while len(items) < max_size:
            items.extend(items.copy())
        subset = dict(random.sample(items, max_size))
        datasets.append(prepare_fastspeech_corpus(  #
                transcript_dict=subset,  #
                corpus_dir=corpus_dir,  #
                lang=lang,  #
                fine_tune_aligner=True))

    if remove_faulty_samples:
        print("Scanning for faulty samples")
        find_and_remove_faulty_samples(net=FastSpeech2(lang_embs=100),  #
                                       datasets=datasets,  #
                                       device=torch.device("cuda"),  #
                                       path_to_checkpoint=resume_checkpoint)

    train_loop(net=FastSpeech2(lang_embs=100),  #
               device=torch.device("cuda"),  #
               datasets=datasets,  #
               batch_size=6,  #
               save_directory=meta_save_dir,  #
               steps=140_001,  #
               steps_per_checkpoint=5_000,  #
               lr=0.001,  #
               path_to_checkpoint=resume_checkpoint,  #
               resume=resume)


@torch.inference_mode()
def find_and_remove_faulty_samples(net, datasets, device, path_to_checkpoint):
    net = net.to(device)
    torch.multiprocessing.set_sharing_strategy('file_system')
    check_dict = torch.load(os.path.join(path_to_checkpoint), map_location=device)
    net.load_state_dict(check_dict["model"])
    for dataset_index in range(len(datasets)):
        nan_ids = list()
        for datapoint_index in tqdm(range(len(datasets[dataset_index]))):
            loss = net(text_tensors=datasets[dataset_index][datapoint_index][0].unsqueeze(0).to(device),
                       text_lengths=datasets[dataset_index][datapoint_index][1].to(device),
                       gold_speech=datasets[dataset_index][datapoint_index][2].unsqueeze(0).to(device),
                       speech_lengths=datasets[dataset_index][datapoint_index][3].to(device),
                       gold_durations=datasets[dataset_index][datapoint_index][4].unsqueeze(0).to(device),
                       gold_pitch=datasets[dataset_index][datapoint_index][6].unsqueeze(0).to(device),
                       # mind the switched order
                       gold_energy=datasets[dataset_index][datapoint_index][5].unsqueeze(0).to(device),
                       # mind the switched order
                       utterance_embedding=datasets[dataset_index][datapoint_index][7].unsqueeze(0).to(device),
                       lang_ids=datasets[dataset_index][datapoint_index][8].unsqueeze(0).to(device),
                       return_mels=False).squeeze()
            if torch.isnan(loss):
                print(f"NAN DETECTED: {dataset_index}, {datapoint_index}")
                nan_ids.append(datapoint_index)
        datasets[dataset_index].remove_samples(nan_ids)
        if nan_ids:
            print(f"Removed {len(nan_ids):,} bad data points")
