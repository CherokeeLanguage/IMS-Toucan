#!/usr/bin/env bash
"""true" '''\'
set -e
eval "$(conda shell.bash hook)"
conda deactivate
conda activate toucan_conda_venv
exec python "$0" "$@"
exit $?
''"""

import datetime
import os
import re
import shutil
import sys
import textwrap
import unicodedata
import warnings
from typing import List

import torch
from pydub import AudioSegment
from tqdm import tqdm

from InferenceInterfaces.InferenceFastSpeech2 import InferenceFastSpeech2

warnings.filterwarnings("ignore", category=UserWarning)

dest_folder: str = "samples.bound-pronouns-app"

model_id:str = "Cherokee_West"

IX_CHEROKEE: int = 7
IX_ENGLISH: int = 8
IX_FILE_NAME: int = 11


def run_tts(tts: InferenceFastSpeech2, speaker_refs: List[str], text_file: str):
    speaker_refs.sort()
    lines: List[str] = list()
    shutil.copy(text_file, dest_folder)

    with open(text_file, "r") as r:
        for line in r:
            line = line.strip()
            if line:
                lines.append(line)

    with open(os.path.join(dest_folder, "bound-pronouns.txt"), "w") as w:
        for line in lines:
            line: str = unicodedata.normalize("NFC", line.strip())
            parts: List[str] = line.split("|")
            pronounce: str = parts[IX_CHEROKEE]
            english: str = parts[IX_ENGLISH]
            file_name: str = parts[IX_FILE_NAME]
            mp3_file: str = f"{file_name}.mp3"
            if file_name == "APP_FILE":
                continue
            w.write(f"{pronounce}|{mp3_file}|{english}\n")

    for speaker_ref in speaker_refs:
        print(f"{speaker_ref}")
        for line in tqdm(lines):
            line: str = unicodedata.normalize("NFC", line.strip())
            parts: List[str] = line.split("|")
            pronounce: str = parts[7]
            file_name: str = parts[9]
            mp3_file: str = f"{file_name}.mp3"
            if file_name == "APP_FILE":
                continue
            path_speaker_ref: str = os.path.join("ref", speaker_ref)
            dest_speaker_mp3: str = os.path.join(dest_folder, "ref-" + speaker_ref)[:-3] + "mp3"
            if not os.path.exists(dest_speaker_mp3):
                audio: AudioSegment = AudioSegment.from_file(path_speaker_ref)
                audio.export(dest_speaker_mp3, parameters=["-qscale:a", "3"])
            tts.set_utterance_embedding(path_speaker_ref)
            voice_folder: str = os.path.join(dest_folder, f"{speaker_ref[:-4]}")
            os.makedirs(voice_folder, exist_ok=True)
            wav_file = os.path.join(voice_folder, f"{mp3_file}.wav")
            tts.read_to_file([pronounce], wav_file, silent=True)
            audio: AudioSegment = AudioSegment.from_file(wav_file)
            audio.export(os.path.join(voice_folder, mp3_file), parameters=["-qscale:a", "3"])
            os.remove(wav_file)


def main():
    text: str
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = InferenceFastSpeech2(device=device, model_name=model_id)
    tts.set_language("chr")
    shutil.rmtree(dest_folder, ignore_errors=True)
    os.mkdir(dest_folder)
    speaker_refs: List[str] = list()
    for file in os.listdir("ref"):
        speaker_refs.append(file)
    text_file: str = os.path.expanduser("~/git/audio-lessons-generator-python/bound-pronouns.txt")
    run_tts(tts, speaker_refs, text_file)


if __name__ == '__main__':
    main()
