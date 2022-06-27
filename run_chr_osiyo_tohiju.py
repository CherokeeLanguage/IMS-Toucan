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

from InferenceInterfaces.InferenceFastSpeech2 import InferenceFastSpeech2

warnings.filterwarnings("ignore", category=UserWarning)

dest_folder: str = "samples.osiyo-tohiju"
model_id:str = "Cherokee_West"
MP3_HZ: int = 22050


def run_tts(tts: InferenceFastSpeech2, speaker_refs: List[str], text_file: str):
    speaker_refs.sort()
    lines: List[str] = list()
    shutil.copy(text_file, dest_folder)

    with open(text_file, "r") as r:
        for line in r:
            line = line.strip()
            if line:
                lines.append(line)

    for line in lines:
        pronounce: str = unicodedata.normalize("NFC", line.strip())
        print(pronounce)

    for speaker_ref in speaker_refs:
        ix: int = 0
        print(f"-{speaker_ref}")
        for line in lines:
            pronounce: str = unicodedata.normalize("NFC", line.strip())  # unicodedata.normalize("NFC", parts[1])
            file_name: str = re.sub("(?i)[^a-z0-9.!?,\\s]", "", unicodedata.normalize("NFD", pronounce))
            if file_name[-1] == ".":
                file_name = file_name[:-1]
            ix += 1
            mp3_file: str = f"{ix:03}-{file_name}.mp3"
            path_speaker_ref: str = os.path.join("ref", speaker_ref)
            dest_speaker_mp3: str = os.path.join(dest_folder, "ref-" + speaker_ref)[:-3] + "mp3"
            if not os.path.exists(dest_speaker_mp3):
                audio: AudioSegment = AudioSegment.from_file(path_speaker_ref)
                audio.set_frame_rate(MP3_HZ).export(dest_speaker_mp3, parameters=["-qscale:a", "3"])
            tts.set_utterance_embedding(path_speaker_ref)
            voice_folder: str = os.path.join(dest_folder, f"{speaker_ref[:-4]}")
            os.makedirs(voice_folder, exist_ok=True)
            wav_file = os.path.join(voice_folder, f"{mp3_file}.wav")
            tts.read_to_file([pronounce], wav_file, silent=True)
            audio: AudioSegment = AudioSegment.from_file(wav_file)
            audio.set_frame_rate(MP3_HZ).export(os.path.join(voice_folder, mp3_file), parameters=["-qscale:a", "3"])
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
    text_file: str = os.path.expanduser("~/git/Cherokee-TTS/data/tests/osiyo-tohiju-then-what/osiyo-tohiju-then-what.txt")
    run_tts(tts, speaker_refs, text_file)


if __name__ == '__main__':
    main()
