#!/usr/bin/env bash
"""true" '''\'
set -e
eval "$(conda shell.bash hook)"
conda deactivate
conda activate toucan_conda_venv
exec python "$0" "$@"
exit $?
''"""
import _csv
import csv
import os
import re
import shutil
import unicodedata
import warnings
from pathlib import Path

import torch
from pydub import AudioSegment
from typing import List

from tqdm import tqdm
from typing import Set

from InferenceInterfaces.InferenceFastSpeech2 import InferenceFastSpeech2

warnings.filterwarnings("ignore", category=UserWarning)

dest_folder: str = "samples.ced"
model_id: str = "Cherokee_West"
MP3_HZ: int = 48_000
QA: str = "3"


def run_tts(tts: InferenceFastSpeech2, text_file: str):
    already: Set[str] = set()
    lines: List[str] = list()
    shutil.copy(text_file, dest_folder)
    with open(text_file, "r") as r:
        for line in r:
            line = line.strip()
            if line:
                lines.append(line)
    csv_file: Path = Path(dest_folder).joinpath("pronunciations.csv")
    pipe_file: Path = Path(dest_folder).joinpath("pronunciations.txt")

    with open(csv_file, "w") as w, open(pipe_file, "w") as u:
        wcsv: _csv.writer = csv.writer(w)
        wcsv.writerow(["MCO_PRONUNCIATION", "MCO_TRANSLIT", "GENDER", "VOICE", "AUDIO_FILE", "DURATION"])
        u.write("MCO|TRANSLIT|G|VOICE|FILE|DURATION\n")
        prev_speaker: str = ""
        for line in tqdm(lines):
            parts: List[str] = line.split("|")

            pronounce: str = unicodedata.normalize("NFC", parts[0]).lower()
            gender: str = parts[1]
            voice: str = parts[2]
            mp3_file: str = parts[3]

            if mp3_file in already:
                print(f"WARNING: Duplicate mp3_file name. {pronounce}→{mp3_file}")
            already.add(mp3_file)

            voice_folder: str = os.path.join(dest_folder, f"{voice}")
            mp3_output: str = os.path.join(voice_folder, mp3_file)

            path_speaker_ref: str = os.path.join("ref", voice)
            if prev_speaker != path_speaker_ref:
                tts.set_utterance_embedding(f"{path_speaker_ref}.wav")
                prev_speaker = path_speaker_ref

            os.makedirs(voice_folder, exist_ok=True)
            wav_file = os.path.join(voice_folder, f"{mp3_file}.wav")
            tts.read_to_file([pronounce], wav_file, silent=True)
            audio: AudioSegment = AudioSegment.from_file(wav_file)
            duration: str = srt_ts(audio.duration_seconds)
            audio.set_frame_rate(MP3_HZ).export(mp3_output+".tmp", parameters=["-qscale:a", QA])
            os.rename(mp3_output+".tmp", mp3_output)
            mco_translit: str = re.sub("(?i)[^a-z ]", "", unicodedata.normalize("NFD", pronounce))
            wcsv.writerow([pronounce, mco_translit, gender, voice, mp3_file, duration])
            u.write(f"{pronounce}|{mco_translit}|{gender}|{voice}|{mp3_file}|{duration}\n")
            os.remove(wav_file)


def srt_ts(position: float) -> str:
    """Returns SRT formatted timestamp string where position is in seconds."""
    ms = int(position * 1000) % 1000
    seconds = int(position) % 60
    minutes = int((position // 60) % 60)
    hours = int(position // (60 * 60))
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"


def main():
    text: str
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = InferenceFastSpeech2(device=device, model_name=model_id, alpha=1.4)
    tts.set_language("chr")
    shutil.rmtree(dest_folder, ignore_errors=True)
    os.makedirs(dest_folder, exist_ok=True)
    speaker_refs: List[str] = list()
    for file in os.listdir("ref"):
        speaker_refs.append(file)
    text_file: str = os.path.expanduser("~/git/audio-lessons-generator-python/data/ced-for-tts.txt")
    run_tts(tts, text_file)


if __name__ == '__main__':
    main()
