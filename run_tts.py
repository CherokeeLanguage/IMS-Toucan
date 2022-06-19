#!/usr/bin/env bash
"""true" '''\'
set -e
eval "$(conda shell.bash hook)"
conda activate toucan_conda_venv
exec python "$0" "$@"
exit $?
''"""
from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import tempfile
from datetime import date
from datetime import datetime

import torch
from pydub import AudioSegment

from InferenceInterfaces.InferenceFastSpeech2 import InferenceFastSpeech2


def main():
    parser = argparse.ArgumentParser(description="IMS-Toucan TTS")
    parser.add_argument("--lang", type=str, help="Language model to use.", required=False, default="chr")
    parser.add_argument("--ref", type=str, help="Path to reference WAV audio for voice.", required=False, default=None)
    parser.add_argument("--mp3", type=str, help="Output mp3 file.", required=True)
    parser.add_argument("--text", type=str, help="The text to process to create the TTS output.", required=True)
    args = parser.parse_args()

    output_file = os.path.realpath(args.mp3)
    if args.ref:
        ref_file = os.path.realpath(args.ref)
    else:
        ref_file = None
    text: str = args.text

    work_dir = os.getcwd()
    my_dir = pathlib.Path(__file__).parent.absolute()
    os.chdir(my_dir)
    model_id = "Cherokee_West"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = InferenceFastSpeech2(device=device, model_name=model_id)
    os.chdir(work_dir)
    tts.set_language(args.lang)
    if args.ref:
        tts.set_utterance_embedding(ref_file)
    tmp_dir: str = tempfile.mkdtemp(prefix="tts-ims-toucan-")
    tmp_wav: str = os.path.join(tmp_dir, "temp.wav")
    tts.read_to_file([text], tmp_wav, silent=True)
    audio: AudioSegment = AudioSegment.from_file(tmp_wav).set_channels(1).set_frame_rate(48_000)
    tags: dict = dict()
    tags["artist"] = "IMS-Toucan (https://github.com/CherokeeLanguage/IMS-Toucan)"
    tags["lyrics"] = text.strip()
    tags["title"] = text.strip()
    tags["genre"] = "Spoken"
    tags["copyright"] = f"©{date.today().year} Michael Conrad CC-BY"
    tags["year"] = date.today().year
    tags["lang"] = args.lang
    tags["publisher"] = "Michael Conrad"
    tags["date"] = str(datetime.utcnow().isoformat(sep="T", timespec="seconds"))
    audio.export(output_file, format="mp3", parameters=["-qscale:a", "0"], tags=tags)
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
