#!/usr/bin/env bash
"""true" '''\'
set -e
eval "$(${CONDA_EXE:-conda} shell.bash hook)"
conda deactivate
conda activate toucan_conda_venv
exec python "$0" "$@"
exit $?
''"""
from __future__ import annotations

import argparse
import dataclasses
import os
import pathlib
import shutil
import tempfile
from datetime import date
from datetime import datetime

import torch
from pydub import AudioSegment
from tqdm import tqdm

from InferenceInterfaces.InferenceFastSpeech2 import InferenceFastSpeech2


@dataclasses.dataclass
class TTSEntry:
    lang: str = "chr"
    pronunciation: str = ""
    ref_voice: pathlib.Path|None = None
    output_mp3: pathlib.Path|None = None

    @classmethod
    def parse(cls, text: str) -> "TTSEntry":
        """
        Parses text formatted as: lang|pronunciation|ref_voice_file|output_mp3_file
        """
        entry: TTSEntry = TTSEntry()
        fields: list[str] = text.split("|")
        if len(fields) != 4:
            raise RuntimeError(f"Bad TTS Entry record: {text}. Wrong field count of {len(fields)}.")
        entry.lang=fields[0].strip()
        entry.pronunciation=fields[1].strip()
        _ = fields[2].strip()
        if _:
            entry.ref_voice = pathlib.Path(_)
        _ = fields[3].strip()
        if _:
            entry.output_mp3= pathlib.Path(_)
        if not entry.pronunciation:
            raise RuntimeError(f"Bad TTS Entry record: {text}. Pronunciation missing.")
        if not entry.output_mp3:
            raise RuntimeError(f"Bad TTS Entry record: {text}. Output MP3 name missing.")
        return entry


def main():
    parser = argparse.ArgumentParser(description="IMS-Toucan TTS")
    parser.add_argument("--ref_dir", type=str, help="""
    Path to reference WAV audio folder for voices.
     Relative to source text file if not specified.""", required=False, default=None)
    parser.add_argument("--mp3_dir", type=str, help="""
    Output mp3 files to this folder.
     Relative to text file if not specified.""", required=False, default=None)
    parser.add_argument("--text_file", type=str, help="""
    Read from this file to create the TTS output.
     Format: lang|pronunciation|ref_voice_wav_file|output_mp3_file""", required=True)
    parser.add_argument("--alpha", type=float, help="""
    Speech duration multiplier.
     Defaults to 1.3 where 1.0 is normal speed. Effects entire batch.""", required=True, default=1.3)
    args = parser.parse_args()

    source_text: pathlib.Path = pathlib.Path(args.text_file)
    if not source_text.exists():
        raise RuntimeError(f"File {source_text} does not exist.")
    source_dir: pathlib.Path = source_text.resolve().parent
    ref_dir: pathlib.Path
    if args.ref_dir:
        ref_dir = pathlib.Path(args.ref_dir)
        if not ref_dir.is_dir():
            raise RuntimeError(f"Reference directory {ref_dir} does not exist.")
    else:
        ref_dir = source_dir
    output_dir: pathlib.Path
    if args.mp3_dir:
        output_dir = pathlib.Path(args.mp3_dir)
        if not output_dir.is_dir():
            raise RuntimeError(f"Output directory {output_dir} does not exist.")
    else:
        output_dir = source_dir

    source_text = source_text.resolve()
    ref_dir = ref_dir.resolve()
    output_dir = output_dir.resolve()

    for_processing: list[TTSEntry] = list()
    with open(source_text, "r") as r:
        for line in r:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tts_entry = TTSEntry.parse(line)
            for_processing.append(tts_entry)
    if not for_processing:
        return  # Ignore empty files

    alpha: float
    if args.alpha:
        alpha = args.alpha
    else:
        alpha = 1.0

    # TTS script needs to run from the TTS code directory for model loading to work.
    work_dir = os.getcwd()
    my_dir = pathlib.Path(__file__).resolve().parent
    os.chdir(my_dir)
    model_id = "Cherokee_West"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = InferenceFastSpeech2(device=device, model_name=model_id, alpha=alpha)
    default_utterance_embedding = tts.default_utterance_embedding
    os.chdir(work_dir)

    tts_entry: TTSEntry

    # Fix paths
    for tts_entry in for_processing:
        if tts_entry.ref_voice:
            tts_entry.ref_voice = ref_dir.joinpath(tts_entry.ref_voice)
        tts_entry.output_mp3 = output_dir.joinpath(tts_entry.output_mp3)

    # Process
    print(f"Batch processing {len(for_processing):,} entries.")
    prev_voice: pathlib.Path | None = None
    for tts_entry in tqdm(for_processing):
        if tts_entry.lang:
            tts.set_language(tts_entry.lang)
        else:
            tts.set_language("chr")  # Assume CHR if not specified.
        if tts_entry.ref_voice:
            if tts_entry.ref_voice != prev_voice:
                os.chdir(my_dir)
                tts.set_utterance_embedding(tts_entry.ref_voice)
                os.chdir(work_dir)
                prev_voice = tts_entry.ref_voice
        else:
            tts.default_utterance_embedding = default_utterance_embedding
            prev_voice = None
        tmp_dir: str = tempfile.mkdtemp(prefix="tts-ims-toucan-")
        tmp_wav: str = os.path.join(tmp_dir, "temp.wav")
        tts.read_to_file([tts_entry.pronunciation], tmp_wav, silent=True)
        audio: AudioSegment = AudioSegment.from_file(tmp_wav).set_channels(1).set_frame_rate(48_000)
        tags: dict = dict()
        tags["artist"] = "IMS-Toucan (https://github.com/CherokeeLanguage/IMS-Toucan)"
        tags["lyrics"] = tts_entry.pronunciation.strip()
        tags["title"] = tts_entry.pronunciation.strip()
        tags["genre"] = "Spoken"
        tags["copyright"] = f"©{date.today().year} Michael Conrad CC-BY"
        tags["year"] = date.today().year
        if tts_entry.lang:
            tags["lang"] = tts_entry.lang
        else:
            tags["lang"] = "chr"
        tags["publisher"] = "Michael Conrad"
        tags["date"] = str(datetime.utcnow().isoformat(sep="T", timespec="seconds"))
        audio.export(tts_entry.output_mp3, format="mp3", parameters=["-qscale:a", "0"], tags=tags)
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == '__main__':
    main()
