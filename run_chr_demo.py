import datetime
import os
import sys
import textwrap
import warnings

import torch

from InferenceInterfaces.InferenceFastSpeech2 import InferenceFastSpeech2

if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=UserWarning)
    available_models = os.listdir("Models")
    available_fastspeech_models = list()
    for model in available_models:
        if model.startswith("FastSpeech2_"):
            available_fastspeech_models.append(model.lstrip("FastSpeech_2"))
    model_id = "Cherokee_West"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = InferenceFastSpeech2(device=device, model_name=model_id)
    tts.set_language("chr")

    speaker_refs = ["ref-cno-f-1.wav", "ref-cno-f-2.wav", "ref-cno-f-3.wav", "ref-cno-m-1.wav", "ref-vctk-p310.wav",
                    "ref-vctk-p341.wav", "ref-bc.wav", "ref-sam-hider.wav", "ref-wwacc.wav", "ref-mconrad.wav",
                    "ref-df.wav", "ref-de-41.wav", "ref-de-29.wav", "ref-ru-04.wav"]

    text = textwrap.dedent("""
    Anǐ:táɂli ani:sgaya à:ni:no:halǐ:dô:he, ahwi dù:ni:hyohe.
    Sà:gwű:hno asgaya galò:gwé ga:ne:he sóɂíhnv́ hlā.
    Ná:hnv́ galò:gwé ga:ne̋:hi u:dlv̌:kwsati gè:sé, ale go:hű:sdi yǔ:dv̂:ne̋:la à:dlv̌:kwsgé.
    À:ná:ɂi:sv̋:hnv go:hű:sdi wǔ:ní:go:he do:juwáɂihlv, ná:hnv́ galò:gwé ga:ne̋:hi kilagwu iyv̋:da widǔ:sdáyo:hlé ǒ:sdagwu nǔ:ksestanv̋:na iyú:sdi dà:sdayo:hihv̋.
    U:do:hiyű:hnv́ wǔ:yó:hlé ale ù:ni:go:hé ganv́:gv̋.
    Ná:hnv́ galò:gwé nigǎ:né:hv̋:na "Ahwi è:ni:yó:ɂa!", ù:dv:hne.
    "Ji:yó:ɂê:ga", ù:dv:hne ná galò:gwé ga:ne̋:hi, à:dlv̌:kwsgv́.
    Ù:ná:ne:lǔ:gî:se do:juwáɂihlv́ dí:dla, naɂv̌:hníge̋:hnv wǔ:ní:luhja ù:ni:go:hé sǒ:gwíli gáɂnv̋.
    "Sǒ:gwílílê ì:nada:hísi", ù:dv:hné ná u:yo:hlv̋.
    "Hada:hísê:gá", à:gò:sě:lé.
    """)
    texts = text.strip().splitlines()

    for speaker_ref in speaker_refs:
        print(f"=== {speaker_ref}")
        tts.set_utterance_embedding(speaker_ref)
        file_location = f"_bragging-hunter-{speaker_ref}"
        tts.read_to_file(texts, file_location)
        print()
