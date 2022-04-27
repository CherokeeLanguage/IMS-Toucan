import datetime
import os
import shutil
import sys
import textwrap
import warnings
from typing import List

import torch

from InferenceInterfaces.InferenceFastSpeech2 import InferenceFastSpeech2

warnings.filterwarnings("ignore", category=UserWarning)


def run_tts(tts: InferenceFastSpeech2, speaker_refs: List[str], file_prefix: str, text: str):
    texts = text.strip().splitlines()
    speaker_refs.sort()
    for text in texts:
        print(text)
    print()
    for speaker_ref in speaker_refs:
        path_speaker_ref: str = os.path.join("ref", speaker_ref)
        dest_speaker_ref: str = os.path.join("samples", "z_ref-" + speaker_ref)
        if not os.path.exists(dest_speaker_ref):
            shutil.copy(path_speaker_ref, dest_speaker_ref)
        tts.set_utterance_embedding(path_speaker_ref)
        file_location = os.path.join("samples", f"{file_prefix}-{speaker_ref}")
        tts.read_to_file(texts, file_location, silent=True)


def main():
    text: str
    model_id = "Cherokee_West"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = InferenceFastSpeech2(device=device, model_name=model_id)
    tts.set_language("chr")
    shutil.rmtree("samples", ignore_errors=True)
    os.mkdir("samples")
    speaker_refs: List[str] = list()
    for file in os.listdir("ref"):
        speaker_refs.append(file)

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
    run_tts(tts, speaker_refs, "bragging-hunter", text)

    text = textwrap.dedent("""
        Wahya ale jí:sdv:na.
        Ko:higv jigè:sv à:nè:he wahya ale jí:sdv:na.
        Sa:wúhnó: iyúwa̋kd à:gadu:lǐsgv aji:yè:sdi jí:sdv:na wahya.
        Wahya ù:ne:nu:hlane jí:sdv:na ju:hntohgǐ:yâsdi̋:ɂi.
        Jí:sdv:náhno kilawiyv dù:lchv̌:yâ:stane u:hntohkǐ:yâsdi̋:ɂi.
        "Gahlgwǒ:gíhnó: de:kánahltv yidé:nalgǒ:na," ù:dv:hne wahya.
        Ù:ndv́:nastanéhnó: ju:hntohgǐyâsdi.
        Agv:yi̋:hno wu:lúhjv́ galv:ndiɂa wikanahltv u:tohi̋:sdi gè:sv̌ɂi.
        Doyúhnó: sgi nù:ndv́:ne:le.
        agv:ydv: wù:tlv:sdane wahya.
        Ganí:daɂdv wù:go:he jí:sdv:na uhnáhnó: wù:tosé:ɂi.
        Wű:luhjahnó: galv̋:nad digè:sv vnawdv:sgwu naɂv ù:to:hi:se jí:sdv:na.
        Nǒ:wúhn taɂli:né wu:ni̋:lúhj u:dlőy nv:ndv́:ne:le alesgwu joɂi:né nvhgi:né hisgi:né sú:dali:né kv̋:hnihnó: gahlgwǒ:gi:né wù:ni:lúhjv́.
        Sda:yosv̋:dv: ù:tohi:se wahya.
        Jí:sdv:náhnó: nasgwu kilagwuyv ù:tohi:se asé:hno wahya,
        "Gadokv e:liw nitsanù:la ǐ:jú:la di:ni̋:lúhg áhan kanahltv̋:ɂi?"
        Jí:sdv:náhnó:, "Ji:yenű:lidv:," ù:da:dosě:le.
        Asé:hno wahya nǎ:wu ù:delho:se gani:dadv̋:hnó: gatosge jí:sdv:n vsgin yű:sd ǐ:jú:lahaw wani:luhgi de:kanahltv̋:ɂi.
        Wahyáhnó: ù:delho:se nǎ:wúhnó: dà:gagahnane jí:sdv:na, "Do:yúdv: hilo:nu:he."
        "Jini:dadv́hnó: tskilawdi:se," à:go:sě:le jí:sdv:na.
        Uhnawdvhno ajikehǐ:dô:le jí:sdv:n nigayejini:yi̋:sg.
        Uhnáhnó: wajini:yv̋:hnó: waji:yaɂohne v̀:sgiwuhnó: nigǎ:ɂa.        
        """)
    run_tts(tts, speaker_refs, "wolf-and-crawdad", text)

    text = textwrap.dedent("""
            Ju:naktenolǐ:dô:le.
            Luhiyv jige:sv tlasi esga̋: ù:nade:hnv.
            Ù:ni:ɂlúhjv jige:se gâ:yul ù:né:dô:le ù:naktě:nô:lǐ:dô:le i:ga̋:d.
            U:ge:lawe:d jige:sv sigwo:y ju:nda:nv̋:tl.
            Dà:tihnǐ:dô:he da:hnawa ane:dő:.
            E:gwő:n ù:wé:yv ù:ni:sdâ:wadv̌:dô:le.
            Hilv̋:sg ane:hv́ ù:ně:dô:lv.
            Dù:nó:ɂe nahiyv jige:sv.
            "Na:nv ju:yo:hu:sv́," jv:dù:do ko:hi jiga.
            Ù:ně:dô:le ù:hna ù:ni:go:wáhv kilo ù:yo:hu:sv.
            Sgihnoɂiyű:sd ju:yo:hu:sv́ dù:nó:ɂe ù:hná:na.
            Nu:le salù:ynige:yv di:dl ù:ně:dô:le ù:hná:na.
            U:ni̋:luhj sgwi:sdosv́ salű:y iyű:sd u:wande̋:sgi iyű:sdi salu:y ge:se.
            V:sginoyusd salû:ynige:yv dù:nó:ɂe.
            ù:hna ù:we:yv nu:le da:hnú:gó, didl ù:we:yv ù:ni:luhje.
            Ù:hnanv sgi:sdosv́ dù:ni:go:he ajaɂd da:hnú:g iysd.
            U:sginoyu:sd jidu:do dahnugó nu:le ajisgvnige:sdv ù:we:yv ù:ni:luhje.
            Ù:hna:nvsgwu ajaɂd sgwi̋:sdosv́.
            Ani:jisgvnige dù:ni:go:he.
            U:sginoyű:sd jidù:dó: ajisgvnige:sdv.
            Nu:le jű:sgwagahli jidu:dó: ù:ni:ɂluhje agő:di yű:sd ge:se ù:hná:nv.
            Ani:ɂahaw ju:ni̋:sgwagahl sgwi:sdosv́ dù:ni:go:he.
            U:sginv "jű:sgwagahli" jidù:dó:ɂa.
            Ù:hna á:mó wù:ni:luhje ù:hnanvsg.
            Á:m sgwi:sdosv́ kvnage:sv galgě:ye u:nasgiyű:sd á:mó jidù:dó:ɂe.
            """)
    run_tts(tts, speaker_refs, "search-party", text)


if __name__ == '__main__':
    main()
