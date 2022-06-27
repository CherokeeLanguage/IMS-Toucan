import datetime
import os
import shutil
import sys
import textwrap
import warnings
from typing import List

import torch
from pydub import AudioSegment

from InferenceInterfaces.InferenceFastSpeech2 import InferenceFastSpeech2

warnings.filterwarnings("ignore", category=UserWarning)

dest_folder: str = "samples.stories"

MP3_HZ: int = 22050


def run_tts(tts: InferenceFastSpeech2, speaker_refs: List[str], file_prefix: str, text: str):
    text = textwrap.dedent(text)
    texts = text.strip().splitlines()
    speaker_refs.sort()
    with open(os.path.join(dest_folder, f"{file_prefix}.txt"), "w") as w:
        for text in texts:
            print(text)
            w.write(text.strip())
            w.write("\n")
        print()

    for speaker_ref in speaker_refs:
        path_speaker_ref: str = os.path.join("ref", speaker_ref)
        dest_speaker_mp3: str = (os.path.join(dest_folder, "z_ref-" + speaker_ref))[:-3] + "mp3"
        if not os.path.exists(dest_speaker_mp3):
            audio: AudioSegment = AudioSegment.from_file(path_speaker_ref)
            audio.set_frame_rate(MP3_HZ).export(dest_speaker_mp3)
        tts.set_utterance_embedding(path_speaker_ref)
        file_location = os.path.join(dest_folder, f"{file_prefix}-{speaker_ref}")
        tts.read_to_file(texts, file_location, silent=True)
        audio: AudioSegment = AudioSegment.from_file(file_location)
        mp3_file = file_location[:-3] + "mp3"
        audio.set_frame_rate(MP3_HZ).export(mp3_file)
        os.remove(file_location)


def main():
    model_id = "Cherokee_West"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = InferenceFastSpeech2(device=device, model_name=model_id)
    tts.set_language("chr")
    shutil.rmtree(dest_folder, ignore_errors=True)
    os.mkdir(dest_folder)
    speaker_refs: List[str] = list()
    for file in os.listdir("ref"):
        speaker_refs.append(file)

    turtle_rabbit = """
        Nu:lstanǐ:dô:lv daks du:ki:yv jisd.
        Niga̋:dadv ù:nahnte jisd ő:sd atló:dő:hi gè:sv́.
        U:ntohgǐ:yâ:sdi u:ni:hno:hě:hle ji:sd nahn daks.
        U:hnte gv:wtlő:hisd ge:hv́, na daks u:sganő:l ge:hv́ atlí:dő:hi.
        Dù:nuktane na yv i:g v̀:sgina yu:dv̌:hndi.
        Daksisgin gè:hv́hno dù:hlinohehtane ju:li̋: no:wle sida:ne:lv ani:ne̋:.
        Nu:sdv du:wu:ktv́ dù:hno:se:le nigv:wadv̋:hnd gèhv gv:wada:tlő:hisd gè:hv́.
        No:w ù:sgwalvhihle i:g ané:hnaɂi nikv́ u:ndahlisane u:naktosdohdi ahntohgǐ:yâ:sdi.
        Nù:ndv:ne:le ani:soɂ daks du:hno:se:lv́ nu:sdv̋hn du:wu:ktnanv́.
        "Nvw agv̋:yi gadú:s. Yigv:lisgohldâ:s agv̋:y wijáɂlohisdi. Si:n ay jo yagilu:l o:hni yiga̋:," u:dv:ne ji:sd.
        V̀:sgin nu:sdv́ u:ni:hno:he:hlv́ na nù:ndv:ne:le.
        Ù:hni:gi:se daksi.
        Agv̋:yi jo:dalv watlí:sé wù:go:he ji:sd.
        Nvv̌:w u:nale:nv́ ahntohkiyasgv́.
        Nu:sdv ù:ni:hno:he:hlv dakshnó: nagw nandv:ne:hv́ sa:gwuha si:danelv́ ane̋: uhna ju:li̋:ɂíle yig.
        Jo:dalv yiwű:luhj ji:sd uhna wagotisge daks wikanalu:sgv́ wù:dé:li:gv́ yu:sdi:ha.
        Sóɂ jo:dale yiwű:luhj v̀:sgi nà:dv́:ne:he.
        V:sgi:yv o:hni jò:dalv.
        Kil wa:tli:sv́ no:gw ju:yvwé:chonv́ ge:se ji:sd.
        No:gw wű:luhj wu:hnalu:sv́ o:hni jo:dalv wù:go:he no:gwu na daks dù:kǐ:yâ:sgv́ wigalo:sgv́ u:ndahlohisdi à:sdanv:hnv́.
        Nogw wu:go:he watli:sv́ na daks ji:sduhnv ju:yawé:chonv́ ge:hehno.
        Wù:nv̂:jî:tle naɂv.
        Tla yu:hnte ji:sdu ge:he nu:ndv́:ně:lv́.
        Asé:, niga̋:d u:ni:tlo:yí:ha ge:se, daks ju:li no:le sidane:lv ani:ne̋:.
        Tla yade:loho:sge nandv́:nè:hv́.
        Na daks sa:gwuha na gadú:s o:dalvle yig è:do:he hahn o:hnihno o:dalv na ju:le:nv́.
        Daks ge:se yu:du:li̋: gő:sd kilo u:tvdi nu:lstani:do:lv́, wǔ:nv̂:ji̋:hla na ji:sd ju:yawé:chonv ge:se u:tlő:yigw jinadv́:ne:ho kohiyv jíg yidu:yawe̋:j yigánv́:gigwu.
        """
    run_tts(tts, speaker_refs, "turtle-beat-rabbit", turtle_rabbit)

    bragging_hunter = """
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
        """
    run_tts(tts, speaker_refs, "bragging-hunter", bragging_hunter)

    wolf_crawdad = """
        Wahya ale jí:sdv:na.
        Ko:higv jigè:sv à:nè:he wahya ale jí:sdv:na.
        Sa:wúhnó: iyúwa̋kd à:gadu:lǐsgv aji:yè:sdi jí:sdv:na wahya.
        Wahya ù:ne:nu:hlane jí:sdv:na ju:hntohgǐ:yâsdi̋:ɂi.
        Jí:sdv:náhno kilawiyv dù:lchv̌:yâ:stane u:hntohkǐ:yâsdi̋:ɂi.
        "Gahlgwǒ:gíhnó: de:kánahltv yidé:nalgǒ:na," ù:dv:hne wahya.
        Ù:ndv́:nastanéhnó: ju:hntohgǐyâsdi.
        Agv:yi̋:hno wu:lúhjv́ galv:ndiɂa wikanahltv u:tohi̋:sdi gè:sv̌ɂi.
        Doyúhnó: sgi nù:ndv́:ne:le.
        Agv:ydv: wù:tlv:sdane wahya.
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
        """
    run_tts(tts, speaker_refs, "wolf-and-crawdad", wolf_crawdad)

    search_party = """
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
            """
    run_tts(tts, speaker_refs, "search-party", search_party)

    ball_of_fire = """
    Ja:gwa:tv:si:di:sv i:lv́hdlv̋ hiksgǒ gahlgwǒ:gi ju:de:tiyv́:sadí:sv̋ hlá a:sí di:na:ga:li:sgi
    yidǒ:gí:si:ladé o:ge:nv́:sv̋ galhjǒ:de o:gi:lv̌:kwdi ge:sv́ naɂv́
    i:jo:ga:da:li ju:né:nv:sv wo:ge:da:sdi di:da:yv:la:tv:sgi u:ni:hv alé
    ju:li:si:hnv:dagwu iyű:sdi ilv̋:sgi iyada:ne:lv da:nadlo:sǐ:híhv́
    di:da:yv:la:tv:sgi ju:na:ga:tǒ:stáni:hlv̋:ɂi sa:gwú iyúwa:go:di
    sv:nǒ:yi wíɂo:gi:lu:hja jo:gé:nv:sv
    gi:tli u:sga:se:hdǐ:gwu nigawe:sgv́ da:suhwi:sgv́ uhló:yigwu go:hű:sdi á:gwuiyv́ ja:go:wa:tǐ:sgo
    u:hló:yi na:dv́:ne:hv́ á:ne:lǔ:gî:sgv́ áge iyú:danv:hi̋:da no:gwúlé díajv́:sgv́
    [17]
    """
    # run_tts(tts, speaker_refs, "narrative_011", ball_of_fire)


if __name__ == '__main__':
    main()
