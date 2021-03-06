import re
import sys

import panphon
import phonemizer
import torch

from Preprocessing.papercup_features import generate_feature_table


class ArticulatoryCombinedTextFrontend:

    def __init__(self,
                 language,
                 use_word_boundaries=True,  # goes together well with
                 # parallel models and an aligner. Doesn't go together
                 # well with autoregressive models.
                 use_explicit_eos=True,
                 use_prosody=False,  # unfortunately the non-segmental
                 # nature of prosodic markers mixed with the sequential
                 # phonemes hurts the performance of end-to-end models a
                 # lot, even though one might think enriching the input
                 # with such information would help.
                 use_lexical_stress=True,
                 silent=True,
                 allow_unknown=False,
                 add_silence_to_end=True,
                 strip_silence=True):
        """
        Mostly preparing ID lookups
        """
        self.strip_silence = strip_silence
        self.use_word_boundaries = use_word_boundaries
        self.allow_unknown = allow_unknown
        self.use_explicit_eos = use_explicit_eos
        self.use_prosody = use_prosody
        self.use_stress = use_lexical_stress
        self.add_silence_to_end = add_silence_to_end
        self.feature_table = panphon.FeatureTable()

        if language == "en":
            self.g2p_lang = "en-us"
            self.expand_abbreviations = english_text_expansion
            if not silent:
                print("Created an English Text-Frontend")

        elif language == "de":
            self.g2p_lang = "de"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a German Text-Frontend")

        elif language == "el":
            self.g2p_lang = "el"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a Greek Text-Frontend")

        elif language == "es":
            self.g2p_lang = "es"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a Spanish Text-Frontend")

        elif language == "fi":
            self.g2p_lang = "fi"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a Finnish Text-Frontend")

        elif language == "ru":
            self.g2p_lang = "ru"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a Russian Text-Frontend")

        elif language == "hu":
            self.g2p_lang = "hu"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a Hungarian Text-Frontend")

        elif language == "nl":
            self.g2p_lang = "nl"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a Dutch Text-Frontend")

        elif language == "fr":
            self.g2p_lang = "fr-fr"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a French Text-Frontend")

        elif language == "it":
            self.g2p_lang = "it"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a Italian Text-Frontend")

        elif language == "pt":
            self.g2p_lang = "pt"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a Portuguese Text-Frontend")

        elif language == "pl":
            self.g2p_lang = "pl"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a Polish Text-Frontend")

        elif language == "chr" or language == "chr-w":
            self.g2p_lang = "chr-w"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created a Western Cherokee Text-Frontend")

        elif language == "chr-e":
            self.g2p_lang = "chr-e"
            self.expand_abbreviations = lambda x: x
            if not silent:
                print("Created an Eastern Cherokee Text-Frontend")

        # remember to also update get_language_id() when adding something here

        else:
            print("Language not supported yet")
            sys.exit()

        self.phone_to_vector_papercup = generate_feature_table()

        self.phone_to_vector = dict()
        for phone in self.phone_to_vector_papercup:
            panphon_features = self.feature_table.word_to_vector_list(phone, numeric=True)
            if panphon_features == []:
                panphon_features = [[0] * 24]
            papercup_features = self.phone_to_vector_papercup[phone]
            self.phone_to_vector[phone] = papercup_features + panphon_features[0]

        self.phone_to_id = {  # this lookup must be updated manually, because the only
            # other way would be extracting them from a set, which can be non-deterministic
            '~': 0,
            '#': 1,
            '?': 2,
            '!': 3,
            '.': 4,
            '??': 5,
            '??': 6,
            '??': 7,
            '??': 8,
            'a': 9,
            '??': 10,
            '??': 11,
            '??': 12,
            '???': 13,
            '??': 14,
            '??': 15,
            '??': 16,
            '??': 17,
            '??': 18,
            '??': 19,
            '??': 20,
            '??': 21,
            '??': 22,
            '??': 23,
            'b': 24,
            '??': 25,
            'd': 26,
            'e': 27,
            'f': 28,
            'g': 29,
            'h': 30,
            'i': 31,
            'j': 32,
            'k': 33,
            'l': 34,
            'm': 35,
            'n': 36,
            '??': 37,
            'o': 38,
            'p': 39,
            '??': 40,
            '??': 41,
            'r': 42,
            's': 43,
            't': 44,
            'u': 45,
            'v': 46,
            'w': 47,
            'x': 48,
            'z': 49,
            '??': 50,
            '??': 51,
            '??': 52,
            '??': 53,
            '??': 54,
            'y': 55,
            '??': 56,
            '??': 57,
            'c': 58,
            '??': 59,
            '??': 60,
            '??': 61,
            '??': 62,
            '??': 63,
            '??': 64,
            'q': 65,
            '??': 66,
            '??': 67,
            '??': 68,
            '??': 69,
            '??': 70,
            '??': 71,
            '??': 72,
            '??': 73,
            '??': 74,
            '??': 75,
            # Tone letters: https://en.wikipedia.org/wiki/Tone_letter
            # They are usually combined like the following:
            # ???? ???? ???? ???? ??????
            # ???? ???? ???? ???? ??????
            '\u02e5': 76,  # ?????
            '\u02e6': 77,  # ?????
            '\u02e7': 78,  # ?????
            '\u02e8': 79,  # ?????
            '\u02e9': 80,  # ?????
            # Lengthened and shortened vowels are grammatically important in some languages
            # https://en.wikipedia.org/wiki/Vowel_length
            '\u02d0': 81,  # ?????
            '\u02d1': 82,  # ?????
            '\u0306': 83,  # ?????
            # Stress impacts things like compound noun formation, dessert vs desert,
            # among other things
            '\u02c8': 84,  # ?? (primary) stress mark
            '\u02cc': 85,  # ?? secondary stress
            # for use by Russian, among other languages
            # see also: https://www.phon.ucl.ac.uk/home/wells/ipa-unicode.htm
            '\u02bc': 86,
            '\u02b4': 87,
            '\u02b0': 88,
            '\u02b1': 89,
            '\u02b7': 90,
            '\u02e0': 91,
            '\u02e4': 92,
            '\u02de': 93,

            }  # for the states of the ctc loss and dijkstra/mas in the aligner

        self.id_to_phone = {v: k for k, v in self.phone_to_id.items()}

    def string_to_tensor(self, text, view=False, device="cpu", handle_missing=True, input_phonemes=False):
        """
        Fixes unicode errors, expands some abbreviations,
        turns graphemes into phonemes and then vectorizes
        the sequence as articulatory features
        """
        if input_phonemes:
            phones = text
        else:
            phones = self.get_phone_string(text=text, include_eos_symbol=True)
        if view:
            print("Phonemes: \n{}\n".format(phones))
        phones_vector = list()
        # turn into numeric vectors
        for char in phones:
            if handle_missing:
                try:
                    phones_vector.append(self.phone_to_vector[char])
                except KeyError:
                    print("unknown phoneme: {}".format(char))
            else:
                phones_vector.append(self.phone_to_vector[char])  # leave error handling to elsewhere

        return torch.Tensor(phones_vector, device=device)

    def get_phone_string(self, text, include_eos_symbol=True):
        # expand abbreviations
        utt = self.expand_abbreviations(text)
        # phonemize
        phones: str
        if self.g2p_lang.startswith("chr"):
            from Preprocessing.lang_utils_chr import chr_mco_ipa
            phones = chr_mco_ipa(text)
        else:
            phones = phonemizer.phonemize(utt,
                                          language_switch='remove-flags',
                                          backend="espeak",
                                          language=self.g2p_lang,
                                          preserve_punctuation=True,
                                          strip=True,
                                          punctuation_marks=';:,.!???????????"??????????~/',
                                          with_stress=self.use_stress)

        phones = re.sub('[\u201C\u201D\u201E\u201F\u2033\u2036]', '"', phones)
        phones = re.sub("[\u2018\u2019\u201A\u201B\u2032\u2035]", "'", phones)
        phones = phones.replace(";", ",").replace("/", " ").replace("???", "").replace(":", ",").replace('"', ",") \
            .replace("-", ",").replace("...", ",").replace("-", ",").replace("\n", " ").replace("\t", " ") \
            .replace("??", "").replace("??", "").replace(",", "~").replace(" ??", "").replace('??', "").replace("??", "") \
            .replace("??", "")
        
        # less than 1 wide characters hidden here
        phones = re.sub("~+", "~", phones)
        if not self.use_prosody:
            # retain ~ as heuristic pause marker, even though all other symbols are removed with this option.
            # also retain . ? and ! since they can be indicators for the stop token
            # phones = phones.replace("??", "").replace("??", "").replace("??", "") \
            #     .replace("??", "").replace("|", "").replace("???", "")
            pass
        if not self.use_word_boundaries:
            phones = phones.replace(" ", "")
        else:
            phones = re.sub(r"\s+", " ", phones)
            phones = re.sub(" ", "~", phones)
        if self.strip_silence:
            phones = phones.lstrip("~").rstrip("~")
        if self.add_silence_to_end:
            phones += "~"  # adding a silence in the end during add_silence_to_end produces more natural sounding prosody
        if include_eos_symbol:
            phones += "#"

        phones = "~" + phones
        phones = re.sub("~+", "~", phones)

        return phones


def english_text_expansion(text):
    """
    Apply as small part of the tacotron style text cleaning pipeline, suitable for e.g. LJSpeech.
    See https://github.com/keithito/tacotron/
    Careful: Only apply to english datasets. Different languages need different cleaners.
    """
    _abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in
                      [('Mrs.', 'misess'), ('Mr.', 'mister'), ('Dr.', 'doctor'), ('St.', 'saint'), ('Co.', 'company'), ('Jr.', 'junior'), ('Maj.', 'major'),
                       ('Gen.', 'general'), ('Drs.', 'doctors'), ('Rev.', 'reverend'), ('Lt.', 'lieutenant'), ('Hon.', 'honorable'), ('Sgt.', 'sergeant'),
                       ('Capt.', 'captain'), ('Esq.', 'esquire'), ('Ltd.', 'limited'), ('Col.', 'colonel'), ('Ft.', 'fort')]]
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def get_language_id(language: str):
    if language == "de":
        return torch.LongTensor([1])
    elif language == "el":
        return torch.LongTensor([2])
    elif language == "es":
        return torch.LongTensor([3])
    elif language == "fi":
        return torch.LongTensor([4])
    elif language == "ru":
        return torch.LongTensor([5])
    elif language == "hu":
        return torch.LongTensor([6])
    elif language == "nl":
        return torch.LongTensor([7])
    elif language == "fr":
        return torch.LongTensor([8])
    elif language == "pt":
        return torch.LongTensor([9])
    elif language == "pl":
        return torch.LongTensor([10])
    elif language == "it":
        return torch.LongTensor([11])
    elif language == "en":
        return torch.LongTensor([12])
    elif language == "chr-w" or language == "chr":
        return torch.LongTensor([13])
    elif language == "chr-e":
        return torch.LongTensor([14])
    # Fallback tensor
    return torch.LongTensor([99])


if __name__ == '__main__':
    # test an English utterance
    tf = ArticulatoryCombinedTextFrontend(language="en")
    print(tf.string_to_tensor("This is a complex sentence, it even has a pause! But can it do this? Nice.", view=True))

    tf = ArticulatoryCombinedTextFrontend(language="de")
    print(tf.string_to_tensor("Alles klar, jetzt testen wir einen deutschen Satz. Ich hoffe es gibt nicht mehr viele unspezifizierte Phoneme.", view=True))
