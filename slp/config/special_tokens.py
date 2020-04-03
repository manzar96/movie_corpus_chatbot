from enum import Enum


class SPECIAL_TOKENS(Enum):
    PAD = '[PAD]'
    MASK = '[MASK]'
    UNK = '[UNK]'
    BOS = '[BOS]'
    EOS = '[EOS]'
    CLS = '[CLS]'



    @classmethod
    def has_token(cls, token):
        return any(token == t.name or token == t.value
                   for t in cls)

    @classmethod
    def to_list(cls):
        return list(map(lambda x: x.value, cls))


class HRED_SPECIAL_TOKENS(Enum):
    """
    end-of-utterance: </s>
    end-of-dialogue: </d>
    first speaker: <first_speaker>
    second speaker: <second_speaker>
    third speaker: <third_speaker>
    minor speaker: <minor_speaker>
    voice over: <voice_over>
    off screen: <off_screen>
    pause: <pause>
    """

    PAD = '<PAD>'
    UNK = '<unk>'
    SOS = '<s>'
    EOS = '</s>'
    EOT = '</t>'

    @classmethod
    def has_token(cls, token):
        return any(token == t.name or token == t.value
                   for t in cls)

    @classmethod
    def to_list(cls):
        return list(map(lambda x: x.value, cls))


class DIALOG_SPECIAL_TOKENS(Enum):
    PAD = '<pad>'
    UNK = '<unk>'
    SOS = '<s>'
    EOS = '</s>'

    @classmethod
    def has_token(cls, token):
        return any(token == t.name or token == t.value
                   for t in cls)

    @classmethod
    def to_list(cls):
        return list(map(lambda x: x.value, cls))