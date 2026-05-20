"""
Simple character-level tokenizer.
"""
import string

SPECIALS = ['<pad>', '<sos>', '<eos>', '<unk>', '<img_start>', '<img_end>']
CHARS = string.printable  # 100 characters
VOCAB = SPECIALS + list(CHARS)
TOKEN_TO_ID = {ch: i for i, ch in enumerate(VOCAB)}
ID_TO_TOKEN = {i: ch for i, ch in enumerate(VOCAB)}

PAD_ID = TOKEN_TO_ID['<pad>']
SOS_ID = TOKEN_TO_ID['<sos>']
EOS_ID = TOKEN_TO_ID['<eos>']
UNK_ID = TOKEN_TO_ID['<unk>']
IMG_START_ID = TOKEN_TO_ID['<img_start>']
IMG_END_ID = TOKEN_TO_ID['<img_end>']
IMG_TOKEN_START = len(VOCAB)  # start of image tokens
NUM_IMG_TOKENS = 256
TOTAL_VOCAB = len(VOCAB) + NUM_IMG_TOKENS


def encode(text, max_len=None):
    ids = [SOS_ID] + [TOKEN_TO_ID.get(ch, UNK_ID) for ch in text] + [EOS_ID]
    if max_len:
        if len(ids) < max_len:
            ids = ids + [PAD_ID] * (max_len - len(ids))
        else:
            ids = ids[:max_len-1] + [EOS_ID]
    return ids


def encode_raw(text, max_len=None):
    ids = [TOKEN_TO_ID.get(ch, UNK_ID) for ch in text]
    if max_len:
        if len(ids) < max_len:
            ids = ids + [PAD_ID] * (max_len - len(ids))
        else:
            ids = ids[:max_len]
    return ids

def decode(ids, skip_specials=True):
    tokens = []
    for i in ids:
        if i == EOS_ID and skip_specials:
            break
        if i < len(VOCAB):
            tok = ID_TO_TOKEN[i]
            if skip_specials and tok.startswith('<'):
                continue
            tokens.append(tok)
        else:
            tokens.append(f'<img_{i}>')
    return ''.join(tokens)


if __name__ == '__main__':
    text = "red circle"
    ids = encode(text, max_len=20)
    print(ids)
    print(decode(ids))
