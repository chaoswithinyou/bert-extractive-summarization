import os
cwd = os.getcwd()
import time
import numpy as np
import torch
#from nltk.tokenize import sent_tokenize
from models.model_builder import ExtSummarizer
import collections
#from nltk import tokenize



def preprocess(raw_sents):
    substring = 'Ảnh'
    fil_sents = []
    for sent in raw_sents:
        if substring not in sent:
            fil_sents.append(sent)
    #processed_text = " [SEP] [CLS] ".join(sents)
    return fil_sents, len(fil_sents)

def load_vocab():
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(cwd+'/added_vocab.txt', "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            token = token.split()
            vocab[token[0]] = index
            index += 1
    with open(cwd+'/vocab.txt', "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            token = token.split()
            vocab[token[0]] = index
            index += 1
    last_1 = list(vocab.keys())[-1]
    last_2 = list(vocab.keys())[-2]
    last_3 = list(vocab.keys())[-3]
    vocab['[unused0]'] = vocab[last_3]
    vocab['[unused1]'] = vocab[last_2]
    vocab['[unused2]'] = vocab[last_1]
    del vocab[last_1]
    del vocab[last_2]
    del vocab[last_3]
    vocab['[mask]'] = index
    return vocab


def load_text(sents, max_pos, device):
    #tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    #token2idx = tokenizer.get_vocab()
    token2idx = load_vocab()
    sep_vid = 2
    cls_vid = 0

    def _process_src(raw):
        #raw = raw.strip().lower()
        #raw = raw.replace("[cls]", "[CLS]").replace("[sep]", "[SEP]")
        #src_subtokens = tokenizer.tokenize(raw)
        #src_subtokens = tokenize.word_tokenize(raw)
        #src_subtokens = ["[CLS]"] + src_subtokens + ["[SEP]"]
        src_subtoken_idxs = []
        for sent in raw:
            #sent_tokens = tokenize.word_tokenize(sent)
            for token in sent:
                try:
                    src_subtoken_idxs.append(token2idx[token])
                except Exception:
                    pass
            src_subtoken_idxs.append(0)
            src_subtoken_idxs.append(2)
        src_subtoken_idxs = [0] + src_subtoken_idxs + [2]
        src_subtoken_idxs = src_subtoken_idxs[:-1][:max_pos]
        src_subtoken_idxs[-1] = sep_vid
        _segs = [-1] + [i for i, t in enumerate(src_subtoken_idxs) if t == sep_vid]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]
        
        segments_ids = []
        segs = segs[:max_pos]
        for i, s in enumerate(segs):
            if i % 2 == 0:
                segments_ids += s * [0]
            else:
                segments_ids += s * [1]

        src = torch.tensor(src_subtoken_idxs)[None, :].to(device)
        mask_src = (1 - (src == 0).float()).to(device)
        cls_ids = [[i for i, t in enumerate(src_subtoken_idxs) if t == cls_vid]]
        clss = torch.tensor(cls_ids).to(device)
        mask_cls = 1 - (clss == -1).float()
        clss[clss == -1] = 0
        return src, mask_src, segments_ids, clss, mask_cls

    src, mask_src, segments_ids, clss, mask_cls = _process_src(sents)
    segs = torch.tensor(segments_ids)[None, :].to(device)
    #src_text = [[sent.replace("[SEP]", "").strip() for sent in processed_text.split("[CLS]")]]
    src_text = [sents]
    return src, mask_src, segs, clss, mask_cls, src_text


def test(model, input_data, top_n, block_trigram=True):
    with torch.no_grad():
        src, mask, segs, clss, mask_cls, src_str = input_data
        sent_scores, mask = model(src, segs, clss, mask, mask_cls)
        sent_scores = sent_scores + mask.float()
        sent_scores = sent_scores.cpu().data.numpy()
        selected_ids = np.argsort(-sent_scores, 1)
        return selected_ids[0][:top_n]



def summarize(raw_sents, model, top_n=10, max_pos=1024):
    model.eval()
    processed_text, full_length = preprocess(raw_sents)
    input_data = load_text(processed_text, max_pos, device="cpu")
    return test(model, input_data, top_n, block_trigram=True)
        
