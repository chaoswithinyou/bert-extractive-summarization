import os
cwd = os.getcwd()
import time
import numpy as np
import torch
from transformers import AutoTokenizer
#from nltk.tokenize import sent_tokenize
from models.model_builder import ExtSummarizer
import py_vncorenlp
import collections

rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=cwd+'/vncorenlp')
from nltk import tokenize



def preprocess(source):
    """
    - Remove \n
    - Sentence Tokenize
    - Add [SEP] [CLS] as sentence boundary
    """
    raw_text = source.replace(".\\n", ". ")
    raw_text = raw_text.replace("\\n", ". ")
    raw_text = raw_text.replace("< br >", " ")
    sents = rdrsegmenter.word_segment(raw_text)
    substring = 'Ảnh :'
    fil_sents = []
    for sent in sents:
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


def load_text(sents, max_pos, device, token2idx):
    # token2idx = load_vocab()
    sep_vid = 2
    cls_vid = 0

    def _process_src(raw):
        src_subtoken_idxs = []
        for sent in raw:
            sent_tokens = tokenize.word_tokenize(sent)
            src_subtoken_idxs.append(0)
            for token in sent_tokens:
                try:
                    src_subtoken_idxs.append(token2idx[token])
                except Exception:
                    pass
            src_subtoken_idxs.append(2)
        # src_subtoken_idxs = [0] + src_subtoken_idxs + [2]
        src_subtoken_idxs = src_subtoken_idxs[:-1][:max_pos]
        # try:
        src_subtoken_idxs[-1] = sep_vid
        # except Exception:
        #     pass
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
    src_text = [sents]
    return src, mask_src, segs, clss, mask_cls, src_text


def test(model, input_data, max_length, block_trigram=True):
    def _get_ngrams(n, text):
        ngram_set = set()
        text_length = len(text)
        max_index_ngram_start = text_length - n
        for i in range(max_index_ngram_start + 1):
            ngram_set.add(tuple(text[i : i + n]))
        return ngram_set

    def _block_tri(c, p):
        tri_c = _get_ngrams(3, c.split())
        for s in p:
            tri_s = _get_ngrams(3, s.split())
            if len(tri_c.intersection(tri_s)) > 0:
                return True
        return False


    with torch.no_grad():
        src, mask, segs, clss, mask_cls, src_str = input_data
        sent_scores, mask = model(src, segs, clss, mask, mask_cls)
        sent_scores = sent_scores + mask.float()
        sent_scores = sent_scores.cpu().data.numpy()
        selected_ids = np.argsort(-sent_scores, 1)
        # print(selected_ids)

        _filter_id = []
        for i, idx in enumerate(selected_ids):
            _pred = []
            if len(src_str[i]) == 0:
                continue
            for j in selected_ids[i][: len(src_str[i])]:
                if j >= len(src_str[i]):
                    continue
                candidate = src_str[i][j].strip()
                if block_trigram:
                    if not _block_tri(candidate, _pred):
                        if len(candidate)<20:
                            continue
                        _pred.append(candidate)
                        _filter_id.append(j)
                else:
                    if len(candidate)<20:
                        continue
                    _pred.append(candidate)
                    _filter_id.append(j)

                if len(_pred) == max_length:
                    break
        
        pred = []
        # print(_filter_id)
        for i in sorted(_filter_id):
            candidate = src_str[0][i].strip()
            pred.append(candidate)
        if len(pred)<5:
            pred = []
            for i in sorted(selected_ids[0][:5]):
                candidate = src_str[0][i].strip()
                if len(candidate)<20:
                    continue
                pred.append(candidate)

        # for i in range(len(pred)):
        #     save_pred.write(pred[i].strip() + "\n\n")
        return pred


def summarize(raw_txt_fp, model, token2idx, max_length=5, max_pos=512):
    model.eval()
    processed_text, full_length = preprocess(raw_txt_fp)
    input_data = load_text(processed_text, max_pos, device="cuda:0", token2idx=token2idx)
    summary = test(model, input_data, max_length, block_trigram=True)
    return summary
        
