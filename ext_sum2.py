import time
import numpy as np
import torch
from transformers import AutoTokenizer
#from nltk.tokenize import sent_tokenize
from models.model_builder import ExtSummarizer
import py_vncorenlp
import collections

rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir='/content')
from nltk import tokenize



def preprocess(source_fp):
    """
    - Remove \n
    - Sentence Tokenize
    - Add [SEP] [CLS] as sentence boundary
    """
    with open(source_fp) as source:
        raw_text = source.read().replace(".\\n", ". ")
        raw_text = raw_text.replace("\\n", ". ")
    sents = rdrsegmenter.word_segment(raw_text)
    #processed_text = " [SEP] [CLS] ".join(sents)
    return sents, len(sents)

def load_vocab():
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open('/content/added_vocab.txt', "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            token = token.split()
            vocab[token[0]] = index
            index += 1
    with open('/content/vocab.txt', "r", encoding="utf-8") as reader:
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
            sent_tokens = tokenize.word_tokenize(sent)
            for token in sent_tokens:
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


def test(model, input_data, result_path, max_length, block_trigram=True):
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

    with open(result_path, "w") as save_pred:
        with torch.no_grad():
            src, mask, segs, clss, mask_cls, src_str = input_data
            sent_scores, mask = model(src, segs, clss, mask, mask_cls)
            sent_scores = sent_scores + mask.float()
            sent_scores = sent_scores.cpu().data.numpy()
            #print(sent_scores)
            selected_ids = np.argsort(-sent_scores, 1)
            #print(selected_ids)

            pred = []
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
                            _pred.append(candidate)
                    else:
                        _pred.append(candidate)

                    if len(_pred) == max_length:
                        break

                _pred = " ".join(_pred)
                pred.append(_pred)

            for i in range(len(pred)):
                save_pred.write(pred[i].strip() + "\n")


def summarize(raw_txt_fp, result_fp, model, max_length=3, max_pos=512, return_summary=True):
    model.eval()
    processed_text, full_length = preprocess(raw_txt_fp)
    input_data = load_text(processed_text, max_pos, device="cpu")
    test(model, input_data, result_fp, max_length, block_trigram=True)
    if return_summary:
        return open(result_fp).read().strip()
        