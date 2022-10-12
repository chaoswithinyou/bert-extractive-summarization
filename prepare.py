import os
import nltk
import os

cwd = os.getcwd()
nltk.download('punkt')

os.system('git clone https://github.com/chaoswithinyou/bert-extractive-summarization')
os.system('pip3 install -r /content/bert-extractive-summarization/req.txt')

import py_vncorenlp
py_vncorenlp.download_model(save_dir=cwd)


os.system('wget https://huggingface.co/vinai/phobert-base/raw/main/vocab.txt')
os.system('wget https://raw.githubusercontent.com/chaoswithinyou/PreSumm/master/src/others/added_vocab.txt')

id = "17XJo6d-yg4VDjttKCxuZY_Yrc3THF1Iz"
gdown.download(id=id, quiet=False)
