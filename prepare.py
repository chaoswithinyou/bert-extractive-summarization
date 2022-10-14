import os

cwd = os.getcwd()

os.system('mkdir vncorenlp')
install_req = 'pip3 install -r '+cwd+'/req.txt'
os.system(install_req)

import py_vncorenlp
import gdown
import nltk
py_vncorenlp.download_model(save_dir=cwd+'/vncorenlp')
nltk.download('punkt')

# os.system('wget https://huggingface.co/vinai/phobert-base/raw/main/vocab.txt')
# os.system('wget https://raw.githubusercontent.com/chaoswithinyou/PreSumm/master/src/others/added_vocab.txt')

id = "1_2lF2lfMqAziPEeefCCBu5HAaQ63f38J"
gdown.download(id=id, quiet=False)
