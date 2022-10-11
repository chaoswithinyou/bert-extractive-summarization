import torch
from models.model_builder import ExtSummarizer
from ext_sum2 import summarize
import argparse
from newsplease import NewsPlease
import time
import os

file_name_postfix = str(int(time.time()))

input_fp = '/content/bert-extractive-summarization/raw_data/input_'+file_name_postfix+'.txt'
result_fp = '/content/bert-extractive-summarization/results/summary_'+file_name_postfix+'.txt'
 
# Initialize parser
parser = argparse.ArgumentParser()
 
# Adding optional argument
parser.add_argument("-url", help = "Insert article's url")
 
# Read arguments from command line
args = parser.parse_args()

article = NewsPlease.from_url(args.url)
with open(input_fp, 'w') as f:
    f.write(article.maintext)

# Load model
model_type = 'phobert'
checkpoint = torch.load('/content/drive/MyDrive/NLP/model_step_50000.pt', map_location='cpu')
model = ExtSummarizer(checkpoint=checkpoint, device='cpu')

# Run summarization

summary = summarize(input_fp, result_fp, model, max_length=3)
summary = summary.replace('_',' ')
print(article.description+'\n'+summary)

try:
    os.remove(input_fp)
except OSError:
    pass
try:
    os.remove(result_fp)
except OSError:
    pass