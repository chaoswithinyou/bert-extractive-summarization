import os
cwd = os.getcwd()
import torch
from models.model_builder import ExtSummarizer
from ext_sum_json import summarize
from tqdm  import tqdm
import json
from ext_sum_json import load_vocab

# Load model
checkpoint = torch.load(cwd+'/model_step_50000.pt', map_location='cuda:0')
model = ExtSummarizer(checkpoint=checkpoint, device='cuda:0')
vocabdict = load_vocab()

with open('/content/drive/MyDrive/NLP/desc.json', 'r') as myfile:
    data=myfile.read()
obj = json.loads(data)
articles =[obj['hits']['hits'][i]['_source'] for i in range(len(obj['hits']['hits']))]
for i in tqdm(range(len(articles))):
    content =  articles[i]['content']
    try:
        summary = summarize(content, model=model, max_length=5, token2idx=vocabdict)
    except:
        articles[i]['imp_sents'] = []
        continue
    # print(summary)
    articles[i]['imp_sents'] = [sum.replace('_',' ') for sum in summary]


with open('extsum_listsen.json', 'w', encoding='utf8') as json_file:
    json.dump(articles, json_file, ensure_ascii=False, indent=4)