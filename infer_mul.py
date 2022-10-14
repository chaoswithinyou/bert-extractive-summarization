import os
cwd = os.getcwd()
import torch
from models.model_builder import ExtSummarizer
from ext_sum3 import summarize
from tqdm  import tqdm
import jsonlines

# Load model
checkpoint = torch.load(cwd+'/model_final.pt', map_location='gpu')
model = ExtSummarizer(checkpoint=checkpoint, device='gpu')

sent_ids = []
# read file
with jsonlines.open("your_original_data.jsonl") as myfile:
    for i, article in tqdm(enumerate(myfile)):
        try:
            # Run summarization
            top_n_ids = summarize(article['text'], model, top_n=10, max_pos=1024)
            sent_ids.append({"sent_id":top_n_ids})
        except Exception:
            pass
 
with jsonlines.open("your_index.jsonl", "w") as outfile:
    outfile.write_all(sent_ids)