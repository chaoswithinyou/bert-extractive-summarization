import torch
from models.model_builder import ExtSummarizer
from ext_sum import summarize

# Load model
model_type = 'phobert'
checkpoint = torch.load('/content/MODEL_PATH/model_step_50000.pt', map_location='cpu')
model = ExtSummarizer(checkpoint=checkpoint, bert_type=model_type, device='cpu')

# Run summarization
input_fp = '/content/bert-extractive-summarization/raw_data/input.txt'
result_fp = '/content/bert-extractive-summarization/results/summary.txt'
summary = summarize(input_fp, result_fp, model, max_length=5)
print(summary)