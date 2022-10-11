# Extractive Summarization with BERT

## Setup
```sh
git clone https://github.com/chriskhanhtran/bert-extractive-summarization.git
cd bert-extractive-summarization
pip install -r requirements.txt
```

Download pretrained checkpoints:

```sh
wget -O "checkpoints/bertbase_ext.pt" "https://www.googleapis.com/drive/v3/files/1t27zkFMUnuqRcsqf2fh8F1RwaqFoMw5e?alt=media&key=AIzaSyCmo6sAQ37OK8DK4wnT94PoLx5lx-7VTDE"
wget -O "checkpoints/distilbert_ext.pt" "https://www.googleapis.com/drive/v3/files/1WxU7cHECfYaU32oTM0JByTRGS5f6SYEF?alt=media&key=AIzaSyCmo6sAQ37OK8DK4wnT94PoLx5lx-7VTDE"
wget -O "checkpoints/mobilebert_ext.pt" "https://www.googleapis.com/drive/v3/files/1umMOXoueo38zID_AKFSIOGxG9XjS5hDC?alt=media&key=AIzaSyCmo6sAQ37OK8DK4wnT94PoLx5lx-7VTDE"
```

## Usage
[![](https://img.shields.io/badge/Colab-Run_in_Google_Colab-blue?logo=Google&logoColor=FDBA18)](https://colab.research.google.com/drive/1hwpYC-AU6C_nwuM_N5ynOShXIRGv-U51#scrollTo=KizhzOxVOjaN)
```python
import torch
from models.model_builder import ExtSummarizer
from ext_sum import summarize

# Load model
model_type = 'mobilebert' #@param ['bertbase', 'distilbert', 'mobilebert']
checkpoint = torch.load(f'checkpoints/{model_type}_ext.pt', map_location='cpu')
model = ExtSummarizer(checkpoint=checkpoint, bert_type=model_type, device='cpu')

# Run summarization
input_fp = 'raw_data/input.txt'
result_fp = 'results/summary.txt'
summary = summarize(input_fp, result_fp, model, max_length=3)
print(summary)
```

## References
- [1] [PreSumm:  Text Summarization with Pretrained Encoders](https://github.com/nlpyang/PreSumm)
- [2] [Extractive Summarization with BERT](https://github.com/chriskhanhtran/bert-extractive-summarization/)
