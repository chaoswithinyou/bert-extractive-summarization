import torch
import torch.nn as nn
from transformers import AutoModel
from models.encoder import ExtTransformerEncoder

class Bert(nn.Module):
    def __init__(self):
        super(Bert, self).__init__()
        self.model = AutoModel.from_pretrained('vinai/phobert-base')
        # Update config to finetune token type embeddings
        self.model.config.type_vocab_size = 2 

        # Create a new Embeddings layer, with 2 possible segments IDs instead of 1
        self.model.embeddings.token_type_embeddings = nn.Embedding(2, self.model.config.hidden_size)
                        
        # Initialize it
        self.model.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0, std=self.model.config.initializer_range)

    def forward(self, x, segs, mask):
        top_vec = self.model(x, attention_mask=mask, token_type_ids=segs).last_hidden_state
        return top_vec


class ExtSummarizer(nn.Module):
    def __init__(self, device, checkpoint=None, max_pos=512):
        super().__init__()
        self.device = device
        self.bert = Bert()
        self.ext_layer = ExtTransformerEncoder(
            self.bert.model.config.hidden_size, d_ff=2048, heads=8, dropout=0.2, num_inter_layers=2
        )

        if(max_pos>256):
            my_pos_embeddings = nn.Embedding(max_pos+2, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:258] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[258:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(max_pos+2-258,1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=False)

        self.to(device)

    def forward(self, src, segs, clss, mask_src, mask_cls):
        top_vec = self.bert(src, segs, mask_src)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
        return sent_scores, mask_cls
