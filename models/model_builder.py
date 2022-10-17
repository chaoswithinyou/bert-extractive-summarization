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
        self.bert1 = Bert()
        self.bert2 = Bert()
        self.bert3 = Bert()
        self.bert4 = Bert()
        self.ext_layer = ExtTransformerEncoder(
            self.bert1.model.config.hidden_size, d_ff=2048, heads=8, dropout=0.2, num_inter_layers=2
        )

        # if(max_pos>256):
        #     my_pos_embeddings = nn.Embedding(max_pos+2, self.bert.model.config.hidden_size)
        #     my_pos_embeddings.weight.data[:258] = self.bert.model.embeddings.position_embeddings.weight.data
        #     my_pos_embeddings.weight.data[258:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,:].repeat(max_pos+2-258,1)
        #     self.bert.model.embeddings.position_embeddings = my_pos_embeddings

        if checkpoint is not None:
            self.load_state_dict(checkpoint['model'], strict=False)

        self.to(device)

    def forward(self, src, segs, clss, mask_src, mask_cls):
        #scr 1,lendoc
        lendoc = src[0].shape[0]
        count = 0
        isbert2 = 0
        isbert3 = 0
        isbert4 = 0
        endbert = lendoc
        numsen = 0
        for i in range(lendoc):
            count += 1
            if src[0,i] == 0:
                istart = i
            if src[0,i] == 2:
                iend = i
                numsen += 1
            if count == 256:
                count = i-istart+1
                if isbert2 == 0:
                    ibert2 = istart
                    isbert2 = 1
                elif isbert3 == 0:
                    ibert3 = istart
                    isbert3 = 1
                elif isbert4 == 0:
                    ibert4 = istart
                    isbert4 = 1
                else:
                    endbert = iend+1
                    break

        if isbert4 == 1:            
            top_vec1 = self.bert1(src[:,:ibert2], segs[:,:ibert2], mask_src[:,:ibert2])
            top_vec2 = self.bert2(src[:,ibert2:ibert3], segs[:,ibert2:ibert3], mask_src[:,ibert2:ibert3])
            top_vec3 = self.bert3(src[:,ibert3:ibert4], segs[:,ibert3:ibert4], mask_src[:,ibert3:ibert4])
            top_vec4 = self.bert4(src[:,ibert4:endbert], segs[:,ibert4:endbert], mask_src[:,ibert4:endbert])
            top_vec = torch.cat((top_vec1,top_vec2,top_vec3,top_vec4),1)
        elif isbert3 == 1:
            top_vec1 = self.bert1(src[:,:ibert2], segs[:,:ibert2], mask_src[:,:ibert2])
            top_vec2 = self.bert2(src[:,ibert2:ibert3], segs[:,ibert2:ibert3], mask_src[:,ibert2:ibert3])
            top_vec3 = self.bert3(src[:,ibert3:], segs[:,ibert3:], mask_src[:,ibert3:])
            top_vec = torch.cat((top_vec1,top_vec2,top_vec3),1)
        elif isbert2 == 1:
            top_vec1 = self.bert1(src[:,:ibert2], segs[:,:ibert2], mask_src[:,:ibert2])
            top_vec2 = self.bert2(src[:,ibert2:], segs[:,ibert2:], mask_src[:,ibert2:])
            top_vec = torch.cat((top_vec1,top_vec2),1)
        else:
            top_vec1 = self.bert1(src, segs, mask_src)
            top_vec = top_vec1
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss[:,:numsen]]
        sents_vec = sents_vec * mask_cls[:, :numsen, None].float()
        sent_scores = self.ext_layer(sents_vec, mask_cls[:,:numsen]).squeeze(-1)
        return sent_scores, mask_cls[:,:numsen]
