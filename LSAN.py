from transformer import *
import torch.nn as nn
import torch
from torch.autograd import Function

class LSAN(nn.Module):
    def __init__(self, dict_len, embedding_dim,  transformer_hidden, attn_heads, transformer_dropout, transformer_layers):
        super().__init__()

        self.embeds = nn.Embedding(dict_len+1, embedding_dim, padding_idx = dict_len)
        classifier_dropout = 0.1
      

        attn_dropout = 0.1
        self.MATT = nn.Sequential(nn.Linear(embedding_dim,int(embedding_dim/4)),
                                        nn.ReLU(),
                                        nn.Dropout(attn_dropout),
                                        nn.Linear(int(embedding_dim/4), int(embedding_dim/8)),
                                        nn.ReLU(),
                                        nn.Dropout(attn_dropout),
                                        nn.Linear(int(embedding_dim/8), 1))

        visit_attn_dropout = 0.1
        visit_ATT_dim = 2 * embedding_dim
        self.visit_ATT = nn.Sequential(nn.Linear(visit_ATT_dim, int(visit_ATT_dim /4)),
                                        nn.ReLU(),
                                        nn.Dropout(visit_attn_dropout),
                                        nn.Linear(int(visit_ATT_dim/4), int(visit_ATT_dim /8)),
                                        nn.ReLU(),
                                        nn.Dropout(visit_attn_dropout),
                                        nn.Linear(int(visit_ATT_dim /8), 1))

        self.Classifier = nn.Linear(visit_ATT_dim,  1)

        self.position = PositionalEmbedding(d_model=embedding_dim, max_len = 1024)            


        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(transformer_hidden, attn_heads, transformer_hidden * 4, transformer_dropout) for _ in range(transformer_layers)])
       

        self.local_conv_layer = nn.Conv1d(in_channels = embedding_dim, out_channels = embedding_dim, kernel_size = 3, padding = 1)
         
    def forward(self, batch_input, lstm=True):
        input_embedding = self.embeds(batch_input)
        feat_size = input_embedding.shape[3]
        batch_size = batch_input.shape[0]
        visit_size = batch_input.shape[1]
        disease_size = batch_input.shape[2]
               
        input_embedding = input_embedding.view(batch_size*visit_size, disease_size, feat_size)
        
        # Framework
        attn_weight = F.softmax(self.MATT(input_embedding), dim = 1)
        diag_result_att = torch.matmul(attn_weight.permute(0,2,1), input_embedding).squeeze(1) 
        diag_result_att = diag_result_att.view(batch_size, visit_size, feat_size)
       

        positional_embedding = self.position(diag_result_att)
        transformer_input = diag_result_att + positional_embedding


        mask = None
        for transformer in self.transformer_blocks:
            transformer_embedding = transformer.forward(transformer_input, mask)
 
        local_conv_feat = self.local_conv_layer(diag_result_att.permute(0,2,1))
        concat_feat = torch.cat((transformer_embedding, local_conv_feat.permute(0,2,1)), dim = 2)
        visit_attn_weight = F.softmax(self.visit_ATT(concat_feat), dim = 1)
        visit_result_att = torch.matmul(visit_attn_weight.permute(0,2,1), concat_feat).squeeze(1)  
        embedding_sum = visit_result_att

        # Attention Analysis
        #diagnosis_code_mask = (batch_input != 8692).type(torch.float32)
        #code_mask_sum = torch.sum(diagnosis_code_mask, dim=-1)
        #visit_mask = (code_mask_sum!=0).type(torch.float32)
        #code_attn = attn_weight.squeeze(-1).view(batch_size, visit_size, disease_size) 
        #mask_out_code = code_attn*diagnosis_code_mask
        #mask_out_code_sum = torch.sum(mask_out_code, -1).unsqueeze(-1).repeat(1,1,disease_size)
        #mask_out_code_sum = mask_out_code_sum + torch.finfo(torch.float32).eps
        #code_attn_weight = mask_out_code/mask_out_code_sum

        #visit_mask_sum = torch.sum(visit_mask, -1)
        #visit_attn = visit_attn_weight.squeeze(-1)
        #visit_attn = visit_attn*visit_mask
        #visit_attn_sum = torch.sum(visit_attn,dim = -1).unsqueeze(-1)
        #visit_attn_sum = visit_attn_sum.repeat(1, visit_size) + torch.finfo(torch.float32).eps
        #visit_attn_weight = visit_attn/visit_attn_sum

        # Prediction
        prediction_output = self.Classifier(embedding_sum)
        binary_output = nn.functional.sigmoid(prediction_output)
        return binary_output

