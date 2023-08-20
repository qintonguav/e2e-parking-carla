import torch
from torch import nn
from tool.config import Configuration
from timm.models.layers import trunc_normal_


class ControlPredict(nn.Module):
    def __init__(self, cfg: Configuration):
        super(ControlPredict, self).__init__()
        self.cfg = cfg
        self.pad_idx = self.cfg.pad_idx

        self.embedding = nn.Embedding(self.cfg.token_nums, self.cfg.tf_de_dim)
        self.pos_drop = nn.Dropout(self.cfg.tf_de_dropout)
        self.pos_embed = nn.Parameter(self.cfg.token_nums, self.cfg.tf_de_dim)

        self.tf_layer = nn.TransformerDecoderLayer(d_model=self.cfg.tf_de_dim, nhead=self.cfg.tf_de_heads)
        self.tf_decoder = nn.TransformerDecoder(self.tf_layer, num_layers=self.cfg.tf_de_layers)
        self.output = nn.Linear(self.cfg.tf_de_dim, self.cfg.token_nums)

    def init_weight(self):
        for name, p in self.named_parameters():
            if 'pos_embed' in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform(p)
        trunc_normal_(self.pos_embed, std=.02)

    def create_mask(self, tgt):
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1])
        tgt_padding_mask = (tgt == self.pad_idx)
        return tgt_mask, tgt_padding_mask

    def decoder(self, encoder_out, tgt_embedding, tgt_mask, tgt_padding_mask):
        encoder_out = encoder_out.transpose(0, 1)
        tgt_embedding = tgt_embedding.transpose(0, 1)
        pred_controls = self.tf_decoder(tgt=tgt_embedding,
                                        memory=encoder_out,
                                        tgt_mask=tgt_mask,
                                        tgt_key_padding_mask=tgt_padding_mask)
        pred_controls = pred_controls.transpose(0, 1)
        pred_controls = self.output(pred_controls)
        return pred_controls

    def forward(self, encoder_out, tgt):
        tgt = tgt[:, :-1]
        tgt_mask, tgt_padding_mask = self.create_mask(tgt)

        tgt_embedding = self.embedding(tgt)
        tgt_embedding = self.pos_drop(tgt_embedding + self.pos_embed)

        pred_controls = self.decoder(encoder_out, tgt_embedding, tgt_mask, tgt_padding_mask)
        return pred_controls

    def predict(self, encoder_out, tgt):
        length = tgt.shape[1]
        padding = torch.ones(tgt.shape[0], self.cfg.tf_de_dim - length - 1).fill_(self.pad_idx).long().cuda()
        tgt = torch.cat([tgt, padding], dim=1)

        tgt_mask, tgt_padding_mask = self.create_mask(tgt)

        tgt_embedding = self.embedding(tgt)
        tgt_embedding = tgt_embedding + self.pos_embed

        pred_controls = self.decoder(encoder_out, tgt_embedding, tgt_mask, tgt_padding_mask)

        pred_controls = pred_controls.softmax(dim=-1)
        pred_controls = pred_controls.argmax(dim=-1)
        return pred_controls
