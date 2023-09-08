import torch

from torch import nn
from timm.models.layers import trunc_normal_
from tool.config import Configuration


class ControlPredict(nn.Module):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.cfg = cfg
        self.pad_idx = self.cfg.token_nums - 1

        self.embedding = nn.Embedding(self.cfg.token_nums, self.cfg.tf_de_dim)
        self.pos_drop = nn.Dropout(self.cfg.tf_de_dropout)
        self.pos_embed = nn.Parameter(torch.randn(1, self.cfg.tf_de_tgt_dim - 1, self.cfg.tf_de_dim) * .02)

        tf_layer = nn.TransformerDecoderLayer(d_model=self.cfg.tf_de_dim, nhead=self.cfg.tf_de_heads)
        self.tf_decoder = nn.TransformerDecoder(tf_layer, num_layers=self.cfg.tf_de_layers)
        self.output = nn.Linear(self.cfg.tf_de_dim, self.cfg.token_nums)

        self.init_weights()

    def init_weights(self):
        for name, p in self.named_parameters():
            if 'pos_embed' in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        trunc_normal_(self.pos_embed, std=.02)

    def create_mask(self, tgt):
        # tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]).cuda()
        tgt_mask = (torch.triu(torch.ones((tgt.shape[1], tgt.shape[1]), device=self.cfg.device)) == 1).transpose(0, 1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask == 0, float('-inf')).masked_fill(tgt_mask == 1, float(0.0))
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
        return pred_controls

    def forward(self, encoder_out, tgt):
        tgt = tgt[:, :-1]
        tgt_mask, tgt_padding_mask = self.create_mask(tgt)

        tgt_embedding = self.embedding(tgt)
        tgt_embedding = self.pos_drop(tgt_embedding + self.pos_embed)

        pred_controls = self.decoder(encoder_out, tgt_embedding, tgt_mask, tgt_padding_mask)
        pred_controls = self.output(pred_controls)
        return pred_controls

    def predict(self, encoder_out, tgt):
        length = tgt.size(1)
        padding = torch.ones(tgt.size(0), self.cfg.tf_de_tgt_dim - length - 1).fill_(self.pad_idx).long().to('cuda')
        tgt = torch.cat([tgt, padding], dim=1)

        tgt_mask, tgt_padding_mask = self.create_mask(tgt)

        tgt_embedding = self.embedding(tgt)
        tgt_embedding = tgt_embedding + self.pos_embed

        pred_controls = self.decoder(encoder_out, tgt_embedding, tgt_mask, tgt_padding_mask)
        pred_controls = self.output(pred_controls)[:, length - 1, :]

        pred_controls = torch.softmax(pred_controls, dim=-1)
        pred_controls = pred_controls.argmax(dim=-1).view(-1, 1)
        return pred_controls
