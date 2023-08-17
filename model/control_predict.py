from torch import nn
from tool.config import Configuration


class ControlPredict(nn.Module):
    def __init__(self, cfg: Configuration):
        super(ControlPredict, self).__init__()
        self.cfg = cfg
        self.pad_idx = self.cfg.pad_idx

        self.embedding = nn.Embedding(self.cfg.token_nums, self.cfg.tf_de_dim)
        self.pos_drop = nn.Dropout(self.cfg.tf_de_dropout)
        self.pos_emb = nn.Parameter(self.cfg.token_num, self.cfg.tf_de_dim)

        self.tf_layer = nn.TransformerDecoderLayer(d_model=self.cfg.tf_de_dim, nhead=self.cfg.tf_de_heads)
        self.tf_decoder = nn.TransformerDecoder(self.tf_layer, num_layers=self.cfg.tf_de_layers)
        self.output = nn.Linear(self.cfg.tf_de_dim, self.cfg.token_nums)

    def init_weight(self):
        pass

    def create_mask(self, tgt):
        pass

    def decoder(self, encoder_out, tgt):
        tgt_mask = self.create_mask(tgt)
        tgt_padding_mask = (tgt == self.pad_idx)

        tgt_embedding = self.embedding(tgt)
        tgt_embedding = self.pos_drop(tgt_embedding + self.pos_emb)

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
        pred_controls = self.decoder(encoder_out, tgt)
        return pred_controls

    def predict(self, encoder_out, tgt):
        tgt = tgt[:, :-1]
        pred_controls = self.decoder(encoder_out, tgt)
        pred_controls = pred_controls.softmax(dim=-1)
        pred_controls = pred_controls.argmax(dim=-1)
        return pred_controls
