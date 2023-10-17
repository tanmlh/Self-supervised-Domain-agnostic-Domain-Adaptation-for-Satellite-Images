''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
from .Layers import EncoderLayer, DecoderLayer
# from models.modules import metrics, components
import torch.nn.functional as F
import pdb

__author__ = "Yu-Hsiang Huang"

def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).byte()
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1, n_position=200):

        super().__init__()

        # self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        # no need to embed src_seq!
        enc_output = self.dropout(self.position_enc(src_seq))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v,
            d_model, d_inner, pad_idx, n_position=200, dropout=0.1):

        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_output = self.dropout(self.position_enc(self.trg_word_emb(trg_seq)))
        dec_output = self.layer_norm(dec_output)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask)
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output,

class ConvexNet(nn.Module):

    def __init__(self, num_nodes, channels):
        self.linears = [nn.Linear(channels[0], channels[1]) for _ in range(num_nodes)]
        self.env_linear = nn.Linear(channels[0], num_nodes)

    def forward(self, seq_features):
        # seq_features: (B, num_seq, num_features)
        temp = F.avg_pool2d(seq_features)



class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(self, net_conf):
        super(Transformer, self).__init__()

        d_inner=2048
        d_k=64
        d_v=64
        dropout=0.1
        n_position=200
        trg_emb_prj_weight_sharing=True

        d_word_vec=net_conf['transformer']['num_channels'] if 'num_channels' in net_conf['transformer'] else 512
        d_model=d_word_vec
        n_trg_vocab = net_conf['transformer']['num_classes']
        trg_pad_idx = net_conf['transformer']['pad_idx']
        linear_type = net_conf['transformer']['linear_type']
        n_layers = net_conf['transformer']['num_layers']
        n_head = net_conf['transformer']['num_heads']

        use_norm = net_conf['transformer']['use_norm'] if 'use_norm' in net_conf['transformer'] else False
        scale = net_conf['transformer']['scale'] if 'scale' in net_conf['transformer'] else 40
        margin = net_conf['transformer']['margin'] if 'margin' in net_conf['transformer'] else 0.1
        use_softmax = net_conf['transformer']['use_softmax'] if 'use_softmax' in net_conf['transformer'] else True
        use_cos_norm = net_conf['env_net']['use_cos_norm'] if 'use_cos_norm' in net_conf else False


        self.trg_pad_idx = trg_pad_idx

        self.encoder = Encoder(
            n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.decoder = Decoder(
            n_trg_vocab=n_trg_vocab, n_position=n_position,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=trg_pad_idx, dropout=dropout)

        if linear_type == 'normal':
            self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        elif linear_type == 'arc_face':
            self.trg_word_prj = metrics.ArcMarginProduct(d_model, n_trg_vocab)

        elif linear_type == 'env_net':
            self.trg_word_prj = components.EnvNet(net_conf['env_net'])
            self.env_metric_type = net_conf['env_net']['metric_type'] if 'metric_type' in net_conf['env_net'] else 'tfm_cosine_face'

            if self.env_metric_type == 'tfm_cosine_face':
                self.trg_word_prj2 = metrics.AddMarginProduct(net_conf['metric'])
            elif self.env_metric_type == 'cosine_face':
                self.trg_word_prj2 = metrics.OutOfDistributionProduct(net_conf['metric'])
            else:
                raise ValueError

        elif linear_type == 'cosine_face':
            self.trg_word_prj = metrics.OutOfDistributionProduct(net_conf['metric'])

        elif linear_type == 'tfm_cosine_face':
            self.trg_word_prj = metrics.AddMarginProduct(net_conf['metric'])

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        self.x_logit_scale = 1.
        if trg_emb_prj_weight_sharing and linear_type == 'normal':
            # Share the weight between target word embedding & last dense layer
            self.trg_word_prj.weight = self.decoder.trg_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)

        # if emb_src_trg_weight_sharing:
        #     self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight
        self.linear_type = linear_type
        self.d_model = d_model


    def forward(self, src_seq, trg_seq, label, phase='train'):
        # src_seq: (B, H, num_features)
        # trg_seq: (B, label_len)
        B, label_len = trg_seq.shape

        trg_mask = get_pad_mask(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        src_mask = None
        enc_output = src_seq
        env_att = None
        # enc_output, *_ = self.encoder(src_seq, src_mask)

        dec_output, self_att, enc_att = self.decoder(trg_seq, trg_mask, enc_output, src_mask, return_attns=True)

        if self.linear_type == 'normal':
            seq_logit = self.trg_word_prj(dec_output) * self.x_logit_scale
            pred_prob_tfm = F.softmax(seq_logit, dim=-1)

        elif self.linear_type == 'arc_face':
            seq_logit = self.trg_word_prj(dec_output.view(-1, self.d_model), trg_seq.view(-1))
            seq_logit = seq_logit.view(B, label_len, -1)
            pred_prob_tfm = F.softmax(seq_logit, dim=-1)

        elif self.linear_type == 'env_net':
            seq_logit, env_att = self.trg_word_prj(src_seq, dec_output)

            if self.env_metric_type == 'tfm_cosine_face':
                seq_logit = self.trg_word_prj2(seq_logit, label, phase=phase)

            elif self.env_metric_type == 'cosine_face':
                seq_logit = self.trg_word_prj2(seq_logit)

            else:
                raise ValueError

            pred_prob_tfm = F.softmax(seq_logit, dim=-1)

        elif self.linear_type == 'cosine_face':
            seq_logit = self.trg_word_prj(dec_output)
            pred_prob_tfm = F.softmax(seq_logit, dim=-1)

        elif self.linear_type == 'tfm_cosine_face':
            pred_prob_tfm = self.trg_word_prj(dec_output, label, phase=phase)
            pred_prob_tfm = F.softmax(pred_prob_tfm, dim=-1)

        out = {}
        out['pred_prob_tfm'] = pred_prob_tfm
        out['enc_att'] = enc_att
        out['dec_out'] = dec_output
        out['env_att'] = env_att

        return out
