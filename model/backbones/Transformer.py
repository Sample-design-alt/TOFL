import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from model.layers.SelfAttention_Family import FullAttention, AttentionLayer
from model.layers.Embed import DataEmbedding
import numpy as np


class transformer(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    modified from https://github.com/thuml/Time-Series-Library/blob/main/models/Transformer.py
    """

    def __init__(self, configs):
        super(transformer, self).__init__()
        self.task_name = configs['task_name']
        self.output_attention = configs['model_params']['output_attention']
        # Embedding
        self.enc_embedding = DataEmbedding(configs['model_params']['enc_in'], configs['model_params']['d_model'],
                                           configs['model_params']['embed'], configs['model_params']['freq'],
                                           configs['model_params']['dropout'])
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs['model_params']['factor'],
                                      attention_dropout=configs['model_params']['dropout'],
                                      output_attention=configs['model_params']['output_attention']),
                        configs['model_params']['d_model'], configs['model_params']['n_heads']),
                    configs['model_params']['d_model'],
                    configs['model_params']['d_ff'],
                    dropout=configs['model_params']['dropout'],
                    activation=configs['model_params']['activation']
                ) for l in range(configs['model_params']['e_layers'])
            ],
            norm_layer=torch.nn.LayerNorm(configs['model_params']['d_model'])
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.dec_embedding = DataEmbedding(configs['model_params']['dec_in'], configs['model_params']['d_model'],
                                               configs['model_params']['embed'], configs['model_params']['freq'],
                                               configs['model_params']['dropout'])
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AttentionLayer(
                            FullAttention(True, configs['model_params']['factor'],
                                          attention_dropout=configs['model_params']['dropout'],
                                          output_attention=False),
                            configs['model_params']['d_model'], configs['model_params']['n_heads']),
                        AttentionLayer(
                            FullAttention(False, configs['model_params']['factor'],
                                          attention_dropout=configs['model_params']['dropout'],
                                          output_attention=False),
                            configs['model_params']['d_model'], configs['model_params']['n_heads']),
                        configs['model_params']['d_model'],
                        configs['model_params']['d_ff'],
                        dropout=configs['model_params']['dropout'],
                        activation=configs['model_params']['activation'],
                    )
                    for l in range(configs['model_params']['d_layers'])
                ],
                norm_layer=torch.nn.LayerNorm(configs['model_params']['d_model']),
                projection=nn.Linear(configs['model_params']['d_model'], configs['model_params']['c_out'], bias=True)
            )
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs['model_params']['d_model'], configs['model_params']['c_out'], bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs['model_params']['d_model'], configs['model_params']['c_out'], bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs['model_params']['dropout'])
            self.projection = nn.Linear(configs['model_params']['d_model'] * configs['model_params']['seq_len'],
                                        configs['model_params']['num_class'])

        self.adaptive_avg_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.flatten = nn.Flatten()

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc=None):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output).transpose(2, 1).contiguous()

        output = self.adaptive_avg_pool(output)
        output = self.flatten(output)
        output = F.normalize(output, dim=1)
        # (batch_size, seq_length * d_model)
        return output

        # output = self.projection(output)  # (batch_size, num_classes)
        # return output

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, N]
        return None
