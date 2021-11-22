import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Transformer(nn.Module):
	def __init__(
		self,
		num_tokens_inp,
		num_tokens_out,
		dim_model,
		num_heads,
		num_encoder_layers,
		num_decoder_layers,
		max_len,
		dropout_p
	):
		super().__init__()

		self.model_type = "Transformer"
		self.dim_model = dim_model

		# Layers
		self.positional_encoder = PositionalEncoding(dim_model, dropout_p, max_len=max_len)
		self.embedding1 = nn.Embedding(num_tokens_inp, dim_model)
		self.embedding2 = nn.Embedding(num_tokens_out, dim_model)
		self.transformer = nn.Transformer(
			d_model=dim_model,
			nhead=num_heads,
			num_encoder_layers=num_encoder_layers,
			num_decoder_layers=num_decoder_layers,
			dropout=dropout_p
		)

		self.out = nn.Linear(dim_model, num_tokens_out)

	def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
		# Why multiply?? => otherwise, position encoding will dominate the embeddings
		src = self.embedding1(src) * math.sqrt(self.dim_model)
		tgt = self.embedding2(tgt) * math.sqrt(self.dim_model)

		src = self.positional_encoder(src)
		tgt = self.positional_encoder(tgt)

		src = src.permute(1, 0, 2)
		tgt = tgt.permute(1, 0, 2)

		transformer_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
		out = self.out(transformer_out)

		return out

	def get_tgt_mask(self, size):
		mask = torch.tril(torch.ones(size, size) == 1)
		mask = mask.float()
		mask = mask.masked_fill(mask == 0, float('-inf'))
		mask = mask.masked_fill(mask == 1, float(0.0))

		return mask

	def create_pad_mask(self, matrix, pad_token):
		return (matrix == pad_token)


class PositionalEncoding(nn.Module):
	def __init__(self, dim_model, dropout_p, max_len):
		super().__init__()

		self.dropout = nn.Dropout(dropout_p)

		pos_encoding = torch.zeros(max_len, dim_model)
		positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
		division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)

		pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
		pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

		pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)

		self.register_buffer("pos_encoding", pos_encoding)

	def forward(self, token_embedding):
		return self.dropout(self.pos_encoding[:token_embedding.size(0), :] + token_embedding)

