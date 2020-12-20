import torch
import torch.nn as nn

# import sys
# sys.path.append('../')
# from dataset import generate_data
# from encoder import GraphAttentionEncoder
# from decoder import DecoderCell

from .encoder import GraphAttentionEncoder
from .decoder import DecoderCell


class AttentionModel(nn.Module):
	
	def __init__(self, embed_dim = 128, n_encode_layers = 3, n_heads = 8, tanh_clipping = 10., FF_hidden = 512):
		super().__init__()
		
		self.Encoder = GraphAttentionEncoder(embed_dim, n_heads, n_encode_layers, FF_hidden)
		self.Decoder = DecoderCell(embed_dim, n_heads, tanh_clipping)

	def forward(self, x, return_pi = False, decode_type = 'greedy'):
		encoder_output = self.Encoder(x)
		decoder_output = self.Decoder(x, encoder_output, return_pi = return_pi, decode_type = decode_type)
		if return_pi:
			cost, ll, pi = decoder_output
			return cost, ll, pi
		cost, ll = decoder_output
		return cost, ll
		
if __name__ == '__main__':
	
	model = AttentionModel()
	model.train()
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	data = generate_data(device, batch = 5, n_car = 15, n_depot = 1, n_customer = 20, seed = None)
	return_pi = False
	output = model(data, decode_type = 'sampling', return_pi = return_pi)
	if return_pi:
		cost, ll, pi = output
		print('\ncost: ', cost.size(), cost)
		print('\nll: ', ll.size(), ll)
		print('\npi: ', pi.size(), pi)
	else:
		print(output[0])# cost: (batch)
		print(output[1])# ll: (batch)

	cnt = 0
	for k, v in model.state_dict().items():
		print(k, v.size(), torch.numel(v))
		cnt += torch.numel(v)
	print('total parameters:', cnt)

	# output[1].mean().backward()
	# print('grad: ', model.Decoder.Wk1.weight.grad[0][0])
	# https://github.com/wouterkool/attention-learn-to-route/blob/master/train.py