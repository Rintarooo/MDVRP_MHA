import torch
import torch.nn as nn
import math

class DotProductAttention(nn.Module):
	def __init__(self, clip = None, return_logits = False, head_depth = 16, **kwargs):
		super().__init__(**kwargs)
		self.clip = clip
		self.return_logits = return_logits
		self.inf = 1e+10 
		self.scale = math.sqrt(head_depth)
		self.tanh = nn.Tanh

	def forward(self, x, mask = None):
		""" Q: (batch, n_heads, q_seq(=n_nodes or =1), head_depth)
			K: (batch, n_heads, k_seq(=n_nodes), head_depth)
			logits: (batch, n_heads, q_seq(this could be 1), k_seq)
			mask: (batch, n_nodes, 1), e.g. tf.Tensor([[ True], [ True], [False]])
			mask[:,None,None,:,0]: (batch, 1, 1, n_nodes) ==> broadcast depending on logits shape
			[True] -> [1 * -np.inf], [False] -> [logits]
			K.transpose(-1,-2).size() == K.permute(0,1,-1,-2).size()
		"""
		Q, K, V = x
		logits = torch.matmul(Q, K.transpose(-1,-2)) / self.scale
		if self.clip is not None:
			logits = self.clip * torch.tanh(logits)
			# logits = self.clip * self.tanh(logits)
			
		if self.return_logits:
			if mask is not None:
				# ~ print('mask.size():', mask.size())
				# ~ print('logits.size():', logits.size())
				return logits.masked_fill(mask.transpose(-1,-2) == True, -self.inf)
			return logits

		if mask is not None:
			# ~ print('mask.size():', mask.size())
			# ~ print('logits.size():', logits.size())
			# print('mask: ', mask[:,None,:,:].squeeze(-1).repeat(1,logits.size(1),1,mask.size(1)).size())
			# print('mask expand:', mask[:,None,:,:,:].repeat(1,logits.size(1),1,1,1).transpose(-1,-2).size())
			# logits = logits.masked_fill(mask[:,None,None,:,0].repeat(1,logits.size(1),1,1) == True, -self.inf)
			# logits = logits.masked_fill(mask[:,None,:,:].squeeze(-1).repeat(1,logits.size(1),1,mask.size(1)) == True, -self.inf)
			logits = logits.masked_fill(mask[:,None,:,:,:].repeat(1,logits.size(1),1,1,1).transpose(-1,-2) == True, -self.inf)
			
		probs = torch.softmax(logits, dim = -1)
		return torch.matmul(probs, V)

class MultiHeadAttention(nn.Module):
	def __init__(self, n_heads = 8, embed_dim = 128, clip = None, return_logits = None, need_W = None):
		super().__init__()
		self.n_heads = n_heads
		self.embed_dim = embed_dim
		self.head_depth = self.embed_dim // self.n_heads
		if self.embed_dim % self.n_heads != 0:
			raise ValueError("embed_dim = n_heads * head_depth")
		
		self.need_W = need_W 
		self.attention = DotProductAttention(clip = clip, return_logits = return_logits, head_depth = self.head_depth)
		if self.need_W:
			self.Wk = nn.Linear(embed_dim, embed_dim, bias = False)
			self.Wv = nn.Linear(embed_dim, embed_dim, bias = False)
			self.Wq = nn.Linear(embed_dim, embed_dim, bias = False)
			self.Wout = nn.Linear(embed_dim, embed_dim, bias = False)
		self.init_parameters()
	
	def init_parameters(self):
		for name, param in self.named_parameters():
			if name == 'Wout.weight':
				stdv = 1. / math.sqrt(param.size(-1))
			elif name in ['Wk.weight', 'Wv.weight', 'Wq.weight']:
				stdv = 1. / math.sqrt(self.head_depth)
			else:
				raise ValueError
			param.data.uniform_(-stdv, stdv)

	def split_heads(self, T):
		""" https://qiita.com/halhorn/items/c91497522be27bde17ce
			T: (batch, n_car, n_nodes(or 1), self.embed_dim)
			T reshaped: (batch, n_car, n_nodes(or 1), self.n_heads, self.head_depth)
			return: (batch, self.n_heads, n_car, n_nodes(or 1), self.head_depth)
			
			https://raishi12.hatenablog.com/entry/2020/04/20/221905
		"""
		shape = T.size()[:-1] + (self.n_heads, self.head_depth)
		T = T.view(*shape)
		return T.permute(0,3,1,2,4)

	def combine_heads(self, T):
		""" T: (batch, self.n_heads, n_car, n_nodes(or 1), self.head_depth)
			T transposed: (batch, n_car, n_nodes(or 1), self.n_heads, self.head_depth)
			return: (batch, n_car, n_nodes(or 1), self.embed_dim)
		"""
		T = T.permute(0,2,3,1,4).contiguous()
		shape = T.size()[:3] + (self.embed_dim, )
		return T.view(*shape)

	def forward(self, x, mask = None):
		"""	q, k, v = x
			encoder arg x: [x, x, x]
			shape of q: (batch, n_nodes, embed_dim)
			output[0] - output[h_heads-1]: (batch, n_nodes, head_depth)
			--> concat output: (batch, n_nodes, head_depth * h_heads)
			return output: (batch, n_nodes, embed_dim)
		"""
		Q, K, V = x
		if self.need_W:
			Q, K, V = self.Wq(Q), self.Wk(K), self.Wv(V)
		Q, K, V = list(map(self.split_heads, [Q, K, V]))
		output = self.attention([Q, K, V], mask = mask)
		output = self.combine_heads(output)
		if self.need_W:
			return self.Wout(output)
		return output

if __name__ == '__main__':
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	mha = MultiHeadAttention(n_heads = 8, embed_dim = 128, need_W = False)
	batch, n_car, n_nodes, embed_dim = 5, 15, 21, 128
	mask = torch.zeros((batch, n_car, n_nodes, 1), dtype = torch.bool, device = device)
	# mask = None
	Q = torch.randn((batch, n_car, 1, embed_dim), dtype = torch.float, device = device)
	K = torch.randn((batch, n_car, n_nodes, embed_dim), dtype = torch.float, device = device)
	V = torch.randn((batch, n_car, n_nodes, embed_dim), dtype = torch.float, device = device)
	output = mha([Q,K,V], mask = mask)
	print('output.size()', output.size())# (batch, n_car, 1, embed_dim)

	sha = DotProductAttention(clip = 10., return_logits = True, head_depth = embed_dim)
	output = sha([output,K,None], mask = mask)
	print('output.size()', output.size())
	output = output.squeeze(2).view(batch, -1)
	print('output.size()', output.size())

	logp = torch.log_softmax(output, dim = -1)	
	# probs = torch.softmax(output, dim = 1)
	idx = torch.multinomial(logp.exp(), num_samples = 1)	
	print(idx)
	print('next car:', idx // n_nodes)
	print('next node:', idx % n_nodes)
	# print(a.size(), a[0])
	# idx = torch.argmax(a, dim = 1)
	# print(idx.size(), idx)


