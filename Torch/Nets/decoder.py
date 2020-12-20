import torch
import torch.nn as nn

# from decoder_layers import MultiHeadAttention, DotProductAttention
# from decoder_utils import TopKSampler, CategoricalSampler, Env
# import sys
# sys.path.append('../')
# from dataset import generate_data

from .decoder_layers import MultiHeadAttention, DotProductAttention
from .decoder_utils import TopKSampler, CategoricalSampler, Env


class DecoderCell(nn.Module):
	def __init__(self, embed_dim = 128, n_heads = 8, clip = 10., **kwargs):
		super().__init__(**kwargs)
		
		self.Wk1 = nn.Linear(embed_dim, embed_dim, bias = False)
		self.Wv = nn.Linear(embed_dim, embed_dim, bias = False)
		self.Wk2 = nn.Linear(embed_dim, embed_dim, bias = False)
		self.Wq_fixed = nn.Linear(embed_dim, embed_dim, bias = False)
		self.Wout = nn.Linear(embed_dim, embed_dim, bias = False)
		self.Wq_step = nn.Linear(embed_dim+1, embed_dim, bias = False)
		
		self.MHA = MultiHeadAttention(n_heads = n_heads, embed_dim = embed_dim, need_W = False)
		self.SHA = DotProductAttention(clip = clip, return_logits = True, head_depth = embed_dim)
		# SHA ==> Single Head Attention, because this layer n_heads = 1 which means no need to spilt heads
		
	def compute_static(self, node_embeddings, graph_embedding):
		Q_fixed = self.Wq_fixed(graph_embedding[:,None,:])
		# self.Q_fixed = Q_fixed[:,None,:,:].repeat(1,self.env.n_car,1,1)
		K1 = self.Wk1(node_embeddings)
		# self.K1 = K1[:,None,:,:].repeat(1,self.env.n_car,1,1)
		V = self.Wv(node_embeddings)
		K2 = self.Wk2(node_embeddings)
		self.Q_fixed, self.K1, self.V, self.K2 = list(
			map(lambda x: x[:,None,:,:].repeat(1,self.env.n_car,1,1)
				, [Q_fixed, K1, V, K2]))
		
	def compute_dynamic(self, mask, step_context):
		Q_step = self.Wq_step(step_context)
		Q1 = self.Q_fixed + Q_step
		# Q1 = Q1.view(self.env.batch, -1, self.env.embed_dim)
		
		# Q1, self.K1, self.V, self.K2 = list(
		# 	map(lambda x: x.view(self.env.batch, -1, self.env.embed_dim)
		# 		, [Q1, self.K1, self.V, self.K2]))
		Q2 = self.MHA([Q1, self.K1, self.V], mask = mask)
		Q2 = self.Wout(Q2)
		logits = self.SHA([Q2, self.K2, None], mask = mask)
		# logits: (batch, n_car, 1, n_node)
		logits = logits.view(self.env.batch, -1)
		return logits

	def forward(self, x, encoder_output, return_pi = False, decode_type = 'sampling'):
		node_embeddings, graph_embedding = encoder_output
		self.env = Env(x, node_embeddings)
		self.compute_static(node_embeddings, graph_embedding)
		mask, step_context = self.env._get_step_t1()
		
		selecter = {'greedy': TopKSampler(), 'sampling': CategoricalSampler()}.get(decode_type, None)
		log_ps, tours, cars, idxs = [[] for _ in range(4)]
		for i in range(self.env.n_node * 4):
			logits = self.compute_dynamic(mask, step_context)
			log_p = torch.log_softmax(logits, dim = -1)	
			idx = selecter(log_p)
			next_car = idx // self.env.n_node
			next_node = idx % self.env.n_node
			mask, step_context = self.env._get_step(next_node, next_car)
			# print('next_node[0]', next_node[0])
			# print('next_car[0]', next_car[0])
			tours.append(next_node)
			cars.append(next_car)
			idxs.append(idx)
			log_ps.append(log_p)	
			if self.env.traversed_customer.all():
				break

		assert self.env.traversed_customer.all(), "not traversed all customer"
		# print('self.env.car_start_node:', self.env.car_start_node)
		self.env.return_depot_all_car()
		# print(self.env.car_run[0])
		cost = self.env.car_run.sum(1)
		# print('self.env.pi.size():', self.env.pi.size())
		
		"""
		print straightforward pi and selected vehicle 
		select_nodes = torch.stack(tours, 1)
		select_cars = torch.stack(cars, 1)
		print(f'select_nodes.size():{select_nodes.size()}\nselect_cars.size():{select_cars.size()}')
		print('select_nodes[0]:', select_nodes[0])
		print('select_cars[0]:', select_cars[0])
		"""

		_idx  = torch.stack(idxs, 1)
		_log_p = torch.stack(log_ps, 1)
		ll = self.env.get_log_likelihood(_log_p, _idx)
		
		if return_pi:
			return cost, ll, self.env.pi
		return cost, ll

if __name__ == '__main__':
	batch, n_car, n_depot, n_customer, n_node = 5, 15, 2, 20, 22
	# batch, n_car, n_depot, n_customer, n_node = 2, 5, 4, 5, 9
	# batch, n_car, n_depot, n_customer, n_node = 2, 5, 3, 5, 8
	# batch, n_car, n_depot, n_customer, n_node = 2, 10, 1, 20, 21
	# batch, n_car, n_depot, n_customer, n_node = 1, 5, 1, 10, 11
	# batch, n_car, n_depot, n_customer, n_node = 2, 5, 3, 10, 13
	assert n_node == n_depot + n_customer
	embed_dim = 128

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	data = generate_data(device, batch = batch, n_car = n_car, n_depot = n_depot, n_customer = n_customer)
	
	decoder = DecoderCell(embed_dim, n_heads = 8, clip = 10.)
	node_embeddings = torch.rand((batch, n_node, embed_dim), dtype = torch.float, device = device)
	graph_embedding = node_embeddings.mean(dim = 1)
	encoder_output = (node_embeddings, graph_embedding)
	# a = graph_embedding[:,None,:].expand(batch, 7, embed_dim)
	# a = graph_embedding[:,None,:].repeat(1, 7, 1)
	
	# decoder.train()
	return_pi = True
	output = decoder(data, encoder_output, return_pi = return_pi, decode_type = 'sampling')
	if return_pi:
		"""cost: (batch)
			ll: (batch)
			pi: (batch, n_car, decode_step)
		"""
		cost, ll, pi = output
		print('\ncost: ', cost.size(), cost)
		print('\nll: ', ll.size(), ll)
		print('\npi: ', pi.size(), pi)
	else:
		cost, ll = output
		print('\ncost: ', cost.size(), cost)
		print('\nll: ', ll.size(), ll)

	cnt = 0
	for k, v in decoder.state_dict().items():
		print(k, v.size(), torch.numel(v))
		cnt += torch.numel(v)
	print(cnt)

	# ll.mean().backward()
	# print(decoder.Wk1.weight.grad)
	# https://discuss.pytorch.org/t/model-param-grad-is-none-how-to-debug/52634	
