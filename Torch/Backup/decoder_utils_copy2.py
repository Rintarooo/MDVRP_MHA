import torch
import torch.nn as nn

class Env():
	def __init__(self, x, node_embeddings):
		super().__init__()
		"""depot_xy: (batch, n_depot, 2)
			customer_xy: (batch, n_customer, 2)
			--> xy: (batch, n_node, 2); Coordinates of depot + customer nodes
			n_node= n_depot + n_customer
			demand: (batch, n_customer)
			??? --> demand: (batch, n_car, n_customer)
			D(remaining car capacity): (batch, n_car)
			node_embeddings: (batch, n_node, embed_dim)
			--> node_embeddings: (batch, n_car, n_node, embed_dim)

			car_start_node: (batch, n_car); start node index of each car
			car_cur_node: (batch, n_car); current node index of each car
			car_run: (batch, car); distance each car has run 
			pi: (batch, n_car, decoder_step); which index node each car has moved 
			dist_mat: (batch, n_node, n_node); distance matrix
			traversed_nodes: (batch, n_node)
			traversed_customer: (batch, n_customer)
		"""
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
		self.demand = x['demand']
		print(self.demand)
		
		self.xy = torch.cat([x['depot_xy'], x['customer_xy']], 1)
		self.car_start_node, self.D = x['car_start_node'], x['car_capacity']
		self.car_cur_node = self.car_start_node
		self.pi = self.car_start_node.unsqueeze(-1)

		self.n_depot = x['depot_xy'].size(1)
		self.n_customer = x['customer_xy'].size(1)
		self.n_car = self.car_start_node.size(1)
		self.batch, self.n_node, self.embed_dim = node_embeddings.size()
		self.node_embeddings = node_embeddings[:,None,:,:].repeat(1,self.n_car,1,1)
		self.demand_include_depot = torch.cat([torch.zeros((self.batch, self.n_depot), dtype = torch.float, device = self.device), self.demand], dim = 1)
		assert self.demand_include_depot.size(1) == self.n_node, 'demand_include_depot'
		
		# self.demand = demand[:,None,:].repeat(1,self.n_car,1)
				
		self.car_run = torch.zeros((self.batch, self.n_car), dtype = torch.float, device = self.device)

		self.dist_mat = self.build_dist_mat()
		self.mask_depot, self.mask_depot_unused = self.build_depot_mask()
		self.traversed_customer = torch.zeros((self.batch, self.n_customer), dtype = torch.bool, device = self.device)
		
	def build_dist_mat(self):
		xy = self.xy.unsqueeze(1).repeat(1, self.n_node, 1, 1)
		const_xy = self.xy.unsqueeze(2).repeat(1, 1, self.n_node, 1)
		dist_mat = torch.sqrt(((xy - const_xy) ** 2).sum(dim = 3))
		return dist_mat

	def build_depot_mask(self):
		a = torch.arange(self.n_depot, device = self.device).reshape(1, 1, -1).repeat(self.batch, self.n_car, 1)
		b = self.car_start_node[:,:,None].repeat(1, 1, self.n_depot)
		depot_one_hot = (a==b).bool()#.long()
		return depot_one_hot, torch.logical_not(depot_one_hot)

	def get_mask(self, next_node, next_car):
		"""next_node: ([[0],[0],[not 0], ...], (batch, 1), dtype = torch.int32), [0] denotes going to depot
			customer_idx **excludes depot**: (batch, 1), range[0, n_nodes-1] e.g. [[3],[0],[5],[11], ...], [0] denotes 0th customer, not depot
			self.demand **excludes depot**: (batch, n_nodes-1)
			selected_demand: (batch, 1)
			if next node is depot, do not select demand
			self.D: (batch, n_car, 1), D denotes "remaining vehicle capacity"
			self.capacity_over_customer **excludes depot**: (batch, n_car, n_customer)
			visited_customer **excludes depot**: (batch, n_customer, 1)
			is_next_depot: (batch, 1), e.g. [[True], [True], ...]

			mask_depot: (batch, n_car, n_depot) 
			mask_customer: (batch, n_car, n_customer) 
			--> return mask: (batch, n_car, n_node ,1)
		"""
		is_next_depot = (self.car_cur_node == self.car_start_node).bool()#.long().sum(-1)
		is_next_customer = torch.logical_not(is_next_depot)
		# e.g., is_next_depot = next_node == 0 or next_node == 1
		# is_next_depot: (batch, n_car), e.g. [[True], [True], ...]

		customer_idx = torch.clamp(next_node - self.n_depot, min = 0., max = self.n_customer)
		# a = torch.arange(self.n_customer, device = self.device).reshape(1,-1).repeat(self.batch,1)
		# b = customer_idx.reshape(self.batch, 1).repeat(1,self.n_customer)
		
		new_traversed_node = torch.eye(self.n_node, device = self.device)[next_node.squeeze(1)].bool()
		new_traversed_customer = new_traversed_node[:,self.n_depot:]
		# new_traversed_node: (batch, node)
		# new_traversed_customer: (batch, n_customer)

		self.traversed_customer = self.traversed_customer | new_traversed_customer
		# traversed_customer: (batch, n_customer)

		selected_demand = torch.gather(input = self.demand, dim = 1, index = customer_idx)
		# selected_demand: (batch, 1)
		selected_car = torch.eye(self.n_car, device = self.device)[next_car.squeeze(1)]
		# selected_car: (batch, n_car)

		car_used_demand = is_next_customer * selected_demand
		# is_next_customer: (batch, n_car)
		# car_used_demand: (batch, n_car) 
		print('is_next_customer', is_next_customer)
		print('selected_demand', selected_demand)
		print(car_used_demand)

		self.D -= car_used_demand
		# D: (batch, n_car)
		# self.D = torch.clamp(self.D, min = 0.)
		# self.D[:,next_car] = max(0., self.D[:,next_car] - selected_demand * (1.0 - self.is_next_depot.float()))
		
		D_over_customer = self.demand[:,None,:].repeat(1,self.n_car,1) > self.D[:,:,None].repeat(1,1,self.n_customer)
		mask_customer = D_over_customer | self.traversed_customer[:,None,:].repeat(1,self.n_car,1)
		
		mask_depot = is_next_depot & ((mask_customer == False).long().sum(dim = 2).sum(dim = 1)[:,None].repeat(1,self.n_car) > 0)
		"""mask_depot: (batch, n_car)
			mask_depot = True --> We cannot choose depot in the next step 
			if 1) the vehicle is at the depot in the next step
			or 2) there is a customer node which has not been visited yet
		"""

		# one_hot = torch.eye(self.n_node, device = self.device)[self.car_start_node]
		# one_hot: (batch, n_car, n_node)

		# mask_depot = self.mask_depot & mask_depot.bool().reshape(self.batch, self.n_car, 1).repeat(1,1,self.n_depot)
		mask_depot = self.mask_depot & mask_depot.bool()[:,:,None].repeat(1,1,self.n_depot)
		# mask_depot: (batch, n_car, n_depot)

		mask_depot = self.mask_depot_unused | mask_depot
		mask = torch.cat([mask_depot, mask_customer], dim = -1).unsqueeze(-1)
		return mask
	
	def generate_step_context(self):
		"""D: (batch, n_car)
			-->　D: (batch, n_car, 1, 1)
			
			each_car_idx: (batch, n_car, 1, embed_dim)
			node_embeddings: (batch, n_car, n_node, embed_dim)
			--> prev_embeddings(initially, depot_embeddings): (batch, n_car, 1, embed)
			node embeddings where car is located
			
			return step_context: (batch, n_car, 1, embed+1)
		"""
		each_car_idx = self.car_cur_node[:,:,None,None].repeat(1,1,1,self.embed_dim)		
		prev_embeddings = torch.gather(input = self.node_embeddings, dim = 2, index = each_car_idx)
		step_context = torch.cat([prev_embeddings, self.D[:,:,None,None]], dim = -1)
		return step_context

	def _get_step(self, next_node, next_car):
		"""next_node **includes depot** : (batch, 1) int, range[0, n_node-1]
			
			return
			mask: (batch, n_car, n_node ,1)
			step_context: (batch, n_car, 1, embed+1)
		"""
		self.update_node_path(next_node, next_car)
		self.update_car_distance()
		mask = self.get_mask(next_node, next_car)
		step_context = self.generate_step_context()
		return mask, step_context

	def _create_t1(self):
		"""return
			mask: (batch, n_car, n_node ,1)
			step_context: (batch, n_car, 1, embed+1)
		"""
		mask_t1 = self.create_mask_t1()
		step_context_t1 = self.generate_step_context()		
		return mask_t1, step_context_t1

	def create_mask_t1(self):
		"""mask_depot: (batch, n_car, n_depot) 
			mask_customer: (batch, n_car, n_customer) 
			--> return mask: (batch, n_car, n_node ,1)
		"""
		mask_depot_t1 = self.mask_depot | self.mask_depot_unused
		mask_customer_t1 = self.traversed_customer[:,None,:].repeat(1,self.n_car,1)
		mask_t1 = torch.cat([mask_depot_t1, mask_customer_t1], dim = -1).unsqueeze(-1)
		return mask_t1
		
	def update_node_path(self, next_node, next_car):
		# car_node: (batch, n_car)
		# pi: (batch, n_car, decoder_step)
		self.car_prev_node = self.car_cur_node
		a = torch.arange(self.n_car, device = self.device).reshape(1, -1).repeat(self.batch, 1)
		b = next_car.reshape(self.batch, 1).repeat(1, self.n_car)
		mask_car = (a == b).long()
		new_node = next_node.reshape(self.batch, 1).repeat(1, self.n_car)
		self.car_cur_node = mask_car * new_node + (1 - mask_car) * self.car_cur_node
		# (1-mask_car) keeps the same node for the unused car, mask_car updates new node for the used car
		self.pi = torch.cat([self.pi, self.car_cur_node.unsqueeze(-1)], dim = -1)

	def update_car_distance(self):
		prev_node_dist_vec = torch.gather(input = self.dist_mat, dim = 1, index = self.car_prev_node[:,:,None].repeat(1,1,self.n_node))
		# dist = torch.gather(input = prev_node_dist_vec, dim = 2, index = self.car_cur_node[:,None,:].repeat(1,self.n_car,1))
		dist = torch.gather(input = prev_node_dist_vec, dim = 2, index = self.car_cur_node[:,:,None])
		self.car_run += dist.squeeze(-1)
		# print(self.car_run[0])

	def return_depot_all_car(self):
		self.pi = torch.cat([self.pi, self.car_start_node.unsqueeze(-1)], dim = -1)
		self.car_prev_node = self.car_cur_node
		self.car_cur_node = self.car_start_node
		self.update_car_distance()

	def get_log_likelihood(self, _log_p, _idx):
		"""_log_p: (batch, decode_step, n_car * n_node)
			_idx: (batch, decode_step, 1), selected index
		"""
		log_p = torch.gather(input = _log_p, dim = 2, index = _idx)
		return log_p.squeeze(-1).sum(dim = 1)

class Sampler(nn.Module):
	"""args; logits: (batch, n_car * n_nodes)
		return; next_node: (batch, 1)
		TopKSampler --> greedy; sample one with biggest probability
		CategoricalSampler --> sampling; randomly sample one from possible distribution based on probability
	"""
	def __init__(self, n_samples = 1, **kwargs):
		super().__init__(**kwargs)
		self.n_samples = n_samples
		
class TopKSampler(Sampler):
	def forward(self, logits):
		return torch.topk(logits, self.n_samples, dim = 1)[1]
		# torch.argmax(logits, dim = 1).unsqueeze(-1)

class CategoricalSampler(Sampler):
	def forward(self, logits):
		return torch.multinomial(logits.exp(), self.n_samples)