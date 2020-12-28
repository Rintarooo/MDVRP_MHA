import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from scipy.stats import ttest_rel
from tqdm import tqdm
import copy

from dataset import Generator
from Nets.model import AttentionModel

def load_model(device, path, embed_dim = 128, n_encode_layers = 3):
	"""https://pytorch.org/tutorials/beginner/saving_loading_models.html
		https://pytorch.org/docs/master/generated/torch.load.html"""
	model_loaded = AttentionModel(embed_dim = embed_dim, n_encode_layers = n_encode_layers, n_heads = 8, tanh_clipping = 10., FF_hidden = 512)
	model_loaded.load_state_dict(torch.load(path, map_location = device))
	# if device == torch.device('cpu'):
	# 	model_loaded.load_state_dict(torch.load(path, map_location = device))
	# else:
	# 	model_loaded.load_state_dict(torch.load(path))
	return model_loaded

class RolloutBaseline:
	# https://github.com/wouterkool/attention-learn-to-route/blob/master/reinforce_baselines.py
	def __init__(self, model, task, weight_dir, n_rollout_samples = 10000, 
				embed_dim = 128, n_car_each_depot = 3, n_depot = 2, n_customer = 20, capa = 1., warmup_beta = 0.8, wp_epochs = 1, device = 'cpu',
				from_checkpoint = False, path_to_checkpoint = None, epoch = 0,
				):
		"""
		Args:
			model: current model
			task: suffix for baseline checkpoint task
			from_checkpoint: start from checkpoint flag
			path_to_checkpoint: path to baseline model weights
			wp_epochs: until when epoch reaches wp_n_epocohs do we warm-up
			epoch: current epoch number
			n_rollout_samples: number of samples to be generated for baseline dataset
			warmup_beta: warmup mixing parameter (exp. exponential moving average parameter)
		"""

		self.n_rollout_samples = n_rollout_samples
		self.cur_epoch = epoch
		self.wp_epochs = wp_epochs
		self.beta = warmup_beta

		# controls the amount of warmup
		self.alpha = 0.0

		self.M = None

		# Checkpoint params
		self.task = task
		self.from_checkpoint = from_checkpoint
		self.path_to_checkpoint = path_to_checkpoint

		# Problem params
		self.embed_dim = embed_dim
		self.n_car_each_depot = n_car_each_depot
		self.n_depot = n_depot
		self.n_customer = n_customer
		self.capa = capa
		self.weight_dir = weight_dir

		self.device = device

		# create and evaluate initial baseline
		self._update_baseline(model, epoch)
		

	def _update_baseline(self, model, epoch):

		# Load or copy baseline model based on self.from_checkpoint condition
		if self.from_checkpoint and self.alpha == 0:
			print('Baseline model loaded')
			self.model = self.load_model(self.device, self.path_to_checkpoint, embed_dim = self.embed_dim, n_customer = self.n_customer)
		else:
			print('Baseline model copied')
			self.model = self.copy_model(model)
			# For checkpoint
			# ~ torch.save(self.model.state_dict(), '%s%s_epoch%s.pt'%(self.weight_dir, self.task, epoch))
		
		self.model = self.model.to(self.device)
		# We generate a new dataset for baseline model on each baseline update to prevent possible overfitting
		self.dataset = Generator(self.device, n_samples = self.n_rollout_samples, n_car_each_depot = self.n_car_each_depot, n_depot = self.n_depot, n_customer = self.n_customer, capa = self.capa, seed = None)

		print(f'Evaluating baseline model on baseline dataset (epoch = {epoch})')
		self.bl_vals = self.rollout(self.model, self.dataset).cpu().numpy()
		self.mean = self.bl_vals.mean()
		print(f"baseline model's mean score on baseline dataset: {np.round(self.mean, 4)}")
		self.cur_epoch = epoch

	def ema_eval(self, cost):
		"""exponential moving average (only for warm-up epochs)
		"""
		if self.M is None:# first iteration
			self.M = cost.mean()
		else:
			self.M = self.beta * self.M + (1. - self.beta) * cost.mean()
		return self.M.detach()

	def eval(self, batch, cost):
		"""Evaluates current baseline model on single training batch
		"""
		if self.alpha == 0:
			return self.ema_eval(cost)

		if self.alpha < 1:
			v_ema = self.ema_eval(cost)
		else:
			v_ema = 0.0

		self.model.eval()
		with torch.no_grad():
			v_b, _ = self.model(batch, decode_type = 'greedy')

		# Combination of baseline cost and exp. moving average cost
		return self.alpha * v_b + (1 - self.alpha) * v_ema

	def eval_all(self, dataset):
		"""Evaluates current baseline model on the whole dataset only for non warm-up epochs
		"""
		if self.alpha < 1:
			return None

		# ~ val_costs = self.rollout(self.model, dataset, batch = 2048)
		val_costs = self.rollout(self.model, dataset, batch = 512)
		return val_costs

	# def epoch_callback(self, model, epoch):
	def epoch_callback(self, model, epoch, batch = 64):
		"""Compares current baseline model with the training model and updates baseline if it is improved
		"""
		self.cur_epoch = epoch

		print(f'Evaluating candidate model on baseline dataset (callback epoch = {self.cur_epoch})')
		candidate_vals = self.rollout(model, self.dataset, batch).cpu().numpy()# costs for training model on baseline dataset
		candidate_mean = candidate_vals.mean()

		print(f'Epoch {self.cur_epoch} candidate mean {candidate_mean}, baseline mean {self.mean}')

		if candidate_mean < self.mean:
			t, p = ttest_rel(candidate_vals, self.bl_vals)# scipy.stats.ttest_rel

			p_val = p / 2 # one-sided
			print(f'p-value: {p_val}')

			if p_val < 0.05:
				print('Update baseline')
				self._update_baseline(model, self.cur_epoch)

		# alpha controls the amount of warmup
		if self.alpha < 1.0:
			self.alpha = (self.cur_epoch + 1) / float(self.wp_epochs)
			print(f'alpha was updated to {self.alpha}')

	def copy_model(self, model):
		new_model = copy.deepcopy(model)
		return new_model

	def rollout(self, model, dataset, batch = 1000, disable_tqdm = False):
		arr = []
		dataloader = DataLoader(dataset, batch_size = batch)
		model.eval()
		with torch.no_grad():
			for inputs in tqdm(dataloader, disable = disable_tqdm, desc = 'Rollout greedy execution'):
				# ~ inputs = list(map(lambda x: x.to(self.device), inputs))
				cost, _ = model(inputs, decode_type = 'greedy')
				# arr.append(cost.data.cpu())
				arr.append(cost)
		return torch.cat(arr, 0)

	def validate(self, model, dataset, batch = 1000):
		"""Validates model on given dataset in greedy mode
		"""
		print('Validate model ...')
		val_costs = self.rollout(model, dataset, batch = batch)
		mean_cost = val_costs.mean()
		print(f"Validation score: {np.round(mean_cost.detach().cpu().numpy(), 4)}")
		return mean_cost
