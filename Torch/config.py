import pickle
import os
import argparse
from datetime import datetime

def arg_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--seed', metavar = 'SE', type = int, default = 123, help = 'random seed number for inference(or validation during training), reproducibility')
	# main config
	parser.add_argument('-n', '--n_customer', metavar = 'N', type = int, default = 20, help = 'number of customer nodes, time sequence')
	parser.add_argument('-c', '--n_car', metavar = 'C', type = int, default = 10, help = 'number of available vehicles')
	parser.add_argument('-d', '--n_depot', metavar = 'D', type = int, default = 2, help = 'number of depot nodes')

	# batch config
	parser.add_argument('-b', '--batch', metavar = 'B', type = int, default = 512, help = 'batch size')
	parser.add_argument('-s', '--batch_steps', metavar = 'S', type = int, default = 2500, help = 'number of samples = batch * batch_steps')
	parser.add_argument('-v', '--batch_verbose', metavar = 'V', type = int, default = 10, help = 'print and logging during training process')
	parser.add_argument('-e', '--epochs', metavar = 'E', type = int, default = 20, help = 'total number of samples = epochs * number of samples')

	parser.add_argument('-nr', '--n_rollout_samples', metavar = 'NR', type = int, default = 5000, help = 'baseline rollout number of samples')
	parser.add_argument('-nv', '--n_val_samples', metavar = 'NV', type = int, default = 5000, help = 'validation number of samples during training')
	parser.add_argument('-em', '--embed_dim', metavar = 'EM', type = int, default = 128, help = 'embedding size')
	parser.add_argument('-nh', '--n_heads', metavar = 'NH', type = int, default = 8, help = 'number of heads in MHA')
	parser.add_argument('-th', '--tanh_clipping', metavar = 'TH', type = float, default = 10., help = 'improve exploration; clipping logits')
	parser.add_argument('-ne', '--n_encode_layers', metavar = 'NE', type = int, default = 3, help = 'number of MHA encoder layers')
	# parser.add_argument('-nw', '--num_workers', metavar = 'NUMW', type = int, default = 6, help = 'args num_workers in Dataloader, pytorch')
	parser.add_argument('--lr', metavar = 'LR', type = float, default = 1e-4, help = 'initial learning rate')
	parser.add_argument('-wb', '--warmup_beta', metavar = 'WB', type = float, default = 0.8, help = 'exponential moving average, warmup')
	parser.add_argument('-we', '--wp_epochs', metavar = 'WE', type = int, default = 1, help = 'warmup epochs')
	
	parser.add_argument('--islogger', action = 'store_false', help = 'flag csv logger, default true')
	parser.add_argument('-ld', '--log_dir', metavar = 'LD', type = str, default = './Csv/', help = 'csv logger dir')
	parser.add_argument('-wd', '--weight_dir', metavar = 'MD', type = str, default = './Weights/', help = 'model weight save dir')
	parser.add_argument('-pd', '--pkl_dir', metavar = 'PD', type = str, default = './Pkl/', help = 'pkl save dir')
	parser.add_argument('-cd', '--cuda_dv', metavar = 'CD', type = str, default = '0', help = 'os CUDA_VISIBLE_DEVICE')
	args = parser.parse_args()
	return args

class Config():
	def __init__(self, **kwargs):	
		for k, v in kwargs.items():
			self.__dict__[k] = v
		self.mode = 'train'
		self.optimizer = 'Adam'
		self.task = 'VRP%d'%(self.n_customer)
		self.dump_date = datetime.now().strftime('%m%d_%H_%M')
		for x in [self.log_dir, self.weight_dir, self.pkl_dir]:
			os.makedirs(x, exist_ok = True)
		self.pkl_path = self.pkl_dir + self.task + '.pkl'
		self.n_samples = self.batch * self.batch_steps

def print_cfg(cfg):
	print(''.join('%s: %s\n'%item for item in vars(cfg).items()))
		
def dump_pkl(args, verbose = True):
	cfg = Config(**vars(args))
	with open(cfg.pkl_path, 'wb') as f:
		pickle.dump(cfg, f)
		print('--- save pickle file at: %s ---\n'%cfg.pkl_path)
		if verbose:
			print_cfg(cfg)
			
def load_pkl(pkl_path, verbose = True):
	if not os.path.isfile(pkl_path):
		raise FileNotFoundError('pkl_path')
	with open(pkl_path, 'rb') as f:
		cfg = pickle.load(f)
		os.environ['CUDA_VISIBLE_DEVICE'] = cfg.cuda_dv
		if verbose:
			print_cfg(cfg)
	return cfg

def train_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', '--path', metavar = 'P', type = str, 
						default = 'Pkl/VRP20.pkl',
						help = 'Pkl/VRP***.pkl, pkl file only, default: Pkl/VRP20.pkl')
	args = parser.parse_args()
	return args

def test_parser():
	parser = argparse.ArgumentParser()
	parser.add_argument('-p', '--path', metavar = 'P', type = str, required = True,  
						help = 'Weights/VRP***_epoch***.pt, pt file required')
	parser.add_argument('-dt', '--decode_type', metavar = 'DT', type = str, default = 'sampling', choices = ['greedy', 'sampling'], help = 'greedy or sampling required')
	parser.add_argument('-b', '--batch', metavar = 'B', type = int, default = 2, help = 'batch size')
	parser.add_argument('-n', '--n_customer', metavar = 'N', type = int, default = 20, help = 'number of customer nodes, time sequence')
	parser.add_argument('-c', '--n_car', metavar = 'C', type = int, default = 10, help = 'number of available vehicles')
	parser.add_argument('-d', '--n_depot', metavar = 'D', type = int, default = 2, help = 'number of depot nodes')
	parser.add_argument('-s', '--seed', metavar = 'S', type = int, default = 123, help = 'random seed number for inference, reproducibility')
	parser.add_argument('-t', '--txt', metavar = 'T', type = str, help = 'if you wanna test out on text file, example: ../OpenData/A-n53-k7.txt')
	
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = arg_parser()
	dump_pkl(args)
	# cfg = load_pkl(file_parser().path)
	# for k, v in vars(cfg).items():
	# 	print(k, v)
	# 	print(vars(cfg)[k])#==v
