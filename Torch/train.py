import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time

from Nets.model import AttentionModel
from baseline import RolloutBaseline
from dataset import generate_data, Generator
from config import Config, load_pkl, train_parser

INF = 100000.

def train(cfg):
	torch.backends.cudnn.benchmark = True
	
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model = AttentionModel(cfg.embed_dim, cfg.n_encode_layers, cfg.n_heads, cfg.tanh_clipping)
	model.to(device)
	baseline = RolloutBaseline(model, cfg.task, cfg.weight_dir, cfg.n_rollout_samples, 
										cfg.embed_dim, cfg.n_car_each_depot, cfg.n_depot, cfg.n_customer, cfg.capa, cfg.warmup_beta, cfg.wp_epochs, device)
	optimizer = optim.Adam(model.parameters(), lr = cfg.lr)
	validation_dataset = Generator(device, n_samples = cfg.n_val_samples, n_car_each_depot = cfg.n_car_each_depot, n_depot = cfg.n_depot, n_customer = cfg.n_customer, capa = cfg.capa, seed = cfg.seed)

	def rein_loss(model, inputs, bs, t, device):
		model.train()
		L, ll = model(inputs, decode_type = 'sampling')
		b = bs[t] if bs is not None else baseline.eval(inputs, L)
		return ((L - b.to(device)) * ll).mean(), L.mean()
	
	cnt = 0
	min_L = INF
	t1 = time()
	for epoch in range(cfg.epochs):
		avg_loss, avg_L, val_L = [0. for _ in range(3)]
		dataset = Generator(device, cfg.n_samples, cfg.n_car_each_depot, cfg.n_depot, cfg.n_customer, cfg.capa, None)
		bs = baseline.eval_all(dataset)
		bs = bs.view(-1, cfg.batch) if bs is not None else None# bs: (cfg.batch_steps, cfg.batch) or None
		dataloader = DataLoader(dataset, batch_size = cfg.batch, shuffle = True)
		
		for t, inputs in enumerate(dataloader):	
			loss, L_mean = rein_loss(model, inputs, bs, t, device)
			optimizer.zero_grad()
			loss.backward()
			
			nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0, norm_type = 2)
			optimizer.step()
			
			avg_loss += loss.item()
			avg_L += L_mean.item()
			
			if t % (cfg.batch_verbose) == 0:
				t2 = time()
				print('Epoch %d (batch = %d): Loss: %1.3f L: %1.3f, %dmin%dsec'%(
					epoch, t, avg_loss/(t+1), avg_L/(t+1), (t2-t1)//60, (t2-t1)%60))
				if cfg.islogger:
					if t == 0 and epoch == 0:
						log_path = '%s%s_%s.csv'%(cfg.log_dir, cfg.task, cfg.dump_date)#cfg.log_dir = ./Csv/
						print(f'generate {log_path}')
						with open(log_path, 'w') as f:
							f.write('time,epoch,batch,loss,cost\n')
					with open(log_path, 'a') as f:
						f.write('%dmin%dsec,%d,%d,%1.3f,%1.3f\n'%(
							(t2-t1)//60, (t2-t1)%60, epoch, t, avg_loss/(t+1), avg_L/(t+1)))
				t1 = time()

		baseline.epoch_callback(model, epoch, 2*cfg.batch)
		# weight_path = '%s%s_epoch%s.pt'%(cfg.weight_dir, cfg.task, epoch)
		# torch.save(model.state_dict(), weight_path)
		# print(f'generate {weight_path}')
		val_L = baseline.validate(model, validation_dataset, 2*cfg.batch)
		if cfg.islogger:
			if epoch == 0:
				val_path = '%s%s_%s_val.csv'%(cfg.log_dir, cfg.task, cfg.dump_date)#cfg.log_dir = ./Csv/
				print(f'generate {val_path}')
				with open(val_path, 'w') as f:
					f.write('epoch,validation_cost\n')
			
			with open(val_path, 'a') as f:
				f.write('%d,%1.4f\n'%(epoch, val_L))

			if(val_L < min_L):
				min_L = val_L

				# model save
				weight_path = '%s%s_epoch%s.pt'%(cfg.weight_dir, cfg.task, epoch)
				torch.save(model.state_dict(), weight_path)
				print(f'generate {weight_path}')
			else:
				cnt += 1
				print(f'cnt: {cnt}/20')
				if(cnt >= 20):
					print('early stop, average cost cant decrease anymore')
					break
				
			if epoch == 0:
				param_path = '%s%s_%s_param.csv'%(cfg.log_dir, cfg.task, cfg.dump_date)# cfg.log_dir = ./Csv/
				print(f'generate {param_path}')
				with open(param_path, 'w') as f:
					f.write(''.join('%s,%s\n'%item for item in vars(cfg).items())) 
				
if __name__ == '__main__':
	cfg = load_pkl(train_parser().path)
	train(cfg)	
