import numpy as np

from tqdm import tqdm

import torch
from torch import nn
from torch.backends import cudnn
import shutil

from agents.base import BaseAgent

# import your classes here

from tensorboardX import SummaryWriter
from utils.metrics import AverageMeter, AverageMeterList
from utils.misc import print_cuda_statistics
from datasets.FunctionIO_loader import FunctionIO_Loader
from models.MLP import MLP
cudnn.benchmark = True


class MLP_Agent(BaseAgent):

	def __init__(self, config):
		super().__init__(config)
	
		# define data_loader
		if config.mode != 'interactive':self.data_loader = FunctionIO_Loader(config)
	
		# Define model	
		self.model = MLP(config.input_size,config.output_size,config.hidden_size,config.hidden_layers)

		# define loss
		self.loss = nn.MSELoss()

		# define optimizers for both generator and discriminator
		self.optimizer = torch.optim.Adam(
		self.model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

		# initialize counter
		self.current_epoch = 0
		self.current_iteration = 0
		self.best_valid_acc = 1

		# set cuda flag
		self.is_cuda = torch.cuda.is_available()
		if self.is_cuda and not self.config.cuda:
			self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

		self.cuda = self.is_cuda & self.config.cuda

		# set the manual seed for torch
		self.manual_seed = self.config.seed
		if self.cuda:
			torch.cuda.manual_seed_all(self.manual_seed)
			torch.cuda.set_device(self.config.gpu_device)
			self.device = torch.device('cuda')
			self.model = self.model.cuda()
			self.loss = self.loss.cuda()
			self.logger.info("Program will run on *****GPU-CUDA***** ")
#			print_cuda_statistics()
		else:
			self.device = torch.device('cpu')
			self.logger.info("Program will run on *****CPU*****\n")

		# Model Loading from the latest checkpoint if not found start from scratch.
		self.load_checkpoint(self.config.checkpoint_file)
		# Summary Writer
		self.summary_writer = SummaryWriter(log_dir=self.config.summary_dir, comment='NP2Dencoder')

	def load_checkpoint(self, filename):
		"""
		Latest checkpoint loader
		:param file_name: name of the checkpoint file
		:return:
		"""
		filename = self.config.checkpoint_dir + filename
		try:
			if self.config.load_checkpoint:
				self.logger.info("Loading checkpoint '{}'".format(filename))
				checkpoint = torch.load(filename,map_location=self.device)
	
				self.current_epoch = checkpoint['epoch']
				self.current_iteration = checkpoint['iteration']

				self.model.load_state_dict(checkpoint['state_dict'])
				self.optimizer.load_state_dict(checkpoint['optimizer'])
	
				self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
								 .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
			else:
				raise OSError
		except OSError as e:
			self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
			self.logger.info("**First time to train**")


	def save_checkpoint(self, is_best=0):
		"""
		Checkpoint saver
		:param file_name: name of the checkpoint file
		:param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
		:return:
		"""
		filename=self.config.checkpoint_file
		state = {
			'epoch': self.current_epoch,
			'iteration': self.current_iteration,
			'state_dict': self.model.state_dict(),
			'optimizer': self.optimizer.state_dict(),
		}
		# Save the state
		torch.save(state, self.config.checkpoint_dir + filename)
		# If it is the best copy it to another file 'model_best.pth.tar'
		if is_best:
			shutil.copyfile(self.config.checkpoint_dir + filename,
							self.config.checkpoint_dir + 'model_best.pth.tar')

	def run(self):
		"""
		The main operator
		:return:
		"""
		try:
			if self.config.mode == 'test':
				self.validate(self.data_loader.test_loader,scalar_name='test/LL')
			elif self.config.mode == 'interactive':
				self.model.eval()
			else:
				self.train()
				self.validate(self.data_loader.test_loader,scalar_name='test/LL')

		except KeyboardInterrupt:
			self.logger.info("You have entered CTRL+C.. Wait to finalize")

	def train(self):
		"""
		Main training loop
		:return:
		"""
		for epoch in range(self.current_epoch, self.config.max_epoch):
			print('Epoch '+str(epoch))
			self.current_epoch = epoch
			self.train_one_epoch()

			valid_acc = self.validate(self.data_loader.valid_loader)
			is_best = valid_acc < self.best_valid_acc
			if is_best:
					self.best_valid_acc = valid_acc
			self.save_checkpoint(is_best=is_best)

	def train_one_epoch(self):
		"""
		One epoch of training
		:return:
		"""
		self.model.train()
		train_loss = 0

		for batch_idx, data in enumerate(self.data_loader.train_loader):
			Input = data[0].float().to(self.device)
			Output = data[1].float().to(self.device)

			self.optimizer.zero_grad()
			loss = self.loss(self.model(Input)[:,0],Output)
			train_loss += loss.item()
			loss.backward()
			self.optimizer.step()
			self.current_iteration += 1

		self.summary_writer.add_scalar('training/loss', loss.item(), self.current_epoch)

	def validate(self,loader,scalar_name='validation/LL'):
		"""
		One cycle of model validation
		:return:
		"""
		self.model.eval()
		val_loss = 0

		for batch_idx, data in enumerate(loader):
			Input = data[0].float().to(self.device)
			Output = data[1].float().to(self.device)

			with torch.no_grad():
					val_loss += self.loss(self.model(Input)[:,0],Output).item()	# sum up batch loss

		self.summary_writer.add_scalar(scalar_name, val_loss / len(loader.dataset), self.current_epoch)

		return val_loss


	def finalize(self):
		"""
		Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
		:return:
		"""
		self.logger.info("Please wait while finalizing the operation.. Thank you")
		self.save_checkpoint()
		self.summary_writer.export_scalars_to_json("{}all_scalars.json".format(self.config.summary_dir))
		self.summary_writer.close()
		self.data_loader.finalize()
		if self.config.output_model == True:
			try:
				self.logger.info('Saving model for external usage.')
				self.load_checkpoint('model_best.pth.tar')
				traced = torch.jit.trace(self.model,self.data_loader.train_loader.dataset[:2][0].float().to(self.device))
				traced.save(self.config.output_model_path)
			except IOError:
				self.logger.info('Output model path not found.')
