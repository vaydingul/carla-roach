import torch.nn as nn
import torch

class MultiStepControl(nn.Module):
	"""Multi-step control branch implementation"""

	def __init__(self, params = None):

		super(MultiStepControl, self).__init__()

		"""" ---------------------- MULTI-STEP ----------------------- """
		if params is None:
			raise ValueError("Creating a NULL MultiStep block")
		
		if 'recurrent_cell' not in params:
			raise ValueError(" Missing the recurrent cell parameter ")

		if 'input_size' not in params:
			raise ValueError(" Missing the input size parameter ")

		if 'hidden_size' not in params:
			raise ValueError(" Missing the hidden size parameter ")

		if 'encoder' not in params:
			raise ValueError(" Missing the encoder parameter ")

		if 'policy_head_mu' not in params:
			raise ValueError(" Missing the policy head mu parameter ")

		if 'policy_head_sigma' not in params:
			raise ValueError(" Missing the policy head sigma parameter ")

		if 'number_of_steps' not in params:
			raise ValueError(" Missing the number of steps parameter ")

		self.input_size = params['input_size']
		self.hidden_size = params['hidden_size']
		self.recurrent_cell = params['recurrent_cell'](params['input_size'], params['hidden_size'])

		self.encoder = params['encoder']#nn.Linear(self.hidden_size, self.hidden_size)
		self.policy_head_mu = params['policy_head_mu']#nn.Linear(self.hidden_size, self.input_size)
		self.policy_head_sigma = params['policy_head_sigma']
		self.number_of_steps = params['number_of_steps']

		# TODO: First hidden state is the feature extracted from the image and 
		# the high level command module

		# TODO: Follow the multi-modal fusion transformer paper for autoregressive GRU implementation

	def forward(self, j):
		
		mu_vector = []
		sigma_vector = []
		j_vector = []
		h = j

		mu = self.policy_head_mu(j)[0]#torch.zeros((j.shape[0], 2), dtype = j.dtype)
		sigma = self.policy_head_sigma(j)[0]#torch.zeros((j.shape[0], 2), dtype = j.dtype)
		
		mu_vector.append(mu)
		sigma_vector.append(sigma)
		j_vector.append(j)

		for _ in range(self.number_of_steps):
			
			x_in = torch.cat([mu, sigma, j], dim = 1)

			h = self.recurrent_cell(x_in, h)

			j = self.encoder(h)

			mu = self.policy_head_mu(j)[0]
			sigma = self.policy_head_sigma(j)[0]

			mu_vector.append(mu)
			sigma_vector.append(sigma)
			j_vector.append(j)

		# Length of the list is the number of steps + 1	
		pred_mu, pred_sigma, pred_j = torch.stack(mu_vector, dim = 1), torch.stack(sigma_vector, dim = 1), torch.stack(j_vector, dim = 1)

		return pred_mu, pred_sigma, pred_j


class MultiStepWaypoint(nn.Module):
	"""Multi-step control branch implementation"""

	def __init__(self, params = None):

		super(MultiStepWaypoint, self).__init__()

		"""" ---------------------- MULTI-STEP ----------------------- """
		if params is None:
			raise ValueError("Creating a NULL MultiStep block")
		
		if 'recurrent_cell' not in params:
			raise ValueError(" Missing the recurrent cell parameter ")

		if 'input_size' not in params:
			raise ValueError(" Missing the input size parameter ")

		if 'hidden_size' not in params:
			raise ValueError(" Missing the hidden size parameter ")

		if 'encoder' not in params:
			raise ValueError(" Missing the encoder parameter ")

		if 'policy_head_waypoint' not in params:
			raise ValueError(" Missing the policy head waypoint parameter ")

		if 'number_of_steps' not in params:
			raise ValueError(" Missing the number of steps parameter ")

		self.input_size = params['input_size']
		self.hidden_size = params['hidden_size']
		self.recurrent_cell = params['recurrent_cell'](params['input_size'], params['hidden_size'])

		self.encoder = params['encoder']#nn.Linear(self.hidden_size, self.hidden_size)
		self.policy_head_waypoint = params['policy_head_waypoint']#nn.Linear(self.hidden_size, self.input_size)
		self.number_of_steps = params['number_of_steps']

		# TODO: First hidden state is the feature extracted from the image and 
		# the high level command module

		# TODO: Follow the multi-modal fusion transformer paper for autoregressive GRU implementation

	def forward(self, j):
		
		waypoint_vector = []
		j_vector = []
		h = j

		# It should be (0, 0) for the first step
		waypoint = torch.zeros((j.shape[0], 2), dtype = j.dtype) # self.policy_head_waypoint(j)[0]
		
		#waypoint_vector.append(waypoint)

		for _ in range(self.number_of_steps):
			
			x_in = torch.cat([waypoint, j], dim = 1)

			h = self.recurrent_cell(x_in, h)

			j = self.encoder(h)

			waypoint = self.policy_head_waypoint(j)[0]

			waypoint_vector.append(waypoint)

		# Length of the list is the number of steps + 1	
		pred_waypoint = torch.stack(waypoint_vector, dim = 1)

		return pred_waypoint

