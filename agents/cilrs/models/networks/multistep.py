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

		if 'initial_hidden_zeros' not in params:
			raise ValueError(" Missing the initial hidden zeros parameter ")	
		self.params = params
		self.input_size = params['input_size']
		self.hidden_size = params['hidden_size']
		self.recurrent_cell = params['recurrent_cell'](params['input_size'], params['hidden_size'])

		self.encoder = params['encoder']#nn.Linear(self.hidden_size, self.hidden_size)
		self.policy_head_mu = params['policy_head_mu']#nn.Linear(self.hidden_size, self.input_size)
		self.policy_head_sigma = params['policy_head_sigma']
		self.number_of_steps = params['number_of_steps']

		self.initial_hidden_zeros = params['initial_hidden_zeros']
		# TODO: First hidden state is the feature extracted from the image and 
		# the high level command module

		# TODO: Follow the multi-modal fusion transformer paper for autoregressive GRU implementation

	def forward(self, j):
		
		mu_vector = []
		sigma_vector = []
		j_vector = []

		if self.initial_hidden_zeros:
			
			h = torch.zeros_like(j)
		
		else:
			
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


class MultiStepTrajectory(nn.Module):
	"""Multi-step control branch implementation"""

	def __init__(self, params = None):

		super(MultiStepTrajectory, self).__init__()

		"""" ---------------------- MULTI-STEP ----------------------- """
		if params is None:
			raise ValueError("Creating a NULL MultiStep block")
		
		if 'recurrent_cell' not in params:
			raise ValueError(" Missing the recurrent cell parameter ")

		if 'input_size' not in params:
			raise ValueError(" Missing the input size parameter ")

		if 'hidden_size' not in params:
			raise ValueError(" Missing the hidden size parameter ")

		if 'policy_head_waypoint' not in params:
			raise ValueError(" Missing the policy head waypoint parameter ")

		if 'number_of_steps' not in params:
			raise ValueError(" Missing the number of steps parameter ")

		if 'initial_hidden_zeros' not in params:
			raise ValueError(" Missing the initial hidden zeros parameter ")

		self.params = params
		self.input_size = params['input_size']
		self.hidden_size = params['hidden_size']
		self.recurrent_cell = params['recurrent_cell'](params['input_size'], params['hidden_size'])

		self.policy_head_waypoint = params['policy_head_waypoint']#nn.Linear(self.hidden_size, self.input_size)
		self.number_of_steps = params['number_of_steps']

		self.initial_hidden_zeros = params['initial_hidden_zeros']
		# TODO: First hidden state is the feature extracted from the image and 
		# the high level command module

		# TODO: Follow the multi-modal fusion transformer paper for autoregressive GRU implementation

	def forward(self, j, target_waypoint = None):
		
		waypoint_vector = []

		if self.initial_hidden_zeros:

			h = torch.zeros_like(j)

		else:
	
			h = j

		# It should be (0, 0) for the first step
		waypoint = torch.zeros((j.shape[0], 2), dtype = j.dtype, device=j.device) # self.policy_head_waypoint(j)[0]
		
		#waypoint_vector.append(waypoint)

		for _ in range(self.number_of_steps):


			x_in = torch.cat([waypoint, target_waypoint], dim = 1)

			
			h = self.recurrent_cell(x_in, h)


			delta_waypoint = self.policy_head_waypoint(h)[0]

			waypoint = waypoint + delta_waypoint

			waypoint_vector.append(waypoint)

		# Length of the list is the number of steps + 1	
		pred_waypoint = torch.stack(waypoint_vector, dim = 1)

		return pred_waypoint

class MultiStepControlCell(nn.Module):
	"""Multi-step control branch implementation"""

	def __init__(self, params = None, multi_step_control:MultiStepControl = None):

		super(MultiStepControlCell, self).__init__()

		if multi_step_control is not None:

			params = multi_step_control.params

		

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

		if 'initial_hidden_zeros' not in params:
			raise ValueError(" Missing the initial hidden zeros parameter ")	

		self.input_size = params['input_size']
		self.hidden_size = params['hidden_size']
		self.recurrent_cell = params['recurrent_cell'](params['input_size'], params['hidden_size'])

		self.encoder = params['encoder']#nn.Linear(self.hidden_size, self.hidden_size)
		self.policy_head_mu = params['policy_head_mu']#nn.Linear(self.hidden_size, self.input_size)
		self.policy_head_sigma = params['policy_head_sigma']
		self.number_of_steps = params['number_of_steps']

		self.initial_hidden_zeros = params['initial_hidden_zeros']
		# TODO: First hidden state is the feature extracted from the image and 
		# the high level command module

		# TODO: Follow the multi-modal fusion transformer paper for autoregressive GRU implementation

	def forward(self,h, mu, sigma, j):
		
		# mu_vector = []
		# sigma_vector = []
		# j_vector = []

		# if self.initial_hidden_zeros:
			
		# 	h = torch.zeros_like(j)
		
		# else:
			
		# 	h = j

		# mu = self.policy_head_mu(j)[0]#torch.zeros((j.shape[0], 2), dtype = j.dtype)
		# sigma = self.policy_head_sigma(j)[0]#torch.zeros((j.shape[0], 2), dtype = j.dtype)
		
		# mu_vector.append(mu)
		# sigma_vector.append(sigma)
		# j_vector.append(j)

			
		x_in = torch.cat([mu, sigma, j], dim = 1)

		h = self.recurrent_cell(x_in, h)

		j = self.encoder(h)

		mu = self.policy_head_mu(j)[0]
		sigma = self.policy_head_sigma(j)[0]

		
		return h, mu, sigma, j


class MultiStepTrajectoryCell(nn.Module):
	"""Multi-step control branch implementation"""

	def __init__(self, params = None, multi_step_trajectory:MultiStepTrajectory = None):

		super(MultiStepTrajectoryCell, self).__init__()

		if multi_step_trajectory is not None:

			params = multi_step_trajectory.params

		
		"""" ---------------------- MULTI-STEP ----------------------- """
		if params is None:
			raise ValueError("Creating a NULL MultiStep block")
		
		if 'recurrent_cell' not in params:
			raise ValueError(" Missing the recurrent cell parameter ")

		if 'input_size' not in params:
			raise ValueError(" Missing the input size parameter ")

		if 'hidden_size' not in params:
			raise ValueError(" Missing the hidden size parameter ")

		if 'policy_head_waypoint' not in params:
			raise ValueError(" Missing the policy head waypoint parameter ")

		if 'number_of_steps' not in params:
			raise ValueError(" Missing the number of steps parameter ")

		if 'initial_hidden_zeros' not in params:
			raise ValueError(" Missing the initial hidden zeros parameter ")


		self.input_size = params['input_size']
		self.hidden_size = params['hidden_size']
		self.recurrent_cell = params['recurrent_cell'](params['input_size'], params['hidden_size'])

		self.policy_head_waypoint = params['policy_head_waypoint']#nn.Linear(self.hidden_size, self.input_size)
		self.number_of_steps = params['number_of_steps']

		self.initial_hidden_zeros = params['initial_hidden_zeros']
		# TODO: First hidden state is the feature extracted from the image and 
		# the high level command module

		# TODO: Follow the multi-modal fusion transformer paper for autoregressive GRU implementation

	def forward(self, h, waypoint, target_waypoint):
		
		# waypoint_vector = []
		# j_vector = []

		# if self.initial_hidden_zeros:

		# 	h = torch.zeros_like(j)

		# else:
	
		# 	h = j

		# # It should be (0, 0) for the first step
		# waypoint = torch.zeros((j.shape[0], 2), dtype = j.dtype, device=j.device) # self.policy_head_waypoint(j)[0]
		
		#waypoint_vector.append(waypoint)

	


		x_in = torch.cat([waypoint, target_waypoint], dim = 1)

		
		h = self.recurrent_cell(x_in, h)

		#j = self.encoder(h)

		delta_waypoint = self.policy_head_waypoint(h)[0]

		waypoint = waypoint + delta_waypoint

		return h, waypoint


# Shit! Some serious stuff :D
class MultiStepTrajectoryGuidedControl(nn.Module):

	def __init__(self, params = None):

		super(MultiStepTrajectoryGuidedControl, self).__init__()

		"""" ---------------------- MULTI-STEP ----------------------- """
		if params is None:
			raise ValueError("Creating a NULL MultiStep block")

		if 'multi_step_control_cell' not in params:
			raise ValueError(" Missing the multi step control cell parameter ")

		if 'multi_step_trajectory_cell' not in params:
			raise ValueError(" Missing the multi step trajectory cell parameter ")
		
		if 'attention_encoder_1' not in params:
			raise ValueError(" Missing the attention encoder 1 parameter ")

		if 'attention_encoder_2' not in params:
			raise ValueError(" Missing the attention encoder 2 parameter ")

		self.multi_step_control_cell = params['multi_step_control_cell']
		self.multi_step_trajectory_cell = params['multi_step_trajectory_cell']
		self.attention_encoder_1 = params['attention_encoder_1']
		self.attention_encoder_2 = params['attention_encoder_2']


	def forward(self, j_control, mu, sigma, h_traj, waypoint, F, target_waypoint, attention_dims):
		
		mu_vector = []
		sigma_vector = []
		waypoint_vector = []
		j_control_vector = []
		attention_map_1_vector = [] # before softmax
		attention_map_2_vector = [] # after softmax
		h_control = torch.zeros_like(j_control, dtype = j_control.dtype, device = j_control.device)

		mu_vector.append(mu)
		sigma_vector.append(sigma)
		j_control_vector.append(j_control)



		for _ in range(self.multi_step_trajectory_cell.number_of_steps):


			h_traj, waypoint = self.multi_step_trajectory_cell(h_traj, waypoint, target_waypoint)
			h_control = self.multi_step_control_cell.recurrent_cell(torch.cat([mu, sigma, j_control], dim = 1), h_control)

			attention_map_1 = self.attention_encoder_1(torch.cat([h_traj, h_control], dim = 1)).view(-1, 1, *(attention_dims[1:]))
			attention_map_2 = torch.softmax(self.attention_encoder_1(torch.cat([h_traj, h_control], dim = 1)), dim = 1).view(-1, 1, *(attention_dims[1:]))
			attended_features = torch.sum(attention_map_2 * F, dim = (2, 3))
			j_control = self.attention_encoder_2(torch.cat([attended_features, h_control], dim = 1))

			delta_j_control = self.multi_step_control_cell.encoder(j_control)[0]

			j_control = j_control + delta_j_control

			mu = self.multi_step_control_cell.policy_head_mu(j_control)[0]
			sigma = self.multi_step_control_cell.policy_head_sigma(j_control)[0]

			mu_vector.append(mu)
			sigma_vector.append(sigma)
			waypoint_vector.append(waypoint)
			j_control_vector.append(j_control)
			attention_map_1_vector.append(attention_map_1)
			attention_map_2_vector.append(attention_map_2)

		pred_mu = torch.stack(mu_vector, dim = 1)
		pred_sigma = torch.stack(sigma_vector, dim = 1)
		pred_waypoint = torch.stack(waypoint_vector, dim = 1)
		pred_j_control = torch.stack(j_control_vector, dim = 1)
		pred_attention_map_1 = torch.stack(attention_map_1_vector, dim = 1)
		pred_attention_map_2 = torch.stack(attention_map_2_vector, dim = 1)


		return pred_mu, pred_sigma, pred_waypoint, pred_j_control, pred_attention_map_1, pred_attention_map_2



		

