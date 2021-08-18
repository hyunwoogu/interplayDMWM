import numpy as np
from .functions import convert_to_rg

__all__ = ['par', 'update_parameters']

# All the relevant parameters ========================================================================
par = {
	# Demonstration parameters 
	'jobs'       : ['train', 'analysis'],
	'architect'  : ['ff', 'fb'],
	'max_iter'   : 300,  # number of iterations for each network
	'n_repeat'   : 10,   # number of networks trained
	'n_print'    : 10,
	'ref_train'  : [-4,-3,-2,-1,1,2,3,4],
	'ref_test'   : [-4,-3,-2,-1,0,1,2,3,4],
	'save_model' : False, 
	'save_data'  : True, 
	'save_figure': True,

	# Experiment design: unit: second(s)
	'design': {'iti'     : (0, 0.3),
               'stim'    : (0.3, 0.9),                      
               'decision': (0.9, 1.2),
               'delay'   : (0.9, 0.9),
               'estim'   : (1.2, 1.5)}, 

	'shorter': {'iti'    : (0, 0.3),
               'stim'    : (0.3, 0.9),                      
               'decision': (0.9, 1.2),
               'delay'   : (0.9, 0.9),
               'estim'   : (1.2, 1.5)}, 

	'longer': {'iti'     : (0, 1.3),
               'stim'    : (1.3, 1.8),                      
               'decision': (1.8, 3.0),
               'delay'   : (3.0, 3.0),
               'estim'   : (3.0, 4.0)}, 

	'dm_output_range': 'design',  # decision period
	'em_output_range': 'design',  # estim period

	# Mask specs
	'dead': 'design',  # global dead period (not contributing to loss)
	'mask_dm': {'iti': 0., 'stim': 0., 'decision': 1., 'delay': 0., 'estim': 0.,
				'rule_iti': 0., 'rule_stim': 0., 'rule_decision': 0,  'rule_delay': 0., 'rule_estim': 0.},
	'mask_em': {'iti': 0., 'stim': 1., 'decision': 1., 'delay': 1., 'estim': 1.,
				'rule_iti': 0., 'rule_stim': 0., 'rule_decision': 0., 'rule_delay': 0., 'rule_estim': 0.},

	# Rule specs
	'input_rule': 'design',  # {'fixation': whole period, 'response':estim}
	'output_dm_rule': 'design',  # {'fixation' : (0,before estim)}
	'output_em_rule': 'design',  # {'fixation' : (0,before estim)}
	'input_rule_strength'     : 0.8,
	'output_dm_rule_strength' : 0.8,
	'output_em_rule_strength' : 0.8,

	# Decision specs
	'reference': [-4, -3, -2, -1, 1, 2, 3, 4], # trained range of relative reference locations
	'strength_ref': 1.,
	'strength_decision': 0.8,

	# stimulus specs
	'type'			     : 'orientation',  # size, orientation
	'stim_dist'		     : 'uniform', # or a specific input
	'ref_dist'		     : 'uniform', # or a specific input

	# stimulus encoding/response decoding type
	'stim_encoding'	    : 'single', # 'single', 'double'
	'resp_decoding'	    : 'disc',   # 'conti', 'disc', 'onehot'
	'noise_sd'          : 0.05,     # noise level of the overall input
	'noise_sd_stim'     : 0.1,      # noise level of stimulus input

	# Tuning function data
	'strength_input'    : 0.8,  # magnitutde scaling factor for von Mises
	'strength_output'   : 0.8,  # magnitutde scaling factor for von Mises
	'kappa'             : 2,    # concentration scaling factor for von Mises

	# Network configuration
	'exc_inh_prop'      : 0.8,  # excitatory/inhibitory ratio
	'connect_prob'	    : 0.1,  # modular connectivity

	# Timings and rates
	'dt'                : 20.,  # unit: ms
	'tau'   			: 100,  # neuronal timescale

	# Neuronal settings
	'n_receptive_fields': 1,
	'n_tuned_input'	 : 24,      # number of possible orientation-tuned neurons (input)
	'n_tuned_output' : 24,      # number of possible orientation-tuned neurons (input)
	'n_ori'	 	     : 24 ,     # number of possible orientaitons (output)
	'noise_rnn_sd'   : 0.5,     # internal noise level
	'n_recall_tuned' : 24,      # resolution at the moment of recall
	'n_hidden1' 	 : 48,      # number of population 1
	'n_hidden2' 	 : 48,      # number of population 2

	# Experimental settings
	'batch_size' 	: 128,

	# Optimizer
	'optimizer' : 'Adam',
}


def update_parameters(par):
	# ranges and masks
	par.update({'design_rg': convert_to_rg(par['design'], par['dt'])})

	#
	par.update({
		'n_timesteps' : sum([len(v) for _ ,v in par['design_rg'].items()]),
		'n_ref'       : len(par['reference']),
	})

	# default settings
	if par['dm_output_range'] == 'design':
		par['dm_output_rg'] = convert_to_rg(par['design']['decision'], par['dt'])
	else:
		par['dm_output_rg'] = convert_to_rg(par['em_output_range'], par['dt'])

	if par['em_output_range'] == 'design':
		_stim     = convert_to_rg(par['design']['stim'], par['dt'])
		_decision = convert_to_rg(par['design']['decision'], par['dt'])
		_delay    = convert_to_rg(par['design']['delay'], par['dt'])
		_estim    = convert_to_rg(par['design']['estim'], par['dt'])
		em_output = np.concatenate((_stim,_decision,_delay,_estim))
		
		par['em_output_rg'] = em_output
	else:
		par['em_output_rg'] = convert_to_rg(par['em_output_range'], par['dt'])

	# TODO(HG): this may not work if design['estim'] is 2-dimensional
	if par['dead'] == 'design':
		par['dead_rg'] = convert_to_rg(((0 ,0.1),
										(par['design']['estim'][0] ,par['design']['estim'][0 ] +0.1)) ,par['dt'])
	else:
		par['dead_rg'] = convert_to_rg(par['dead'], par['dt'])

	if par['input_rule'] == 'design':
		par['input_rule_rg'] = convert_to_rg({'decision'  : par['design']['decision']}, par['dt'])
		par['n_rule_input']  = 0
	else:
		par['input_rule_rg']  = convert_to_rg(par['input_rule'], par['dt'])
		par['n_rule_input']   = len(par['input_rule'])

	## set n_input
	if par['stim_encoding'] == 'single':
		par['n_input'] = par['n_rule_input'] + par['n_tuned_input']

	elif par['stim_encoding'] == 'double':
		par['n_input'] = par['n_rule_input'] + par['n_tuned_input'] * 2

	## Decision-phase range
	if par['output_dm_rule'] == 'design':
		par['output_dm_rule_rg'] = convert_to_rg({'fixation'  : ((0, par['design']['decision'][0]),
																 (par['design']['decision'][1], par['design']['estim'][1]))}, par['dt'])
		par['n_rule_output_dm']  = 0
	else:
		par['output_dm_rule_rg'] = convert_to_rg(par['output_dm_rule'], par['dt'])
		par['n_rule_output_dm']  = len(par['output_dm_rule'])

	## Estimation-phase range
	if par['output_em_rule'] == 'design':
		par['output_em_rule_rg'] = convert_to_rg({'fixation'  : (0, par['design']['estim'][0])}, par['dt'])
		par['n_rule_output_em']  = 0
	else:
		par['output_em_rule_rg'] = convert_to_rg(par['output_em_rule'], par['dt'])
		par['n_rule_output_em']  = len(par['output_em_rule'])

	## set n_estim_output
	par['n_output_dm'] = par['n_rule_output_dm'] + 2
	if par['resp_decoding'] == 'conti':
		par['n_output_em'] = par['n_rule_output_em'] + 1
	elif par['resp_decoding'] in ['disc', 'onehot']:
		par['n_output_em'] = par['n_rule_output_em'] + par['n_tuned_output']

	## stimulus distribution
	if par['stim_dist'] == 'uniform':
		par['stim_p'] = np.ones(par['n_ori'])
	else:
		par['stim_p'] = par['stim_dist']
	par['stim_p'] = par['stim_p' ] /np.sum(par['stim_p'])

	if par['ref_dist'] == 'uniform':
		par['ref_p'] = np.ones(par['n_ref'])
	else:
		par['ref_p'] = par['ref_dist']
	par['ref_p'] = par['ref_p' ] /np.sum(par['ref_p'])

	return par

par = update_parameters(par)


