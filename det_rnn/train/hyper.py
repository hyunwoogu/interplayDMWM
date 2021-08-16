import numpy as np
import tensorflow as tf
from det_rnn.base import par
from det_rnn.base.functions import initialize, w_design

__all__ = ['hp', 'hp_spec', 'update_hp']

# Model hyperparameters(modifiable)
hp  = {
	'w_in1_fix'   : True,
	'w_in2_fix'   : True,
	'w_out_dm_fix': True, # fix linear voting from two separate populations
	'w_out_em_fix': True, # fix circular voting from two separate populations

	'DtoE_off'    : False,
	'EtoD_off'    : False,

	'gain'          : 0., # amplitude of random initialization (higher values make dynamics chaotic) 
	'loss_fun'	    : 1., # 0:'mse', 1:'centropy'
	'task_type'     : 0., # 0:'decision', 1:'estimation'
	'learning_rate' : 2e-2,	  # adam optimizer learning rate
	'dt'            : 20.,
	'clip_max_grad_val' : 0.1,
	'spike_cost'    : 0,
	'weight_cost'   : 0.,
	'noise_rnn_sd'  : 0.1,

	'lam_decision': 1.,
	'lam_estim'   : 1.,
    'tau_neuron'  : 100.,
	'batch_size_v': np.zeros(par['batch_size']),
	
	'w_in1'   : w_design('w_in1', par),
	'w_in2'   : w_design('w_in2', par),
	'w_out_dm': w_design('w_out_dm', par),
	'w_out_em': w_design('w_out_em', par),
}

def update_hp(hp):
	hp.update({
		'w_in10'    : initialize((par['n_input'],   par['n_hidden1']),   gain=hp['gain']),
		'w_in20'    : initialize((par['n_input'],   par['n_hidden2']),   gain=hp['gain']),
		'w_rnn110'  : initialize((par['n_hidden1'], par['n_hidden1']),   gain=hp['gain']),
		'w_rnn120'  : initialize((par['n_hidden1'], par['n_hidden2']),   gain=hp['gain']),
		'w_rnn210'  : initialize((par['n_hidden2'], par['n_hidden1']),   gain=hp['gain']),
		'w_rnn220'  : initialize((par['n_hidden2'], par['n_hidden2']),   gain=hp['gain']),
		'w_out_dm0' : initialize((par['n_hidden1'], par['n_output_dm']), gain=hp['gain']),
		'w_out_em0' : initialize((par['n_hidden2'], par['n_output_em']), gain=hp['gain'])
	})

	hp.update({
		'alpha_neuron1': np.float32(hp['dt']/hp['tau_neuron']),
		'alpha_neuron2': np.float32(hp['dt']/hp['tau_neuron']),
	})

	return hp

hp = update_hp(hp)

# Tensorize hp
for k, v in hp.items():
    hp[k] = tf.constant(v, name=k)	

# hp_spec: NEED TO BE CHANGED
hp_spec = {}
for k, v in hp.items():
	hp_spec[k] = tf.TensorSpec(v.numpy().shape, tf.dtypes.as_dtype(v.numpy().dtype), name=k)