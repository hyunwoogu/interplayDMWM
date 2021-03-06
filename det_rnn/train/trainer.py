import numpy as np
import tensorflow as tf
from .model import Model
from .hyper import hp_spec

__all__ = ['initialize_rnn', 'append_model_performance', 'print_results',
           'tensorize_trial', 'gen_ti_spec']

def initialize_rnn(ti_spec,hp_spec=hp_spec):
    model = Model()
    model.__call__.get_concrete_function(
        trial_info=ti_spec,
        hp=hp_spec
    )
    model.rnn_model.get_concrete_function(
        input_data1=ti_spec['neural_input1'],
        input_data2=ti_spec['neural_input2'],
        hp=hp_spec
    )
    return model

def append_model_performance(model_performance, trial_info, Y, Loss, par):
    decision_perf, estim_perf = _get_eval(trial_info, Y, par)
    model_performance['loss'].append(Loss['loss'].numpy())
    model_performance['perf_loss_dm'].append(Loss['perf_loss_dm'].numpy())
    model_performance['perf_loss_em'].append(Loss['perf_loss_em'].numpy())
    model_performance['spike_loss'].append(Loss['spike_loss'].numpy())
    model_performance['perf_dm'].append(decision_perf)
    model_performance['perf_em'].append(estim_perf)
    return model_performance

def print_results(model_performance, iteration):
    print_res = 'Iter. {:4d}'.format(iteration)
    print_res += ' | Decision Performance {:0.4f}'.format(model_performance['perf_dm'][iteration]) + \
                 ' | Estimation Performance {:0.4f}'.format(model_performance['perf_em'][iteration]) + \
                 ' | Loss {:0.4f}'.format(model_performance['loss'][iteration])
    print_res += ' | Spike loss {:0.4f}'.format(model_performance['spike_loss'][iteration])
    print(print_res)

def tensorize_trial(trial_info):
    for k, v in trial_info.items():
        trial_info[k] = tf.constant(v, name=k)
    return trial_info

def gen_ti_spec(trial_info) :
    ti_spec = {}
    for k, v in trial_info.items():
        _shape = list(v.shape)
        if len(_shape) > 1: _shape[0] = None; _shape[1] = None
        ti_spec[k] = tf.TensorSpec(_shape, tf.dtypes.as_dtype(v.dtype), name=k)
    return ti_spec

def _get_eval(trial_info, output, par):
    argoutput = tf.math.argmax(output['dm'], axis=2).numpy()
    perf_dm   = np.mean(np.array([argoutput[t,:] == ((trial_info['reference_ori'].numpy() > 0)) for t in par['design_rg']['decision']]))

    if par['resp_decoding'] == 'disc':
        cenoutput = tf.nn.softmax(output['em'], axis=2).numpy()
        post_prob = cenoutput[:, :, par['n_rule_output_em']:]
        post_prob = post_prob / (np.sum(post_prob, axis=2, keepdims=True) + np.finfo(np.float32).eps)  # Dirichlet normaliation
        post_support = np.linspace(0, np.pi, par['n_ori'], endpoint=False) + np.pi / par['n_ori'] / 2
        pseudo_mean = np.arctan2(post_prob @ np.sin(2 * post_support),
                                 post_prob @ np.cos(2 * post_support)) / 2
        estim_sinr = (np.sin(2 * pseudo_mean[par['design_rg']['estim'], :])).mean(axis=0)
        estim_cosr = (np.cos(2 * pseudo_mean[par['design_rg']['estim'], :])).mean(axis=0)
        estim_mean = np.arctan2(estim_sinr, estim_cosr) / 2
        perf_em = np.mean(np.cos(2. * (trial_info['stimulus_ori'].numpy() * np.pi / par['n_ori'] - estim_mean)))
    else:
        perf_em = np.mean(np.cos(2. * (trial_info['stimulus_ori'].numpy() * np.pi / par['n_ori'] - estim_mean)))

    return perf_dm, perf_em

