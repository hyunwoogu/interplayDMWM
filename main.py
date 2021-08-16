import os, datetime, warnings
import numpy as np

from det_rnn import *
import det_rnn.train as dt
import det_rnn.analysis as da

now = datetime.datetime.now()
warnings.simplefilter(action='ignore', category=FutureWarning)

def main(par=par):

    # ==========================================================================================
    # 1. Training part
    # ==========================================================================================
    if par['save_model']:
        model_dir = './save/' + 'model_' + now.strftime('%Y%m%d_%H%M%S') + '/'
        os.makedirs(model_dir)

    if par['save_data']:
        data_dir  = './save/' + 'data_'  + now.strftime('%Y%m%d_%H%M%S') + '/'
        os.makedirs(data_dir)

    for v_arch in par['architect']:

        if v_arch == 'ff':
            dt.hp['DtoE_off'] = True  # no decision-to-estimation connection
        if v_arch == 'fb':
            dt.hp['DtoE_off'] = False # decision-to-estimation connection assumed
        dt.hp                 = dt.update_hp(dt.hp) 

        for i_repeat in range(par['n_repeat']):
            print('\n' + '#'*50 + '\n' + v_arch.upper() + ' Training, {:01d}/{:01d}'.format(i_repeat+1,par['n_repeat']) + '\n' + '#'*50 + '\n')

            ## 1. Train network using "shorter" structure
            par['design'].update(par['shorter'])
            par['reference']  = par['ref_train']
            par['ref_dist']   = np.ones(len(par['ref_train']))
            par               = update_parameters(par)
            stimulus          = Stimulus()
            ti_spec           = dt.gen_ti_spec(stimulus.generate_trial())
            model             = dt.initialize_rnn(ti_spec)
            model_performance = {'perf_dm': [], 'perf_em': [], 'loss': [], 'perf_loss_dm': [], 'perf_loss_em': [], 'spike_loss': []}

            for iter in range(par['max_iter']):
                trial_info        = dt.tensorize_trial(stimulus.generate_trial())
                Y, Loss           = model(trial_info, dt.hp)
                model_performance = dt.append_model_performance(model_performance, trial_info, Y, Loss, par)
                if iter % par['n_print'] == 0: dt.print_results(model_performance, iter) # Print
            
            if par['save_model']:
                import tensorflow as tf
                tf.saved_model.save(model, model_dir + v_arch + '{:04d}'.format(i_repeat))

            ## 2. Generalize network using "longer" structure
            par['design'].update(par['longer'])
            par        = update_parameters(par)
            stimulus   = Stimulus()
            ti_spec    = dt.gen_ti_spec(stimulus.generate_trial())
            trial_info = dt.tensorize_trial(stimulus.generate_trial())

            if par['save_data']:
                import pandas as pd
                data = []

            for i_ref, v_ref in enumerate(par['ref_test']):
                par['reference']  = par['ref_test']
                ref_dist          = np.zeros(len(par['ref_test']))
                ref_dist[i_ref]   = 1.
                par['ref_dist']   = ref_dist
                par               = update_parameters(par)
                stimulus          = Stimulus(par)
                ti_spec    = dt.gen_ti_spec(stimulus.generate_trial())
                trial_info = dt.tensorize_trial(stimulus.generate_trial())
                pred_output_dm, pred_output_em, H1, H2  = model.rnn_model(trial_info['neural_input1'], trial_info['neural_input2'], dt.hp)
                stim       = trial_info['stimulus_ori'].numpy()
                output_dm  = pred_output_dm.numpy()[par['design_rg']['decision'],:,:]
                output_em  = pred_output_em.numpy()[par['design_rg']['estim'],:,:]
                _, _, choice            = da.behavior_summary_dm(output_dm, par=par)
                _, estim_mean, error, _ = da.behavior_summary_em({'stimulus_ori': stim}, output_em, par=par)

                if par['save_data']:
                    sub_data = pd.DataFrame({
                        'reference' : v_ref * 7.5,
                        'estimation': estim_mean * 180/np.pi,
                        'decision'  : choice * 1,
                        'error'     : (error - np.pi/24/2) * 180/np.pi
                    })
                    data.append(sub_data)
            
            if par['save_data']:
                pd.concat(data).to_csv(data_dir + v_arch + '{:04d}.csv'.format(i_repeat), index=False)



    # ==========================================================================================
    # 2. Analysis part (Visualization)
    # ==========================================================================================
    print('\n' + '#'*50 + '\n' + 'Training ended! Now summary figure will be generated.' + '\n' + '#'*50 + '\n')

    if par['save_figure']:
        import pandas as pd
        import matplotlib.pyplot as plt
        figure_file  = './save/' + 'figure_'  + now.strftime('%Y%m%d_%H%M%S') + '.png'

    data_list = os.listdir(data_dir)
    data_plot = {}
    for v_arch in par['architect']:
        data_plot[v_arch] = pd.concat([pd.read_csv(data_dir + d).assign(network = i).assign(type = v_arch) for i, d in enumerate(data_list) if v_arch in d])

    _, ax = plt.subplots(1,2,figsize=[8,5], sharex=True, sharey=True)
    for i_arch, v_arch in enumerate(par['architect']):
        ax[i_arch].hlines(0, xmin=-30, xmax=30, linestyle=(0,(1,3)), color='black', linewidth=1)
        ax[i_arch].vlines(0, ymin=-50, ymax=50, linestyle=(0,(1,3)), color='black', linewidth=1)
        im = ax[i_arch].hist2d(-data_plot[v_arch].reference, 
                data_plot[v_arch].error - data_plot[v_arch].reference, bins=(9, 50), cmap=plt.cm.gray_r, density=True)
        ax[i_arch].set_ylim([-50,50])
        ax[i_arch].plot([-30,30], [-30,30], linestyle=(0,(5,5)), color='white', linewidth=1)
        ax[i_arch].set_xlabel('Stimulus orientation (deg)')
        plt.colorbar(im[3], ax=ax[i_arch])

    ax[0].set_ylabel('Estimation orientation (deg)')
    ax[0].set_title('Without Feedback Connectivity')
    ax[1].set_title('With Feedback Connectivity')
    plt.tight_layout()
    plt.savefig(figure_file, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    
    main()


