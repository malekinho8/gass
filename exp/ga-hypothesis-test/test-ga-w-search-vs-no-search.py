import numpy as np
import os
import pandas as pd
import torch
import librosa
import time
import sys; sys.path.append(f'.{os.sep}')
import dawdreamer as daw
from src.utils import find_closest_preset_from_mfcc, save_mfcc_history_comparison_plots, load_plugin_with_dawdreamer, get_fundamental_frequency, get_closest_note_from_ff, optimize_preset_with_ga_mfcc, make_parameter_with_fitness_plots
from src.config import *

# load the preset dataset
preset_dataset_path = os.path.join(".", "dataset", "processed_preset_dataset_musicnn.pt")
df = pd.DataFrame(torch.load(preset_dataset_path))

# specify the names of the target audio files to investigate
target_names = ['bill-evans-piano.wav','miles-davis-trumpet.wav','itzhak-perlman-violin.wav']

# specify the names of the methods you wish to compare
methods = ['alternate']

# specify the folder where the target audio files are located
target_dataset_folder = os.path.join('.', 'timbre-exp', 'target-dataset')# './timbre-exp/target-dataset/'

# create the target audio paths
target_audio_paths = [os.path.join(target_dataset_folder, x) for x in target_names]

# instantiate the dawdreamer plugin and engine
engine = daw.RenderEngine(sample_rate=daw_settings['SAMPLE_RATE'], block_size=daw_settings['BLOCK_SIZE'])
plugin = load_plugin_with_dawdreamer(daw_settings['SYNTH_PLUGIN_PATH'],daw_settings['SYNTH_NAME'],engine)

# loop over all target audio paths
for i, target_audio_path in enumerate(target_audio_paths):
    # specify the target name
    target_name = target_names[i].split('.wav')[0]

    # initialize the output csv for this audio file
    output_csv = pd.DataFrame(columns=['Method', 'Best MSE', 'Best MSE std', 'Elapsed Time', 'Elapsed Time std'])

    # load the audio
    target_audio, target_sr = librosa.load(target_audio_path, sr=daw_settings['SAMPLE_RATE'])

    # determine the length of the target audio in seconds
    target_audio_length = librosa.get_duration(y=target_audio, sr=target_sr)

    # determine the target note
    ff = get_fundamental_frequency(target_audio, target_sr)
    target_note = get_closest_note_from_ff(ff[1])

    # extract features from audio
    target_mfcc = librosa.feature.mfcc(y=target_audio, sr=target_sr).reshape(-1)

    for method in methods:
        # create the temporary output list
        best_fitness_vals = []
        elapsed_times = []

        for j in range(NUM_TRIALS):
            print(f'\n\nRunning trial {j+1} of {NUM_TRIALS} for method {method} on target {target_name}\n\n')
            if method == 'null':
                # set the daw_settins initial population parameter to be False
                ga_settings['use_initial_population'] = False

                # use GA to find the closest preset
                output = optimize_preset_with_ga_mfcc(None,plugin,engine,target_mfcc,target_audio_length,target_note,daw_settings,ga_settings,verbosity=1)

                # define the history
                history = output['plot_history']

                # plot the history
                fig = make_parameter_with_fitness_plots(history,plot_flag=False)

                # save the plot as a pdf
                output_folder = os.path.join(".", "sandbox", "ga-hypothesis-test-may-22","fitness-plots")
                output_file_name = f'{target_name}-null-trial-{j+1}.pdf'
                fig_path = os.path.join(output_folder,output_file_name)
                fig.savefig(fig_path)

                # save the output as a torch pt
                output_folder = os.path.join(".", "sandbox", "ga-hypothesis-test-may-22")
                output_file_name = f'{target_name}-null-trial-{j+1}.pt'
                output_path = os.path.join(output_folder, output_file_name)
                torch.save(history, output_path)

                # append the results to the corresponding vals list
                best_fitness_vals.append(np.min(history['f(x)']))
                elapsed_times.append(history['elapsed min'])
            elif method == 'alternate':
                # set the daw_settins initial population parameter to be True
                ga_settings['use_initial_population'] = True
                
                # start the timer
                start_time = time.time()

                # apply the search algorithm
                top_presets = find_closest_preset_from_mfcc(target_audio, target_sr, df, num_presets=ga_settings['sol_per_pop'], return_note=False, verbosity=VERBOSITY)

                # return the top 10 preset rows in order from the df
                topnpreset_rows = df[df['preset_names'].isin(top_presets)]

                # order the rows based on the order of the top10presets
                topnpreset_rows = topnpreset_rows.set_index('preset_names').loc[top_presets].reset_index()

                # the function below takes the following arguments: top10preset_rows, plugin, engine, target_mfcc, target_audio_length, target_note, ga_settings, daw_settings, print_flag
                output = optimize_preset_with_ga_mfcc(topnpreset_rows,plugin,engine,target_mfcc,target_audio_length,target_note,daw_settings,ga_settings,verbosity=1)

                # stop the timer
                elapsed_time = time.time() - start_time

                # define the history
                history = output['plot_history']

                # plot the history
                fig = make_parameter_with_fitness_plots(history,plot_flag=False)

                # save the plot as a pdf
                output_folder = os.path.join(".", "sandbox", "ga-hypothesis-test-may-22","fitness-plots")
                output_file_name = f'{target_name}-alternate-trial-{j+1}.pdf'
                fig_path = os.path.join(output_folder,output_file_name)
                fig.savefig(fig_path)

                # save the output as a torch pt
                output_folder = os.path.join(".", "sandbox", "ga-hypothesis-test-may-22")
                output_file_name = f'{target_name}-alternate-trial-{j+1}.pt'
                output_path = os.path.join(output_folder, output_file_name)
                torch.save(history, output_path)

                # append the results to the corresponding vals list
                best_fitness_vals.append(np.min(history['f(x)'])) 
                elapsed_times.append(elapsed_time/60)
            else:
                raise ValueError(f'Invalid method: {method}')
            
        # take the mean of the lists and append to the output csv
        new_row = pd.DataFrame({'Method': method, 'Best MSE': np.mean(best_fitness_vals), 'Best MSE std': np.std(best_fitness_vals), 'Elapsed Time': np.mean(elapsed_times), 'Elapsed Time std': np.std(elapsed_times)}, index=[0])
        output_csv = pd.concat([output_csv, new_row], ignore_index=True)
    
    # save the output csv
    output_path = os.path.join(output_folder,f'{target_name}-results.csv')
    output_csv.to_csv(output_path, index=False)