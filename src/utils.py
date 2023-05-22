import numpy as np
import torch
import scipy.signal as signal
from scipy.io import wavfile
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import librosa as lb
import librosa.display as lbd
import os
import random
import dawdreamer as daw
import json
import xmltodict
import re
import librosa
import pandas as pd
import sounddevice as sd
import pygad
import time
import xml.etree.ElementTree as ET
import plotly.graph_objs as go
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.display import Audio, display
from src.config import *

# set plt settings to use latex
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 12
})

def save_params_to_pjunoxl(history, synth_name, preset_folder=daw_settings['PRESET_FOLDER']):
    """
    This function takes a history object containing synth parameters and objective function values,
    and a user-provided synth name. It finds the synth configuration with the best fitness score,
    modifies a preset XML file using these parameters, and saves it.

    Args:
        history (torch.Tensor): A history object containing 'x' (synth params) and 'f(x)' (objective function values).
        synth_name (str): A user-provided name for the synth.
        preset_folder (str, optional): The path of the folder to save the preset to. 
                                        Defaults to daw_settings['PRESET_FOLDER'].

    Returns:
        None. Prints a message indicating success or failure of the save operation.
    """
    # Check the length of the mapping
    assert len(dawdreamer_param_name_to_xml_key_mapping) == NUM_TAL_UNO_PARAMETERS, f"The length of dawdreamer_param_name_to_xml_key_mapping {len(dawdreamer_param_name_to_xml_key_mapping)} is not equal to the NUM_TAL_UNO_PARAMETERS ({NUM_TAL_UNO_PARAMETERS})!"

    # Find the best synth configuration
    best_fitness_index = np.argmin(history['f(x)'])
    synth_parameter_vector = history['x'][best_fitness_index][0:NUM_TAL_UNO_PARAMETERS]

    # Set the preset filename
    preset_filename = f'{synth_name}.pjunoxl'
    preset_folder_path = os.path.join(preset_folder, 'ml-presets')
    os.makedirs(preset_folder_path, exist_ok=True)

    # Define preset path
    preset_path = os.path.join(preset_folder_path, preset_filename)
    dd_param_names = list(dawdreamer_param_name_to_tal_uno_index_mapping.keys())

    # Load and parse the default preset
    default_preset_path = os.path.join(preset_folder, 'Default.pjunoxl')
    with open(default_preset_path, 'r') as f:
        default_preset_string = f.read()
    root = ET.fromstring(default_preset_string)

    # Modify the preset
    programs = root.find('programs')
    program = programs.find('program')
    program.set('path', preset_path)
    program.set('programname', preset_filename.split('.')[0])

    for i in range(NUM_TAL_UNO_PARAMETERS):
        param_name = dd_param_names[i]
        param_value = synth_parameter_vector[i]
        xml_key = dawdreamer_param_name_to_xml_key_mapping[param_name]
        if xml_key in program.attrib:
            program.set(xml_key, str(param_value))
        else:
            raise ValueError(f"xml_key {xml_key} not found in program.attrib!")

    # Save the preset
    try:
        new_tree = ET.ElementTree(root)
        new_tree.write(preset_path)
        print("Preset file saved successfully!")
    except Exception as e:
        print(f"Failed to save preset file. Error: {e}")

    return None

def save_mfcc_history_comparison_plots(history, save_path):
    for i, generation in enumerate(history['generation']):
        if i % 10 != 0:
            continue # this will only plot every 25 generations
        # Create the figure and the two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

        # generation fitness
        generation_fitness = history['f(x)'][i]

        # Add a title for the whole figure that contains the generation number and the fitness
        fig.suptitle(f'Generation {generation} Best Fitness: {generation_fitness}')

        # First subplot
        im1 = ax1.imshow(history['target mfcc'], origin='lower', aspect='auto')
        ax1.set_title('Target MFCC')
        ax1.set_xlabel('Time (frames)')
        ax1.set_ylabel('MFCC Coefficient')

        # Second subplot
        im2 = ax2.imshow(history['synth mfcc'][i], origin='lower', aspect='auto')
        ax2.set_title('Best Solution MFCC')
        ax2.set_xlabel('Time (frames)')
        ax2.set_ylabel('MFCC Coefficient')

        # Create an axes on the right side of ax2 for the colorbar
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        # Add colorbar to the figure
        fig.colorbar(im1, cax=cax)

        plt.tight_layout()
        
        fig.savefig(f'{save_path}{os.sep}generation-{generation}.png')

        plt.close()

def make_parameter_with_fitness_plots(history, plot_flag=True, objective_function_type=OBJECTIVE_FUNCTION):
    # get the proper indices
    optimize_indices, fixed_indices = get_optimize_indices()

    # specify the variable params
    x_var = [x[np.array(optimize_indices.tolist() + [NUM_TAL_UNO_PARAMETERS, NUM_TAL_UNO_PARAMETERS + 1])] for x in history['x']]
    
    # Create subplots
    fig, axes = plt.subplots(1,2,figsize=(16,8))
    
    # Plot for each dimension in the parameter vector
    for d in range(len(x_var[0])):
        axes[0].plot([x[d] for x in history['x']], label=f'x[{d}]')
    
    axes[0].set_xlabel('Generation')
    axes[0].set_ylabel('Parameter Values')
    axes[0].set_title('Parameter Values vs. Generation')
        
    # Set label for objective function based on type
    if objective_function_type == 'MAE':
        axes[1].plot(history['f(x)'], label='$f(x) = T(y) - T(\hat{y}(x))$')
    elif objective_function_type == 'MSE':
        axes[1].plot(history['f(x)'], label='$f(x) = \sum (T(y) - T(\hat{y}(x)))^2$')
    
    # Set labels for axes
    axes[1].set_xlabel('Generation')
    axes[1].set_ylabel('Objective Function Values')
    axes[1].set_title('Objective Function Values vs. Generation')
    
    # Add legend and tight layout
    plt.legend()
    plt.tight_layout()
    
    # Show plots
    if plot_flag:
        plt.show()
    
    return fig

def load_synth_from_dataset(plugin, engine, top_preset_row, verbosity=VERBOSITY):
    """
    This function initializes the DawDreamer engine with a specified sample rate,
    loads a plugin into the engine, sets parameters on the plugin using values from 
    a dataset row, and returns the plugin as a loaded synth.

    Args:
        plugin (object): The DawDreamer plugin to be loaded.
        engine (object): The DawDreamer engine to be initialized.
        top_preset_row (DataFrame row): The row of a DataFrame containing parameter values 
                                        and their corresponding names to be set on the plugin.
        target_sample_rate (int): The sample rate to be set on the engine.
        verbosity (int, optional): The verbosity level. Defaults to 0.

    Returns:
        object: The plugin, now a loaded synth with parameters set from the dataset.
    """
    # Loop through parameters and set them in the synth
    for i, parameter in enumerate(top_preset_row['parameters'].iloc[0]):
        index = top_preset_row['mapped_parameter_names'].iloc[0][i]['dawdreamer index']
        mapped_param_value = top_preset_row['mapped_parameter_names'].iloc[0][i]['value']
        assert mapped_param_value == parameter, f'Parameter value mismatch: {mapped_param_value} != {parameter}!'
        param_name = top_preset_row['mapped_parameter_names'].iloc[0][i]['dawdreamer param name']
        print(f'Param Name: {param_name}, Param Value: {parameter}') if verbosity > 0 else None
        plugin.set_parameter(index, parameter)
    
    # Return the plugin (now it's a loaded synth)
    return plugin

def set_parameters(plugin, parameters):
    """
    This function sets parameters on a plugin using values from a dataset row.

    Args:
        plugin (object): The DawDreamer plugin to be loaded.
        parameters (list): A list of parameter values to be set on the plugin.

    Returns:
        object: The plugin, now a loaded synth with parameters set from the dataset.
    """
    # Loop through parameters and set them in the synth
    for i, parameter in enumerate(parameters):
        index = tal_uno_to_dawdreamer_index_mapping[i]
        plugin.set_parameter(index, parameter)
    
    # Return the plugin (now it's a loaded synth)
    return plugin

def render_audio(midi_piano_note, velocity, midi_duration, loaded_synth, engine, target_audio_length, target_sample_rate, mono=True, verbosity=VERBOSITY):
    """
    This function renders audio for a given MIDI note, using a loaded synthesizer and engine.

    Args:
        midi_piano_note (int): MIDI note number (0-127).
        velocity (int): Velocity of the MIDI note (0-127).
        midi_duration (float): Duration of the MIDI note (0.0-1.0).
        loaded_synth (DawDreamer.Synth): Loaded synthesizer object with a preset.
        engine (DawDreamer.Engine): Engine object used to render the audio.
        target_audio_length (float): The target length of the audio to be generated.
        target_sample_rate (int): The sample rate of the target audio.
        mono (bool, optional): Whether to return mono audio. Defaults to True.
        verbosity (int): Verbosity level for printing.

    Returns:
        audio (np.array): The generated audio.
    """
    # clear the midi notes
    loaded_synth.clear_midi()

    # generate a sound using the plugin (MIDI note, velocity, start sec, duration sec)
    loaded_synth.add_midi_note(midi_piano_note, velocity, 0.0, midi_duration)

    # load the plugin into the engine
    engine.load_graph([(loaded_synth, [])])

    # Render the sound
    engine.render(target_audio_length)

    # get the audio from the engine
    audio = engine.get_audio()

    # reshape the audio depending on mono or stereo
    audio = np.mean(audio, axis=0) if mono and audio.ndim > 1 and audio.shape[0] == 2 else audio

    # print the shape of the audio
    print(f'Audio shape: {audio.shape}') if verbosity >= 2 else None

    return audio

def render_audio_and_generate_mfcc(midi_piano_note, velocity, midi_duration, loaded_synth, engine, target_audio_length, target_sample_rate, verbosity=VERBOSITY, return_audio=False):
    """
    This function generates Mel-frequency cepstral coefficients (MFCCs) for a given MIDI note, using a loaded synthesizer and engine.

    Args:
        midi_piano_note (int): MIDI note number (0-127).
        velocity (int): Velocity of the MIDI note (0-127).
        midi_duration (float): Duration of the MIDI note (0.0-1.0).
        loaded_synth (DawDreamer.Synth): Loaded synthesizer object with a preset.
        engine (DawDreamer.Engine): Engine object used to render the audio.
        target_audio_length (float): The target length of the audio to be generated.
        target_sample_rate (int): The sample rate of the target audio.
        verbosity (int): Verbosity level for printing.

    Returns:
        current_mfcc (np.array): The MFCCs of the generated audio.
    """
    # render the audio
    audio = render_audio(midi_piano_note, velocity, midi_duration, loaded_synth, engine, target_audio_length, target_sample_rate, verbosity=VERBOSITY)

    # Extract the MFCC features from the generated sound using librosa
    current_mfcc = librosa.feature.mfcc(y=audio, sr=target_sample_rate) # use default settings

    # print the shape of the MFCCs
    print(f'MFCC shape: {current_mfcc.shape}') if verbosity >= 2 else None

    # reshape the MFCCs to be 1D
    current_mfcc = current_mfcc.reshape(-1)

    return current_mfcc if not return_audio else (current_mfcc, audio)

def get_midi_durations(df_rows):
    """
    Compute the MIDI durations for a list of dataset rows.

    Args:
        df_rows (list): A list of dataset rows.
    
    Returns:
        midi_durations (list): A list of MIDI durations.
    """
    initial_parameters = [x['parameters'] for i, x in df_rows.iterrows()]
    midi_durations = []
    for i, row in df_rows.iterrows():
        parameter_mapped_names = [x['dawdreamer param name'] for x in row['mapped_parameter_names']]
        attack_idx = parameter_mapped_names.index('attack')
        attack_value = initial_parameters[i][attack_idx]
        midi_duration = scale_midi_duration_by_attack(attack_value)
        assert midi_duration < 1, "midi_duration (how long the midi note is played for) must be less than 1 second"
        midi_durations.append(midi_duration)
    
    return midi_durations

def objective_func(current_mfcc, target_mfcc, type=OBJECTIVE_FUNCTION):
    type = type.lower()
    if type == 'mae':
        return np.linalg.norm(target_mfcc - current_mfcc, ord=1)/len(target_mfcc) # mean absolute error (MAE)
    elif type == 'mse':
        return np.mean((target_mfcc - current_mfcc)**2) # mean squared error (MSE)
    else:
        raise ValueError(f'Invalid type: {type}')

def obtain_parameter_objective_space_video_data(plugin, engine, target_mfcc, target_audio_length, closest_note, daw_settings, top_preset_row, master_parameter_name, verbosity=VERBOSITY):
    # obtain the tal uno index of the master parameter
    master_parameter_index = dawdreamer_param_name_to_tal_uno_index_mapping[master_parameter_name]
    
    # define the closest midi note from the closest note obtained previously
    midi_piano_note = piano_note_to_midi_note(closest_note)

    # Initialize x as the preset parameters
    x_init = torch.tensor(top_preset_row['parameters'].iloc[0], requires_grad=False)

    # add midi parameters to the x_init tensor
    x_init = torch.cat((x_init, torch.tensor([INITIAL_MIDI_VELOCITY,INITIAL_MIDI_DURATION])), dim=0)

    # create a clone for future use
    x = x_init.clone()

    # get the optimized and fixed indices for the Tal Uno
    optimize_indices, fixed_indices = get_optimize_indices()

    # create the master_param_range
    master_param_range = np.linspace(0,1,50)

    # create a linspace vector that spans from 0 to 1 with a step size of 0.001
    param_range = np.linspace(0,1,100)

    # create a master list to store the plot_data from each iteration
    master_plot_data = {
        'master param name':master_parameter_name,
        'video frames': [],             
    }
    for j, matser_param_value in enumerate(master_param_range):
        plot_data = []
        c = 0
        print(f'------------ MASTER PARAM ITERATION {j} STARTED -----------\n\n')
        for i in range(len(x)):
            x_temp = x_init.clone()
            temp_out = {
                'param_name': None, # placeholder
                'param_values': [],
                'objective_function_values':[],
                }
            if i not in fixed_indices:
                for param_value in param_range:
                    x_temp[i] = param_value if i != master_parameter_index else matser_param_value
                    # Apply the solution to the synthesizer
                    loaded_synth_temp = set_parameters(plugin, x_temp[0:NUM_TAL_UNO_PARAMETERS].numpy())

                    # convert the midi parameters to the correct format
                    midi_velocity = midi_velocity_float_to_int(x_temp[NUM_TAL_UNO_PARAMETERS].item())
                    midi_duration = midi_duration_float_to_sec(x_temp[NUM_TAL_UNO_PARAMETERS+1].item())

                    # Generate the sound and extract its MFCC features
                    current_mfcc_temp, audio = render_audio_and_generate_mfcc(midi_piano_note, midi_velocity, midi_duration, loaded_synth_temp, engine, target_audio_length, daw_settings['SAMPLE_RATE'], verbosity=VERBOSITY, return_audio=True)
                    
                    # write the audio to a wav file
                    wavfile.write(f'audio_{i}.wav', daw_settings['SAMPLE_RATE'], audio) if verbosity == 4 else None

                    # objective function variable assignment
                    current_objective = objective_func(current_mfcc_temp, target_mfcc)   

                    # get the parameter name assoicated with the current parameter
                    if i < NUM_TAL_UNO_PARAMETERS:
                        param_name = top_preset_row['mapped_parameter_names'].iloc[0][i]['dawdreamer param name']
                    elif i == NUM_TAL_UNO_PARAMETERS:
                        param_name = 'midi velocity'
                    elif i == NUM_TAL_UNO_PARAMETERS + 1:
                        param_name = 'midi duration'
                    else:
                        raise ValueError(f"Invalid index: {i}")

                    temp_out['param_name'] = param_name
                    temp_out['param_values'].append(param_value)
                    temp_out['objective_function_values'].append(current_objective)

                    print('Data appended...') if verbosity >= 3 else None
                c += 1
                print(f'Iteration {c} complete for param {param_name}...') if verbosity >= 1 else None
                plot_data.append(temp_out)
        master_plot_data['video frames'].append(plot_data)
    return plot_data

def obtain_parameter_objective_space(plugin, engine, target_mfcc, target_audio_length, closest_note, daw_settings, top_preset_row, verbosity=VERBOSITY):
    midi_piano_note = piano_note_to_midi_note(closest_note)
    midi_duration = get_midi_durations(top_preset_row)[0]

    # Initialize x as the preset parameters
    x_init = torch.tensor(top_preset_row['parameters'].iloc[0], requires_grad=False)
    x = x_init.clone()

    # create a linspace vector that spans from 0 to 1 with a step size of 0.001
    param_range = np.linspace(0,1,250)

    plot_data = []
    for i in range(len(x)):
        x_temp = x_init.clone()
        temp_out = {
            'param_name': None, # placeholder
            'param_values': [],
            'objective_function_values':[],
            }
        for param_value in param_range:
            x_temp[i] = param_value
            # Apply the solution to the synthesizer
            loaded_synth_temp = set_parameters(plugin, x_temp.numpy())
            # Generate the sound and extract its MFCC features
            current_mfcc_temp, audio = render_audio_and_generate_mfcc(midi_piano_note, 127, midi_duration, loaded_synth_temp, engine, target_audio_length, daw_settings['SAMPLE_RATE'], verbosity=VERBOSITY, return_audio=True)
            
            # write the audio to a wav file
            wavfile.write(f'audio_{i}.wav', daw_settings['SAMPLE_RATE'], audio) if verbosity == 4 else None

            # objective function variable assignment
            current_objective = objective_func(current_mfcc_temp, target_mfcc)   

            # get the parameter name assoicated with the current parameter
            param_name = top_preset_row['mapped_parameter_names'].iloc[0][i]['dawdreamer param name']

            temp_out['param_name'] = param_name
            temp_out['param_values'].append(param_value)
            temp_out['objective_function_values'].append(current_objective)

            print('Data appended...') if verbosity >= 3 else None

        print(f'Iteration {i} complete for param {param_name}...') if verbosity >= 1 else None
        plot_data.append(temp_out)
        
    return plot_data

def optimize_preset_with_greedy_gradient_mfcc(plugin, engine, target_mfcc, target_audio_length, closest_note, daw_settings, top_preset_row, N_ITERATIONS=greedy_gradient_settings['num_generations'], DX=greedy_gradient_settings['dx_size'], learn_rate=greedy_gradient_settings['learn_rate'], verbosity=VERBOSITY):
    """
    Optimizes synthesizer parameters using a greedy gradient to match a target sound's MFCC features.

    Parameters:
        plugin (Plugin): Plugin object related to the synthesizer settings.
        engine (Engine): Engine object related to the synthesizer settings.
        target_mfcc (numpy.ndarray): Target sound's Mel-frequency cepstral coefficients (MFCC) features.
        target_audio_length (float): Target sound's length in seconds.
        closest_note (str): The closest note to the target sound's pitch, i.e. "C4".
        daw_settings (dict): Dictionary containing DawDreamer settings. Expected keys include 'SAMPLE_RATE' and 'BLOCK_SIZE'.
        top_preset_row (Series): DataFrame row containing the top preset from the dataset.
        N_ITERATIONS (int): Number of iterations for RFD.
        DX (float): The absolute value of dx.
        learn_rate (float): The learning rate for the optimization.
        verbosity (int): Verbosity level. 0 = no print statements, 1 = print some statements, 2 = print all statements.

    Returns:
        dict: The best solution found by the greedy gradient, i.e., the set of synthesizer parameters 
            that results in the sound closest to the target sound (according to the MFCC features).
    """
    # specify the number of dimensions of the parameter
    N_DIM = NUM_TAL_UNO_PARAMETERS + NUM_MIDI_PARAMETERS

    # initialize the parameter vector
    x = torch.zeros((N_DIM,))

    # initialize x as the preset parameters
    x[0:NUM_TAL_UNO_PARAMETERS] = torch.tensor(top_preset_row['parameters'].iloc[0])

    # initialize the midi parameters
    x[-NUM_MIDI_PARAMETERS::] = torch.tensor([INITIAL_MIDI_VELOCITY, INITIAL_MIDI_DURATION])

    # specify the midi piano note
    midi_piano_note = piano_note_to_midi_note(closest_note)

    # get the indices of the parameters that are fixed and optimized
    optimize_indices, fixed_indices = get_optimize_indices()    

    # define midi indices
    midi_indices = [NUM_TAL_UNO_PARAMETERS + 0, NUM_TAL_UNO_PARAMETERS + 1]

    # Initialize the history dictionary
    history = {'x': [], 'f(x)': [],'target mfcc': target_mfcc.reshape(20,87), 'synth mfcc':[], 'elapsed time': None, 'generation':[]}

    # Initialize the x differential step
    num_variable_params = len(optimize_indices) + NUM_MIDI_PARAMETERS

    # begin the timer
    start_time = time.time()

    # Optimization loop
    for i in range(N_ITERATIONS):
        # specify the synth params
        synth_params = x[0:NUM_TAL_UNO_PARAMETERS]

        # specify the midi parameters
        midi_params = x[-NUM_MIDI_PARAMETERS::]

        # convert MIDI parameters to their corresponding values
        midi_duration = midi_duration_float_to_sec(midi_params[1])
        midi_velocity = midi_velocity_float_to_int(midi_params[0])

        # Apply the solution to the synthesizer
        loaded_synth = set_parameters(plugin, synth_params.numpy())
        
        # Generate the sound and extract its MFCC features
        current_mfcc, audio_old = render_audio_and_generate_mfcc(midi_piano_note, midi_velocity, midi_duration, loaded_synth, engine, target_audio_length, daw_settings['SAMPLE_RATE'], verbosity=VERBOSITY, return_audio=True)
        
        # write the audio to a wav file
        wavfile.write(f'audio_old.wav', daw_settings['SAMPLE_RATE'], audio_old) if verbosity == 4 else None

        # compute the function value at the current point
        objective = objective_func(current_mfcc, target_mfcc)

        history['f(x)'].append(objective.item())
        history['x'].append(x.numpy())
        history['generation'].append(i + 1)
        history['synth mfcc'].append(current_mfcc.reshape(20,87))

        # print the current generation number, objective function value
        print(f'Generation {i+1} Started. Current Objective function value: {objective}...') if verbosity >= 1 else None

        # compute the black box gradient
        grad = []
        for i in range(N_DIM):
            if i not in fixed_indices:
                x_temp = x.clone()
                x_temp[i] += DX # nudge the ith parameter a bit

                # Apply the solution to the synthesizer
                loaded_synth_temp = set_parameters(plugin, x_temp[0:NUM_TAL_UNO_PARAMETERS].numpy())

                # define the midi velocity and duration
                midi_duration = midi_duration_float_to_sec(x_temp[-1])
                midi_velocity = midi_velocity_float_to_int(x_temp[-2])

                # Generate the sound and extract its MFCC features
                current_mfcc_temp, audio = render_audio_and_generate_mfcc(midi_piano_note, midi_velocity, midi_duration, loaded_synth_temp, engine, target_audio_length, daw_settings['SAMPLE_RATE'], verbosity=VERBOSITY, return_audio=True)
                
                # write the audio to a wav file
                wavfile.write(f'audio_{i}.wav', daw_settings['SAMPLE_RATE'], audio) if verbosity == 4 else None

                grad.append(objective_func(current_mfcc_temp, target_mfcc) - objective) # compute the gradient
            else:
                grad.append(0)
            
        # compute the gradient step
        deltax = -learn_rate * torch.tensor(grad) * DX

        # clip the gradient
        deltax = gradient_clip(deltax, magnitude=1)
        
        x += deltax # take a step in the good direction!

        # clip the parameters
        x = torch.clamp(x, 0, 1)    
    
    # end the timer
    end_time = time.time()

    # compute the elapsed time
    elapsed_time = end_time - start_time

    # store the elapsed time
    history['elapsed time'] = elapsed_time
    
    return history

def gradient_clip(delta,magnitude):
    """
    Clips the gradient according to the magnitude.

    Parameters:
        delta (np.ndarray): The gradient.
        magnitude (float): The magnitude of the gradient.
    """
    if np.linalg.norm(delta) > magnitude:
        delta = delta / np.linalg.norm(delta) * magnitude
    return delta


def optimize_preset_with_rfd_mfcc(plugin, engine, target_mfcc, target_audio_length, closest_note, daw_settings, top_preset_row, N_ITERATIONS=100, DX=0.0001, verbosity=VERBOSITY):
    """
    Optimizes synthesizer parameters using a random finite difference (RFD) to match a target sound's MFCC features.

    Parameters:
        plugin (Plugin): Plugin object related to the synthesizer settings.
        engine (Engine): Engine object related to the synthesizer settings.
        target_mfcc (numpy.ndarray): Target sound's Mel-frequency cepstral coefficients (MFCC) features.
        target_audio_length (float): Target sound's length in seconds.
        closest_note (str): The closest note to the target sound's pitch, i.e. "C4".
        daw_settings (dict): Dictionary containing DawDreamer settings. Expected keys include 'SAMPLE_RATE' and 'BLOCK_SIZE'.
        top_preset_row (Series): DataFrame row containing the top preset from the dataset.
        N_ITERATIONS (int): Number of iterations for RFD.
        DX (float): The absolute value of dx.
        verbosity (int): Verbosity level. 0 = no print statements, 1 = print some statements, 2 = print all statements.

    Returns:
        dict: The best solution found by the RFD, i.e., the set of synthesizer parameters 
            that results in the sound closest to the target sound (according to the MFCC features).
    """
    # specify the number of dimensions of the parameter space
    N_DIM = NUM_TAL_UNO_PARAMETERS + NUM_MIDI_PARAMETERS  # the number of dimensions

    # initialize the parameter vector
    x = np.zeros((N_DIM,))

    # Initialize x as the preset parameters
    x[0:NUM_TAL_UNO_PARAMETERS] = np.array(top_preset_row['parameters'].iloc[0])

    # initialize the midi parameters
    x[-NUM_MIDI_PARAMETERS::] = np.array([INITIAL_MIDI_VELOCITY, INITIAL_MIDI_DURATION])

    # create a copy of x
    x_init = x.copy()

    # get the indices of the parameters that are fixed and not fixed
    optimize_indices, fixed_indices = get_optimize_indices()

    # specify the midi piano note
    midi_piano_note = piano_note_to_midi_note(closest_note)

    # Initialize the history dictionary
    history = {'preset name': top_preset_row['preset_names'].iloc[0], 'x': [], 'f(x)': [], 'x_star': [], 'f(x)_star': [], 'best_mfcc': [], 'target_mfcc': [target_mfcc]}
    
    # initialize best f(x)
    best_fx = np.inf

    for i in range(N_ITERATIONS):
        # specify the synth params
        synth_params = x[0:NUM_TAL_UNO_PARAMETERS]

        # specify the midi params
        midi_params = x[-NUM_MIDI_PARAMETERS::]

        # convert midi duration to seconds
        midi_duration = midi_duration_float_to_sec(midi_params[1])

        # convert midi velocity to integer
        midi_velocity = midi_velocity_float_to_int(midi_params[0])

        # Load the synth with current parameters
        loaded_synth = set_parameters(plugin, synth_params)

        # Generate the sound and extract its MFCC features
        current_mfcc = render_audio_and_generate_mfcc(midi_piano_note, midi_velocity, midi_duration, loaded_synth, engine, target_audio_length, daw_settings['SAMPLE_RATE'], verbosity=VERBOSITY)

        # Compute initial f(x)
        f_start = objective_func(current_mfcc, target_mfcc)

        # Add the current value of x and f(x) to the history dictionary
        history['x'].append(x)
        history['f(x)'].append(f_start)

        # Randomly choose the sign of dx for each dimension
        dx_signs = np.random.choice([-1, 1], size=len(optimize_indices) + NUM_MIDI_PARAMETERS)

        # Specify randomly signed dx for each dimension
        dx = dx_signs * DX

        print(f'dx: {dx}') if verbosity >= 3 else None

        # create the perturbed x
        x_perturbed = np.zeros(len(optimize_indices) + NUM_MIDI_PARAMETERS)
        x_perturbed[0:len(optimize_indices)] = x[optimize_indices] + dx[0:len(optimize_indices)] # perturb the synth parameters
        x_perturbed[-NUM_MIDI_PARAMETERS::] = x[-NUM_MIDI_PARAMETERS::] + dx[-NUM_MIDI_PARAMETERS::] # perturb the midi parameters

        # clip the perturbed x to the range [0, 1]
        x_perturbed = np.clip(x_perturbed, 0, 1)

        # specify the new synth params
        new_synth_params = np.zeros(NUM_TAL_UNO_PARAMETERS)
        new_synth_params[optimize_indices] = x_perturbed[0:len(optimize_indices)]
        new_synth_params[fixed_indices] = x_init[fixed_indices]

        # specify the new midi params
        new_midi_params = x_perturbed[-NUM_MIDI_PARAMETERS::]

        # convert midi duration to seconds
        midi_duration = midi_duration_float_to_sec(new_midi_params[1])

        # convert midi velocity to integer
        midi_velocity = midi_velocity_float_to_int(new_midi_params[0])

        # Load the synth with updated parameters
        loaded_synth_dx = set_parameters(plugin, new_synth_params)

        # Generate the sound and extract its MFCC features
        current_mfcc_dx = render_audio_and_generate_mfcc(midi_piano_note, midi_velocity, midi_duration, loaded_synth_dx, engine, target_audio_length, daw_settings['SAMPLE_RATE'], verbosity=VERBOSITY)

        # Compute new f(x)
        f_new = objective_func(current_mfcc_dx, target_mfcc) 

        # Compute the difference in f(x) due to the change in x by dx
        df = f_new - f_start

        print(f'f_start: {f_start}, f_rfd: {f_new}, df: {df}') if verbosity >= 1 else None

        # step in the opposite direction of the gradient
        delta_x = - dx * np.sign(df)

        print(f'delta_x: {delta_x}') if verbosity >= 3 else None

        # update x
        x[optimize_indices] = x[optimize_indices] + delta_x[0:len(optimize_indices)] # update the synth parameters
        x[-NUM_MIDI_PARAMETERS::] = x[-NUM_MIDI_PARAMETERS::] + delta_x[-NUM_MIDI_PARAMETERS::] # update the midi parameters

        # clip x to be in the range [0, 1]
        x = np.clip(x, 0, 1)

        # add the best x and f(x) to the history dictionary
        if f_start < best_fx:
            print('Saving checkpoint...')
            best_fx = f_start
            history['x_star'] = x
            history['f(x)_star'] = f_start
            history['best_mfcc'] = current_mfcc

    return history  

def get_optimize_indices():
    bad_param_keys = list(fixed_parameters.keys())
    all_param_keys = list(dawdreamer_param_name_to_tal_uno_index_mapping.keys())
    bad_indices = []
    good_indices = []
    for bad_param in bad_param_keys:
        for param in all_param_keys:
            if bad_param == param:
                bad_indices.append(dawdreamer_param_name_to_tal_uno_index_mapping[param])
    good_indices = np.setdiff1d(list(dawdreamer_param_name_to_tal_uno_index_mapping.values()), bad_indices)
    return good_indices, np.array(bad_indices)

def midi_duration_float_to_sec(midi_duration_float):
    # convert from 0 to 1 to 0.25 to 0.9 seconds
    midi_duration = midi_duration_float * 0.65 + 0.25
    return midi_duration

def midi_velocity_float_to_int(midi_velocity_float):
    # convert from 0 to 1 to 60 to 127
    midi_velocity = int(midi_velocity_float * 67 + 60)
    return midi_velocity

def optimize_preset_with_ga_mfcc(top10_preset_rows, plugin, engine, target_mfcc, target_audio_length, closest_note, daw_settings, ga_settings, verbosity=VERBOSITY):
    """
    Optimizes synthesizer parameters using a genetic algorithm (GA) to match a target sound's MFCC features.

    Parameters:
        top10_preset_rows: DataFrame rows containing the top 10 presets from the dataset.
        plugin (Plugin): Plugin object related to the synthesizer settings.
        engine (Engine): Engine object related to the synthesizer settings.
        target_mfcc (numpy.ndarray): Target sound's Mel-frequency cepstral coefficients (MFCC) features.
        target_audio_length (float): Target sound's length in seconds.
        closest_note (str): The closest note to the target sound's pitch, i.e. "C4".
        daw_settings (dict): Dictionary containing DawDreamer settings. Expected keys include 'SAMPLE_RATE' and 'BLOCK_SIZE'.
        ga_settings (dict): Dictionary containing GA settings. Expected keys are 'num_generations', 
                            'num_parents_mating', 'sol_per_pop', 'crossover_type', 'mutation_type', 
                            and 'mutation_percent_gene'.
        verbosity (int): Verbosity level. 0 = no print statements, 1 = print some statements, 2 = print all statements.

    Returns:
        numpy.ndarray: The best solution found by the GA, i.e., the set of synthesizer parameters 
                    that results in the sound closest to the target sound (according to the MFCC features).

    The function uses the PyGAD library to run a GA where the fitness of each solution (set of synthesizer 
    parameters) is determined by how close the sound it generates is to the target sound. The closeness 
    is measured as the Euclidean distance between the MFCC features of the generated and target sounds.

    The GA uses steady state selection for parent selection, uniform crossover, and random mutation. 
    It keeps the best solution from each generation.
    """
    # print the ga_settings dictionary as a check
    print(f'\n\n\n GA SETTINGS: \n\n {ga_settings}') if verbosity >= 1 else None

    # Load initial parameters from top preset
    if ga_settings['use_initial_population'] == True:
        initial_parameters = [row['parameters'] for i, row in top10_preset_rows.iterrows()] # 2D list, 10 rows, each list has 77 parameters (for Tal Uno LX)
    else:
        initial_parameters = [np.random.uniform(0, 1, NUM_TAL_UNO_PARAMETERS) for i in range(ga_settings['sol_per_pop'])] # 2D list, 10 rows, each list has 77 parameters (for Tal Uno LX)
    
    # obtain the indices of the parameters that we want to optimize from variables set in src.config
    optimize_indices, fixed_indices = get_optimize_indices()

    # print the dict values corresponding to fixed_indices as a check
    print({k: list(dawdreamer_param_name_to_tal_uno_index_mapping.keys())[k] for k in list(fixed_indices)}) if verbosity == 0.5 else None

    # check shape of indices
    print(f'optimize_indices: {optimize_indices.shape}, fixed_indices: {fixed_indices.shape}') if verbosity == 0.5 else None

    # post-process the initial parameters to remove parameters that we don't want to optimize
    fixed_parameters = [np.array(param_vector)[fixed_indices] for param_vector in initial_parameters]
    initial_parameters = [np.array(param_vector)[optimize_indices] for param_vector in initial_parameters]

    # add midi note and duration to the initial parameters
    initial_parameters = [np.append(param_vector, [INITIAL_MIDI_VELOCITY, INITIAL_MIDI_DURATION]) for param_vector in initial_parameters]

    # check the shape of one of the initial parameters
    print(f'initial_parameters[0] shape: {initial_parameters[0].shape}') if verbosity == 0.5 else None

    assert len(initial_parameters) == ga_settings['sol_per_pop'], "There must be 10 initial parameters."
    assert len(set([len(x) for x in initial_parameters])) == 1, "All initial parameters must have the same length."

    # Generate the sound and extract its MFCC features
    midi_piano_note = piano_note_to_midi_note(closest_note)

    # get the midi duration by converting
    midi_duration = midi_duration_float_to_sec(0)

    # get the midi velocity by converting
    midi_velocity = midi_velocity_float_to_int(0)
    
    # make initial_best_params a global variable so it can be accessed by the on_generation callback
    global initial_best_params

    # specify the initial best parameters
    initial_best_params = top10_preset_rows.iloc[0]['parameters']

    # obtain the name of the top preset
    top_preset_name = top10_preset_rows.iloc[0]['preset_names']

    # load the best preset to the synthesizer plugin
    loaded_synth = set_parameters(plugin, initial_best_params)

    # print the name of the top preset
    print(f'top preset name: {top_preset_name}') if verbosity >= 1 else None

    # get the starting mfcc to calculate the starting fitness
    initial_mfcc = render_audio_and_generate_mfcc(midi_piano_note, midi_velocity, midi_duration, loaded_synth, engine, target_audio_length, daw_settings['SAMPLE_RATE'])

    # calculate the initial fitness
    initial_fitness = -objective_func(initial_mfcc, target_mfcc)

    # print the initial fitness
    print(f'initial_fitness: {initial_fitness}') if verbosity >= 1 else None

    # test_mfcc
    test_mfcc = 0

    # Define callback function for each generation
    plot_history = {'x':[], 'f(x)':[], 'mean f(x)':[], 'synth mfcc':[], 'target mfcc': None, 'generation':[], 'synth params':[], 'generation time':[]}
    def on_generation(ga_instance):      
        # specify the generation number
        generation_number = ga_instance.generations_completed

        # append to plot_history
        plot_history['generation'].append(generation_number)

        # Define the fitness values
        fitness_vals = ga_instance.last_generation_fitness

        # print shape of fitness_vals
        print(f'fitness_vals.shape: {fitness_vals.shape}') if verbosity == 0.5 else None
    
        # Calculating the mean fitness
        mean_fitness = np.mean(fitness_vals)

        # Append to mean f(x)
        plot_history['mean f(x)'].append(-mean_fitness)

        # Get the best solution
        best_solution_params = ga_instance.best_solution(ga_instance.last_generation_fitness)[0]
        best_solution_fit = ga_instance.best_solution(ga_instance.last_generation_fitness)[1]
        best_match_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)[2]

        print(f'best_solution_params.shape: {best_solution_params.shape}') if verbosity == 1 else None

        # Append to f(x)
        plot_history['f(x)'].append(-best_solution_fit)

        print(f'Generation = {ga_instance.generations_completed}, Best Fitness = {best_solution_fit}, Mean Fitness = {mean_fitness}') if verbosity >= 0.5 else None

        # obtain the midi duration for this solution
        midi_params = best_solution_params[-2::]

        # convert the midi from 0 to 1 to integer value
        midi_duration = midi_duration_float_to_sec(midi_params[1])
        midi_velocity = midi_velocity_float_to_int(midi_params[0])

        # create the parameter vector based on
        synth_params = np.zeros((NUM_TAL_UNO_PARAMETERS,))

        # Create the full solution, combining the fixed and variable parameters
        synth_params[optimize_indices] = best_solution_params[:-2]
        synth_params[fixed_indices] = fixed_parameters[best_match_idx]
        # apply the best solution to the synthesizer
        loaded_synth = set_parameters(plugin, synth_params)

        # Generate the sound and extract its MFCC features
        test_mfcc, test_audio = render_audio_and_generate_mfcc(midi_piano_note, midi_velocity, midi_duration, loaded_synth, engine, target_audio_length, daw_settings['SAMPLE_RATE'], verbosity=VERBOSITY, return_audio=True)

        # save the audio to an mp3
        if SAVE_AUDIO:
            if not os.path.exists('./audio'):
                os.mkdir('./audio')
            wavfile.write(f'./audio/synth_gen:{generation_number}.mp3', daw_settings['SAMPLE_RATE'], test_audio)

        # calculate the fitness of the best solution
        test_fitness = -objective_func(test_mfcc, target_mfcc)

        # append the full x vector to plot history
        plot_history['x'].append(np.concatenate((synth_params,midi_params)))

        # append test mfcc and target mfcc to plot history
        plot_history['synth mfcc'].append(test_mfcc.reshape(20,87))
        plot_history['target mfcc'] = target_mfcc.reshape(20,87)

        # append the synth params to plot history
        plot_history['synth params'].append(synth_params)

        # print the fitness of the best solution
        print(f'test_fitness: {test_fitness}') if verbosity >= 3 else None
        if verbosity == 3:
            # Create the figure and the two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

            # Add a title for the whole figure that contains the generation number and the fitness
            fig.suptitle(f'Generation {generation_number} Best Fitness: {test_fitness}')

            # First subplot
            im1 = ax1.imshow(target_mfcc.reshape(20,87), origin='lower', aspect='auto')
            ax1.set_title('Target MFCC')

            # Second subplot
            im2 = ax2.imshow(test_mfcc.reshape(20,87), origin='lower', aspect='auto')
            ax2.set_title('Best Solution MFCC')

            # Create an axes on the right side of ax2 for the colorbar
            divider = make_axes_locatable(ax2)
            cax = divider.append_axes("right", size="5%", pad=0.05)

            # Add colorbar to the figure
            fig.colorbar(im1, cax=cax)

            plt.tight_layout()
            plt.show()
        
    # Define the fitness function
    def fitness_func(self, solution, solution_idx):
        # obtain the midi duration for this solution
        midi_params = solution[-2::]
        print(f'midi params shape: {midi_params.shape}') if verbosity == 0.5 else None

        # convert the midi from 0 to 1 to integer value
        midi_duration = midi_duration_float_to_sec(midi_params[1])
        midi_velocity = midi_velocity_float_to_int(midi_params[0])

        # as a check, print the solution preset name and its duration
        print(f'Solution {solution_idx} original preset name: {top10_preset_rows.iloc[solution_idx]["preset_names"]}, midi duration: {midi_duration}') if verbosity >= 2 else None

        # create the parameter vector based on
        synth_params = np.zeros((NUM_TAL_UNO_PARAMETERS,))

        # print solution[:-2] shape
        print(f'solution[:-2] shape: {solution[:-2].shape}') if verbosity == 0.5 else None

        # Create the full solution, combining the fixed and variable parameters
        synth_params[optimize_indices] = solution[:-2]
        synth_params[fixed_indices] = fixed_parameters[solution_idx]

        # Apply the solution to the synthesizer
        loaded_synth = set_parameters(plugin, synth_params)
        
        # Generate the sound and extract its MFCC features (the arguments below in order: midi note, velocity, midi duration, loaded synth, engine, target audio length, target sample rate)
        current_mfcc, audio = render_audio_and_generate_mfcc(midi_piano_note, midi_velocity, midi_duration, loaded_synth, engine, target_audio_length, daw_settings['SAMPLE_RATE'], verbosity=VERBOSITY, return_audio=True)        

        # Calculate the difference between the current and target MFCC
        fitness = -objective_func(current_mfcc, target_mfcc)

        # check the shape of the MFCC features
        print(f'Current MFCC shape: {current_mfcc.shape}, Target MFCC shape: {target_mfcc.shape}, Fitness: {fitness}') if verbosity >= 2 else None

        return fitness

    # Create a GA instance
    ga_instance = pygad.GA(num_generations=ga_settings['num_generations'],
                           num_parents_mating=ga_settings['num_parents_mating'],
                           sol_per_pop=ga_settings['sol_per_pop'],
                           fitness_func=fitness_func,
                           mutation_by_replacement=ga_settings['mutation_by_replacement'],
                           random_mutation_min_val=ga_settings['random_mutation_min_val'],
                           random_mutation_max_val=ga_settings['random_mutation_max_val'],
                           mutation_probability=ga_settings['mutation_rate'],
                           mutation_type=ga_settings['mutation_type'],
                           mutation_percent_genes=ga_settings['mutation_percent_gene'],                           initial_population=initial_parameters,
                           init_range_low=0,
                           init_range_high=1,
                           parent_selection_type=ga_settings['parent_selection_type'],
                           keep_elitism=int(ga_settings['elitism_percent']/100*ga_settings['sol_per_pop']),
                           crossover_type=ga_settings['crossover_type'],
                           crossover_probability=ga_settings['crossover_probability'],
                           on_generation=on_generation
                           )

    # Run the GA
    start_time = time.time()
    ga_instance.run()
    elapsed_time = time.time() - start_time

    # add the elapsed time variable to the plot history dictionary
    plot_history['elapsed min'] = elapsed_time / 60

    # Get the best solution
    best_solution, best_solution_fitness, best_match_idx = ga_instance.best_solution()

    output = {
        'best_solution': best_solution,
        'best_solution_fitness': best_solution_fitness,
        'best_match_idx': best_match_idx,
        'plot_history': plot_history,
    }

    return output

def find_closest_preset_from_mfcc(target_audio, target_sr, dataset, num_presets=10, return_note=False, verbosity=VERBOSITY):
    """
    Given the path to a target audio file and a dataset of presets, this function finds and returns the names of the 
    top n presets whose MFCCs are closest to that of the target audio.

    The function calculates the MFCCs (Mel-Frequency Cepstral Coefficients) for the target audio and each audio in the 
    preset dataset. It then uses the Euclidean distance to determine the similarity between the MFCCs of the target audio 
    and each audio in the preset dataset. The presets corresponding to the top 10 closest MFCCs are returned.

    Parameters:
    target_audio (np.ndarray): The loaded target audio clip
    target_sr (int): The sample rate of the target audio clip
    dataset (pandas.DataFrame): The dataset of TAL-U-NO-LX presets.

    Returns:
    numpy.ndarray: An array containing the names of the top 10 presets that are the closest to the target audio file 
                   based on the MFCC feature. The presets are sorted from closest to farthest.
    """
    # Find the Fundamental Frequency of the Target Audio
    ff = get_fundamental_frequency(target_audio, target_sr)

    # Find which of C2, C3, C4 the FF is Closest to
    closest_note = get_closest_note_from_ff(ff[1])

    # Define the comparison vectors
    comparison_audio_set = np.stack([x[closest_note] for x in dataset['raw_audio']])

    # Adapt the comparison audio set to the target audio if their lengths are not equal
    if comparison_audio_set.shape[1] < target_audio.shape[0]:
        comparison_audio_set = np.pad(comparison_audio_set, ((0, 0), (0, target_audio.shape[0] - comparison_audio_set.shape[1])), 'constant', constant_values=0)
    elif comparison_audio_set.shape[1] > target_audio.shape[0]:
        comparison_audio_set = comparison_audio_set[:, :target_audio.shape[0]]

    # Evaluate MFCC Timbral Feature

    # Define the comparison vectors
    comparison_mfcc_set = np.stack([librosa.feature.mfcc(y=x, sr=daw_settings['SAMPLE_RATE']).reshape(-1) for x in comparison_audio_set])

    # Define the target mfcc
    target_mfcc = librosa.feature.mfcc(y=target_audio, sr=target_sr).reshape(-1)

    # Find where the euclidean distance is the smallest
    distances = [objective_func(current_mfcc, target_mfcc) for current_mfcc in comparison_mfcc_set]

    # Find the top 10 closest presets
    top_n = np.argsort(distances)[:num_presets]

    # print the best distance achieved
    print(f'Best Distance Achieved: {distances[top_n[0]]}') if verbosity >= 1 else None

    # Return the preset names of the top 10 closest presets
    return dataset.iloc[top_n]["preset_names"].values if not return_note else (dataset.iloc[top_n]["preset_names"].values, ff[0])


def get_cosine_similarity(vector_a, vector_b):
    # Calculate the dot product
    dot_product = np.dot(vector_a, vector_b)

    # Calculate the norms (lengths) of each vector
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    # Calculate the cosine similarity
    similarity = dot_product / (norm_a * norm_b)

    return similarity

def scale_midi_duration_by_attack(attack_value,min_midi_duration=0.4,max_midi_duration=0.9):
    """
    Scale the midi duration from 0.4 seconds to 0.9 seconds depending on the attack parameter.

    Parameters
        min_midi_duration : float
            Minimum midi duration.
        max_midi_duration : float
            Maximum midi duration.
        attack_value : float
            Attack value of the preset, between 0 and 1.
    
    Returns
        midi_duration : float
            Scaled midi duration.
    """
    midi_duration = min_midi_duration + (max_midi_duration - min_midi_duration) * attack_value
    return midi_duration

def play_audio(raw_signal_numpy, sr=daw_settings['SAMPLE_RATE']):
    display(Audio(raw_signal_numpy, rate=sr))

def create_custom_scatter(df, x_col, y_col, label_col, audio_col):
    fig = go.Figure()

    for label in df[label_col].unique():
        filtered_df = df[df[label_col] == label]
        scatter = go.Scatter(
            x=filtered_df[x_col],
            y=filtered_df[y_col],
            mode="markers",
            name=label,
            text=filtered_df.index,
            customdata=filtered_df[audio_col],
            hovertemplate="Index: %{text}<extra></extra>",
        )
        fig.add_trace(scatter)

    fig.update_layout(title="TSNE Plot", hovermode="closest")

    def on_hover(trace, points, state):
        if points.point_inds:
            index = points.point_inds[0]
            raw_audio = trace.customdata[index]
            play_audio(raw_audio)

    fig.data[0].on_hover(on_hover)

    return fig

def normalize_zero_to_one(data, min_value=None, max_value=None):
    """
    Normalize the data to be between 0 and 1.

    Parameters
        data : numpy.ndarray
            Data to normalize.
    
    Returns
        normalized_data : numpy.ndarray
            Normalized data.
    """
    if min_value is None and max_value is None:
        normalized_data = (data - torch.min(data)) / (torch.max(data) - torch.min(data))
    else:
        normalized_data = (data - min_value) / (max_value - min_value)
    return normalized_data

def create_muscinn_preset_dataset(preset_path:str,synth_plugin,synth_name,sample_rate,buffer_size,piano_notes,preset_ext,extractor,verbose=False):
    """
    Create a dataset of audio files with different synth presets.

    Parameters
        preset_path : str
            Path to the preset folder.
        synth_plugin : str
            Path to the plugin.
        synth_name : str
            Name of the plugin.
        sample_rate : int
            Sampling rate of the audio.
        buffer_size : int
            Buffer size of the audio.
        piano_notes : list of str
            List of piano notes to generate audio for.
        preset_ext : str
            Extension of the preset files.
        extractor : function
            Musicnn function to extract the parameters from the preset.
        verbose : bool, optional
            Print information about the process. Default is False.
    
    Returns
        preset_dataset : dict
            Dictionary of audio files with different synth presets.

    """
    # create a RenderEngine object
    engine = daw.RenderEngine(sample_rate=sample_rate, block_size=buffer_size)
    
    # load plugin with dawdreamer
    plugin = load_plugin_with_dawdreamer(synth_plugin,synth_name,engine)

    # create a dictionary to store the audio files
    preset_dataset = {
        'preset_names':[],
        'parameters':[],
        'parameters_names':[],
        'mapped_parameter_names':[],
        'raw_audio':[],
        'musicnn_features':[],
    }

    # get a full list of presets to iterate over
    preset_paths = []
    for root, dirs, files in os.walk(preset_path):
        for file in files:
            if file.endswith(preset_ext):
                preset_paths.append(os.path.join(root, file))

    # iterate over the presets
    for preset_path in preset_paths:
        # get the preset name
        preset_name = os.path.basename(preset_path).split('.')[0]

        # obtain the parameter mapping and save to json file
        json_file_location = make_json_parameter_mapping(plugin,preset_path,verbose=verbose)

        # obtain the parameters and their names
        parameter_names, parameter_mapped, parameter_values  = get_parameter_lists(json_file_location) 

        # load the preset to the synth
        loaded_preset_synth = load_xml_preset(plugin, json_file_location)

        # create a dictionary to store the audio files
        preset_dataset['preset_names'].append(preset_name)

        # create a dictionary to store the audio files
        preset_dataset['raw_audio'].append({})

        # create a dictrionary to store the MFCCs
        preset_dataset['musicnn_features'].append({})

        # create a list to store the parameters
        preset_dataset['parameters'].append(parameter_values)

        # create a list to stor the parameter names
        preset_dataset['parameters_names'].append(parameter_names)

        # create a list to store the mapped parameter names
        preset_dataset['mapped_parameter_names'].append(parameter_mapped)

        # scale the midi duration from 0.4 to 0.9 depending on the attack parameter
        # convert parameter_mapped, which is a list of dictionaries, to a list of strings corresponding to the key of 'match'
        parameter_mapped_names = [x['match'] for x in parameter_mapped]
        attack_idx = parameter_mapped_names.index('attack')
        attack_value = parameter_values[attack_idx]
        midi_duration = scale_midi_duration_by_attack(attack_value)
        assert midi_duration < 1, "midi_duration (how long the midi note is played for) must be less than 1 second"


        # iterate over the piano notes
        for piano_note in piano_notes:
            # convert the piano note to midi (0 to 127)
            midi_piano_note = piano_note_to_midi_note(piano_note)

            # clear the midi notes
            loaded_preset_synth.clear_midi()

            # generate a sound using the plugin (MIDI note, velocity, start sec, duration sec)
            loaded_preset_synth.add_midi_note(midi_piano_note, 127, 0.0, midi_duration)

            engine.load_graph([(loaded_preset_synth, [])])

            # loaded_preset_synth.open_editor()
            engine.render(1) # have to render audio for 3 seconds because of musicnn audio length requirement
            
            # render the audio
            audio = engine.get_audio()

            # pad the audio with 2 seconds of the last value in audio to make it 3 seconds long. Note that audio has shape (2, n_samples)
            padding = ((0, 0), (0, sample_rate*3 - audio.shape[1]))
            audio = np.pad(audio, padding, 'constant', constant_values=(audio[-1, -1], audio[-1, -1]))            

            # save the file temporarily
            file_name = f'temp{piano_note}.wav'
            wavfile.write(file_name, sample_rate, audio[0,:])

            # obtain timbral features from sound using pre-trained neural network (pons 2018)
            taggram, tag, features = extractor(file_name, model='MTT_musicnn', extract_features=True)

            # remove the wav file
            os.remove(file_name)

            # store the audio in the dictionary
            preset_dataset["raw_audio"][-1][piano_note] = torch.tensor(audio[0,:],dtype=torch.float32)

            # store the MFCC in the dictionary
            preset_dataset["musicnn_features"][-1][piano_note] = torch.tensor(features['timbral'][0,:],dtype=torch.float32)

    # save the dataset as a torch file
    torch.save(preset_dataset, 'preset_dataset_musicnn.pt')

    return preset_dataset

def generate_tsne(data, perplexity=15, learning_rate=200, random_state=42):
    """
    Reduce the dimensionality of the data using t-SNE to 2 dimensions.

    Parameters
        data : numpy.ndarray
            Data to reduce the dimensionality of.
        perplexity : int, optional
            Perplexity of the t-SNE algorithm. Default is 40.
        learning_rate : int, optional
            Learning rate of the t-SNE algorithm. Default is 200.
        random_state : int, optional
            Random state of the t-SNE algorithm. Default is 42.
    
    Returns
        transformed_data : numpy.ndarray
            Data with reduced dimensionality.        
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, random_state=random_state)
    transformed_data = tsne.fit_transform(data)
    return transformed_data

# add documentation for this code in the same style sa the rest of the code
def categorize_name(name):
    """
    Put the preset name into an instrumental category.

    Parameters
        name : str
            Name of the preset.
    
    Returns
        category : str
            Instrumental category of the preset.
    
    Examples
    --------
    >>> categorize_name("PAD 1")
    "Synth Pads"

    """    
    for category, keywords in tal_uno_categories.items():
        for keyword in keywords:
            if keyword in name:
                return category
                
    return "Miscellaneous/Other"

def load_plugin_with_dawdreamer(synth_plugin,synth_name,engine):
    """
    Load a plugin with dawdreamer.

    Parameters
        synth_plugin : str
            Path to the plugin.
        synth_name : str
            Name of the plugin.
        sample_rate : int
            Sampling rate of the audio.
        buffer_size : int
            Buffer size of the audio.
    
    Returns
        plugin : dawdreamer.plugin.PluginProcessor
            Plugin object. See dawdreamer documentation for more information.

    """
    # create the plugin object
    plugin = engine.make_plugin_processor(synth_name, synth_plugin)
    assert plugin.get_name() == synth_name

    return plugin

def create_preset_dataset(preset_path:str,synth_plugin,synth_name,sample_rate,buffer_size,piano_notes,midi_duration,preset_ext,verbose=False):
    """
    Create a dataset of audio files with different synth presets.

    Parameters
        preset_path : str
            Path to the preset folder.
        synth_plugin : str
            Path to the plugin.
        synth_name : str
            Name of the plugin.
        sample_rate : int
            Sampling rate of the audio.
        buffer_size : int
            Buffer size of the audio.
        piano_notes : list of str
            List of piano notes to generate audio for.
        midi_duration : int
            Duration of the audio in seconds.
        preset_ext : str
            Extension of the preset files.
        verbose : bool, optional
            Print information about the process. Default is False.
    
    Returns
        preset_dataset : dict
            Dictionary of audio files with different synth presets.

    """
    assert midi_duration < 1, "midi_duration (how long the midi note is played for) must be less than 1 second"
    # create a RenderEngine object
    engine = daw.RenderEngine(sample_rate=sample_rate, block_size=buffer_size)
    
    # load plugin with dawdreamer
    plugin = load_plugin_with_dawdreamer(synth_plugin,synth_name,engine)

    # create a dictionary to store the audio files
    preset_dataset = {
        'preset_names':[],
        'raw_audio':[],
        'MFCCs':[],
    }

    # get a full list of presets to iterate over
    preset_paths = []
    for root, dirs, files in os.walk(preset_path):
        for file in files:
            if file.endswith(preset_ext):
                preset_paths.append(os.path.join(root, file))

    # iterate over the presets
    for preset_path in preset_paths:
        # get the preset name
        preset_name = os.path.basename(preset_path).split('.')[0]

        # obtain the parameter mapping and save to json file
        json_file_location = make_json_parameter_mapping(plugin,preset_path,verbose=verbose)

        # load the preset to the synth
        loaded_preset_synth = load_xml_preset(plugin, json_file_location)

        # create a dictionary to store the audio files
        preset_dataset['preset_names'].append(preset_name)

        # create a dictionary to store the audio files
        preset_dataset['raw_audio'].append({})

        # create a dictrionary to store the MFCCs
        preset_dataset['MFCCs'].append({})

        # iterate over the piano notes
        for piano_note in piano_notes:
            # convert the piano note to midi (0 to 127)
            midi_piano_note = piano_note_to_midi_note(piano_note)

            # generate a sound using the plugin (MIDI note, velocity, start sec, duration sec)
            loaded_preset_synth.add_midi_note(midi_piano_note, 127, 0.0, midi_duration)

            engine.load_graph([(loaded_preset_synth, [])])

            # loaded_preset_synth.open_editor()
            engine.render(1) # use *1.2 to capture release/reverb
            
            # render the audio
            audio = engine.get_audio()

            # compute the MFCC of the audio
            mfcc = lb.feature.mfcc(y=audio[0,:],sr=sample_rate)

            # store the audio in the dictionary
            preset_dataset["raw_audio"][-1][piano_note] = torch.tensor(audio[0,:],dtype=torch.float32)

            # store the MFCC in the dictionary
            preset_dataset["MFCCs"][-1][piano_note] = torch.tensor(mfcc,dtype=torch.float32)

    # save the dataset as a torch file
    torch.save(preset_dataset, 'preset_dataset.pt')

    return preset_dataset
            
def play_audio(audio,sample_rate):
    sd.play(audio, sample_rate)
    sd.wait()

def plot_specs(spectrograms, sample_rate, f_max, plot_size=(2, 2), num_cols=3, f_min=0, hop_length=512, cmap='jet', range_db=80.0, high_boost_db=0.0):
    """
    Plot a grid of spectrograms using librosa and matplotlib.

    Parameters
        spectrograms : list of np.ndarray
            List of spectrograms, where each spectrogram is a 2D numpy array with shape (T, F).
        plot_size : tuple of int, optional
            Size of each spectrogram plot in inches. Default is (4, 4).
        num_cols : int, optional
            Number of columns in the plot grid. Default is 4.
        sample_rate : int, optional
            Sampling rate of the audio. Default is 22050.
        f_min : int, optional
            Minimum frequency for the mel spectrogram. Default is 0.
        f_max : int, optional
            Maximum frequency for the mel spectrogram. Default is None (i.e., sample_rate / 2).
        hop_length : int, optional
            Hop length (in samples) between consecutive frames of the spectrogram. Default is 512.
        cmap : str or matplotlib colormap, optional
            Colormap for the spectrogram plot. Default is 'jet'.
        range_db : float, optional
            Dynamic range of the spectrogram (in decibels). Default is 80.0.
        high_boost_db : float, optional
            High-frequency boost (in decibels) applied to the spectrogram. Default is 10.0.

    Returns
        fig : matplotlib.figure.Figure
            The matplotlib figure object.

    Examples
    --------
    >>> import numpy as np
    >>> from plot_grid import plot_grid
    >>> import librosa
    >>> audio, sr = librosa.load("example.wav", sr=22050)
    >>> spectrograms = [librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=2048, hop_length=512, n_mels=128) for _ in range(10)]
    >>> plot_specs(spectrograms, plot_size=(6, 6), num_cols=3, sample_rate=sr, f_min=0, f_max=8000)
    """

    # Calculate the number of rows needed for the plot grid
    num_rows = int(np.ceil(len(spectrograms) / num_cols))

    # Create a new figure and axis object for the plot
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(num_cols*plot_size[0], num_rows*plot_size[1]))

    # Flatten the axis object if there is only one row
    if num_rows == 1:
        ax = np.array([ax])

    # Loop through the spectrograms and plot each one in the grid
    for i, spectrogram in enumerate(spectrograms):
        row_idx = i // num_cols
        col_idx = i % num_cols
        lbd.specshow(spectrogram, x_axis='time', y_axis='mel', sr=sample_rate, fmin=f_min, fmax=f_max, hop_length=hop_length, cmap=cmap, ax=ax[row_idx, col_idx], vmin=-range_db, vmax=spectrogram.max() + high_boost_db)
        if not(row_idx == 0 and col_idx == 0):
            ax[row_idx, col_idx].set_xticks([])
            ax[row_idx, col_idx].set_yticks([])
            ax[row_idx, col_idx].set_yticklabels([])
            ax[row_idx, col_idx].set_xticklabels([])
            
    plt.show()
    plt.tight_layout()

def check_threshold(audio, threshold):
        """
        Checks if the disproportionality threshold condition is met.
        
        Parameters:
            audio (numpy.ndarray): The audio signal as a NumPy array.
            threshold (float): The disproportionality threshold between 0 and 1.
                        A lower value means the user wants more of the signal to be non-silent.
        
        Returns:
            bool: True if the threshold condition is met, False otherwise.
        """
        silence_threshold = 0.02 * np.max(audio)
        silent_samples = np.sum(np.abs(audio) <= silence_threshold)
        non_silent_samples = np.sum(np.abs(audio) > silence_threshold)
        ratio = silent_samples / (silent_samples + non_silent_samples)
        return ratio <= threshold

def find_duration_by_truncation(NOTE, synth_duration, midi_duration, disproportionality_threshold, engine, synth_plugin):
    """
    Finds the optimal midi duration to use for a synth preset with an iteratively adjusted duration based on a disproportionality threshold.
    
    The function starts with the initial duration provided by the user and shortens it iteratively
    until the disproportionality threshold is met or the duration is too short.
    
    Parameters:
        NOTE (str): The piano note as a string (e.g., "C4", "A#3").
        synth_duration (float): The initial duration of the audio signal in seconds.
        midi_duration (float): The initial duration of the midi note in seconds.
        disproportionality_threshold (float): The disproportionality threshold between 0 and 1.
                                        A lower value means the user wants more of the signal to be non-silent.
                                        
    Returns:
        numpy.ndarray: The optimal duration for the synth plugin preset given as input.
    """
    duration = synth_duration
    while True:
        # Convert the piano note to midi (0 to 127)
        midi_piano_note = piano_note_to_midi_note(NOTE)

        # Generate a sound using the plugin (MIDI note, velocity, start sec, duration sec)
        synth_plugin.add_midi_note(midi_piano_note, 100, 0.0, midi_duration)
        engine.load_graph([(synth_plugin, [])])

        # Render the audio
        engine.render(synth_duration)
        audio = engine.get_audio()

        # Check if the threshold condition is met, or if the duration cannot be shortened further
        if check_threshold(audio, disproportionality_threshold) or midi_duration <= 0.1:
            break

        # Shorten the duration by 10% for the next iteration
        midi_duration *= 0.9

    return midi_duration

def normalize_data(data):
    """
    Apply min-max scaling to a data array.
    """
    min_val = np.min(data)
    max_val = np.max(data)
    data = data - min_val
    data = data / (max_val - min_val)
    return data

def piano_note_to_midi_note(note_name):
    """
    Convert a string representation of a piano note to its corresponding MIDI note number.
    
    Args:
        note_name (str): A string representation of a piano note (e.g. 'C4').
    
    Returns:
        int: The MIDI note number corresponding to the input piano note.
    """
    # Define a dictionary that maps note names to their corresponding MIDI note numbers

    # Convert the input note_name to uppercase
    note_name = note_name.upper()

    # Check if the note_name is in the note_to_midi dictionary
    if note_name in note_to_midi:
        return note_to_midi[note_name]
    else:
        raise ValueError(f"Invalid note name: {note_name}")

def read_txt(path: str) -> str:
    """
    Read the contents of a text file and return as a string.
    
    Args:
        path (str): The path to the text file to be read.
    
    Returns:
        str: The contents of the text file as a string.
    """
    with open(path, 'r') as file:
        txt = file.read()
    return txt

def get_xml_preset_settings(preset_path: str):
    """
    Read a preset file in XML format and convert it to a dictionary.
    
    Args:
        preset_path (str): The path to the preset file.
    
    Returns:
        str: The preset settings in JSON format as a string.
    """
    # read the preset_path using with ... as 'rb' ... etc.
    txt = read_txt(preset_path)
    preset_settings = None

    # Assuming the presence of "<?xml" at the start of the file indicates an XML file
    if txt.strip().startswith("<?xml"):
        try:
            # Convert XML to a dictionary
            xml_dict = xmltodict.parse(txt)

            # Convert the dictionary to JSON (optional)
            preset_settings = json.dumps(xml_dict)
        except Exception as e:
            print(f"Error: {e}")
            raise ValueError("Unable to parse XML file")
    else:
        raise ValueError("Unsupported file type")

    return preset_settings

def make_json_parameter_mapping(plugin, preset_path:str, verbose=True):
    """
    Read a preset file in XML format, apply the settings to the plugin, and create a JSON file
    that maps the preset parameters to the plugin parameters.
    
    Args:
        plugin (dawdreamer.PluginProcessor): The plugin to which the preset settings will be applied.
        preset_path (str): The path to the preset file in XML format.
        verbose (bool): if True, it will print parameter mapping. Default is True.
    
    Returns:
        str: The name of the JSON file containing the parameter mapping.
    """
    # create the json preset folder if it does not already exist
    json_preset_folder = f'TAL-UNO_json_presets'
    if not os.path.exists(json_preset_folder):
        os.mkdir(json_preset_folder)

    # specify the output json filename
    preset_name = preset_path.split(os.sep)[-1].split('.pjunoxl')[0]
    output_name = f'{json_preset_folder}{os.sep}TAL-UNO-{preset_name}-parameter-mapping.json'

    if not os.path.exists(output_name):
        # read the XML preset path
        preset_settings = get_xml_preset_settings(preset_path)

        # apply the synth preset settings to the synth plugin processor object
        parameter_mapping = {}

        # Load JSON settings
        settings = json.loads(preset_settings)

        # Extract the program settings
        json_keys = settings["tal"]["programs"]["program"]

        # Get the parameters description from the plugin
        parameters = plugin.get_parameters_description()

        # Create a dictionary with parameter names as keys and their indices as values
        param_name_to_index = {param["name"]: param["index"] for param in parameters}

        # Iterate over each JSON key
        for key in json_keys:
            # specify the exceptions to map manually
            try:
                # get closest_match from exceptions list
                closest_match = tal_uno_to_dawdreamer_mapping[key]

                if closest_match is not None:
                    # Extract the value of the JSON key from the JSON string using regex
                    match_value = re.search(r'"{}":\s*"([\d.]+)"'.format(key), preset_settings)
                    if match_value:
                        param_value = float(match_value.group(1))
                        index = param_name_to_index[closest_match]
                        parameter_mapping[key] = {'match': closest_match, 'value': param_value, 'index': index}
            except KeyError:
                print(f'Key {key} was not found in mapping dictionary. Continuing...')
        
        with open(output_name, 'w') as outfile:
            json.dump(parameter_mapping, outfile)  

    return output_name

def get_parameter_lists(parameter_mapping_json):
    """
    Get the parameter lists from a JSON file that maps preset parameters to plugin parameters.
    
    Args:
        parameter_mapping_json (str): The path to the JSON file that maps preset parameters to plugin parameters.
    
    Returns:
        list: A list of parameter names.
        list: A list of parameter values.
        list: A list of parameter indices.
    """
    # Load JSON file into a dictionary
    with open(parameter_mapping_json, 'r') as infile:
        parameter_map = json.load(infile)

    # Get the parameter names, values, and indices
    parameter_names = [param for param in parameter_map.keys()]
    parameter_mapped = [param for param in parameter_map.values()]
    parameter_values = [parameter_map[param]['value'] for param in parameter_map.keys()]
    parameter_indices = [parameter_map[param]['index'] for param in parameter_map.keys()]

    return parameter_names, parameter_mapped, parameter_values

def load_xml_preset(dawdreamer_plugin,parameter_mapping_json):
    """
    Load a preset into a plugin using a JSON file that maps preset parameters to plugin parameters.
    
    Args:
        dawdreamer_plugin (dawdreamer.PluginProcessor): The plugin to which the preset settings will be applied.
        parameter_mapping_json (str): The path to the JSON file that maps preset parameters to plugin parameters.
    Returns:
        dawdreamer.PluginProcessor: The plugin with the preset settings applied.
    """
    # Load JSON file into a dictionary
    with open(parameter_mapping_json, 'r') as infile:
        parameter_map = json.load(infile)

    # Get the parameters description from the plugin
    parameters = dawdreamer_plugin.get_parameters_description()

    # Create a dictionary with parameter names as keys and their indices as values
    param_name_to_index = {param["name"]: param["index"] for param in parameters}

    # Iterate over each JSON key
    for key in parameter_map.keys():
        dawdreamer_plugin.set_parameter(parameter_map[key]['index'], parameter_map[key]['value'])
    
    return dawdreamer_plugin

def select_preset_path(folder_path, preset_ext):
    """
    Select a random preset file path from a preset folder path and its subdirectories.
    
    Args:
        folder_path (str): The path to the folder to search for preset files.
        preset_ext (str): The file extension of the preset files to search for (e.g. ".pjunoxl").
    
    Returns:
        str or None: The path to a randomly selected preset file, or None if no preset files are found.
    """
    preset_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(preset_ext):
                preset_files.append(os.path.join(root, file))
    return random.choice(preset_files) if preset_files else None

def wav2linearspec(signal: np.ndarray, sample_rate: int, plot_flag=False, window_size=2048, zero_padding_factor=1,
                     window_type='hann', gain_db=0.0, range_db=80.0, high_boost_db=0.0, f_min=0, f_max=20000):
    """
    Convert a signal to a linear-scaled spectrogram.

    Args:
        signal (np.ndarray): The input signal as a NumPy array.
        sample_rate (int): The sample rate of the input signal.
        plot_flag (bool, optional): Whether to plot the linear-scaled spectrogram. Defaults to False.
        window_size (int, optional): The size of the FFT window to use. Defaults to 2048.
        zero_padding_factor (int, optional): The amount of zero-padding to use in the FFT. Defaults to 1.
        window_type (str, optional): The type of window to use. Defaults to 'hann'.
        gain_db (float, optional): The gain to apply to the spectrogram in decibels. Defaults to 0.0.
        range_db (float, optional): The range of the spectrogram in decibels. Defaults to 80.0.
        high_boost_db (float, optional): The amount of high-frequency boost to apply to the spectrogram in decibels.

    Returns:
        np.ndarray: The linear-scaled spectrogram as a NumPy array.
    """

    # Compute the spectrogram
    spectrogram = librosa.stft(signal, n_fft=window_size * zero_padding_factor, hop_length=window_size,
                                 window=window_type)
    
    # Convert the spectrogram to decibels
    spectrogram_db = librosa.amplitude_to_db(np.abs(spectrogram), ref=np.max, top_db=range_db)

    # Apply gain
    spectrogram_db += gain_db

    # Apply high-frequency boost
    if high_boost_db > 0:
        spectrogram_db[window_size // 2:, :] += high_boost_db
    
    # Plot the spectrogram
    if plot_flag:
        lbd.specshow(spectrogram_db, sr=sample_rate, hop_length=window_size, x_axis='time', y_axis='linear',
                                 fmin=f_min, fmax=f_max)
        plt.colorbar(format='%+2.0f dB')
        plt.show()
    
    return spectrogram_db

def wav2spec(signal: np.ndarray, sample_rate: int, plot_flag=False, window_size=2048, zero_padding_factor=1,
             window_type='hann', gain_db=0.0, range_db=80.0, high_boost_db=0.0, f_min=0, f_max=20000, n_mels=256):
    """
    Convert a signal to a mel-scaled spectrogram.

    Args:
        signal (np.ndarray): The input signal as a NumPy array.
        sample_rate (int): The sample rate of the input signal.
        plot_flag (bool, optional): Whether to plot the mel-scaled spectrogram. Defaults to False.
        window_size (int, optional): The size of the FFT window to use. Defaults to 2048.
        zero_padding_factor (int, optional): The amount of zero-padding to use in the FFT. Defaults to 1.
        window_type (str, optional): The type of window function to use in the FFT. Defaults to 'hann'.
        gain_db (float, optional): The gain to apply to the audio signal in decibels. Defaults to 0.0.
        range_db (float, optional): The range of the mel-scaled spectrogram in decibels. Defaults to 80.0.
        high_boost_db (float, optional): The amount of high-frequency boost to apply to the mel-scaled spectrogram in decibels. Defaults to 0.0.
        f_min (int, optional): The minimum frequency to include in the spectrogram (Hz). Defaults to 0.
        f_max (int, optional): The maximum frequency to include in the spectrogram (Hz). Defaults to 20000.
        n_mels (int, optional): The number of mel frequency bins to include in the spectrogram. Defaults to 256.

    Returns:
        np.ndarray: The mel-scaled spectrogram.
    """

    # Apply gain to the audio signal
    signal = lb.util.normalize(signal) * lb.db_to_amplitude(gain_db)

    # Compute the mel-scaled spectrogram
    fft_size = window_size * zero_padding_factor
    hop_length = window_size // 2
    mel_filterbank = lb.filters.mel(sr=sample_rate, n_fft=fft_size, n_mels=n_mels)
    window = lb.filters.get_window(window_type, window_size, fftbins=True)
    spectrogram = np.abs(lb.stft(signal, n_fft=fft_size, hop_length=hop_length, window=window))**2
    mel_spectrogram = lb.feature.melspectrogram(S=spectrogram, sr=sample_rate, n_mels=n_mels,
                                                 fmax=f_max, htk=True, norm=None)
    mel_spectrogram = lb.power_to_db(mel_spectrogram, ref=np.max)

    # Apply range and high boost to the mel-scaled spectrogram
    mel_spectrogram = np.clip(mel_spectrogram, a_min=-range_db, a_max=None)
    mel_spectrogram = mel_spectrogram + high_boost_db

    # Plot the mel-scaled spectrogram if plot_flag is True
    if plot_flag:
        plt.figure(figsize=(10, 4))
        lb.specshow(mel_spectrogram,x_axis='time', y_axis='mel', sr=sample_rate, fmin=f_min, fmax=f_max, hop_length=hop_length, cmap='jet', vmin=-range_db, vmax=mel_spectrogram.max() + high_boost_db)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel spectrogram')
        plt.tight_layout()
        plt.show()

    return mel_spectrogram


def audio2mel_spectrogram(audio_folder_path, plot_flag=False, window_size=2048, zero_padding_factor=1,
                           window_type='hann', gain_db=0.0, range_db=80.0, high_boost_db=0.0, f_min=0, f_max=20000, n_mels=256):
    """
    Convert a collection of audio files to mel-scaled spectrograms.
    
    Args:
        audio_folder_path (str): The path to the folder containing the audio files.
        plot_flag (bool, optional): Whether to plot the mel-scaled spectrograms. Defaults to False.
        window_size (int, optional): The size of the FFT window to use. Defaults to 2048.
        zero_padding_factor (int, optional): The amount of zero-padding to use in the FFT. Defaults to 1.
        window_type (str, optional): The type of window function to use in the FFT. Defaults to 'hann'.
        gain_db (float, optional): The gain to apply to the audio signal in decibels. Defaults to 0.0.
        range_db (float, optional): The range of the mel-scaled spectrogram in decibels. Defaults to 80.0.
        high_boost_db (float, optional): The amount of high-frequency boost to apply to the mel-scaled spectrogram in decibels. Defaults to 0.0.
        f_min (int, optional): The minimum frequency to include in the spectrogram (Hz). Defaults to 0.
        f_max (int, optional): The maximum frequency to include in the spectrogram (Hz). Defaults to 20000.
        n_mels (int, optional): The number of mel frequency bins to include in the spectrogram. Defaults to 256.
    
    Returns:
        list: A list of mel-scaled spectrograms, where each element is a NumPy array.
    """

    # Get a list of audio file names in the folder
    audio_file_names = os.listdir(audio_folder_path)
    np.random.shuffle(audio_file_names)

    # Compute mel-scaled spectrograms for each audio file
    mel_spectrograms = []
    for file_name in audio_file_names:
        audio_file_path = os.path.join(audio_folder_path, file_name)
        signal, sample_rate = lb.load(audio_file_path)

        # Apply gain to the audio signal
        signal = lb.util.normalize(signal) * lb.db_to_amplitude(gain_db)

        # Compute the mel-scaled spectrogram
        fft_size = window_size * zero_padding_factor
        hop_length = window_size // 2
        mel_filterbank = lb.filters.mel(sr=sample_rate, n_fft=fft_size, n_mels=n_mels)
        window = lb.filters.get_window(window_type, window_size, fftbins=True)
        spectrogram = np.abs(lb.stft(signal, n_fft=fft_size, hop_length=hop_length, window=window))**2
        mel_spectrogram = lb.feature.melspectrogram(S=spectrogram, sr=sample_rate, n_mels=n_mels,
                                                     fmax=f_max, htk=True, norm=None)
        mel_spectrogram = lb.power_to_db(mel_spectrogram, ref=np.max)

        # Apply range and high boost to the mel-scaled spectrogram
        mel_spectrogram = np.clip(mel_spectrogram, a_min=-range_db, a_max=None)
        mel_spectrogram = mel_spectrogram + high_boost_db

        # Plot the mel-scaled spectrogram if plot_flag is True
        if plot_flag:
            plt.figure(figsize=(10, 4))
            # TODO: Need to fix spectrogram visualization frequency axis!
            lbd.specshow(mel_spectrogram, x_axis='time', y_axis='mel',sr=sample_rate, fmin=f_min, fmax=f_max, hop_length=hop_length, cmap='jet', vmin=-range_db, vmax=mel_spectrogram.max() + high_boost_db)
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel spectrogram for {}'.format(file_name))
            plt.tight_layout()
            plt.show()
        
        mel_spectrograms.append(mel_spectrogram)

        if len(mel_spectrograms) > 4:
            break

    return mel_spectrograms

def get_western_scale(freq_low:str,freq_high:str):
    """
    This function returns a dictionary that contains the musical notes in the Western scale in the frequency range specified by freq_low and freq_high.
    
    Parameters:
    freq_low (str): the lowest note in the frequency range, in the format "C0", "C#0", "D0", etc.
    freq_high (str): the highest note in the frequency range, in the format "C0", "C#0", "D0", etc.
    
    Returns:
    Dict[str, float]: a dictionary that maps musical notes to their corresponding frequencies in Hz
    """
    # Define the list of musical notes in the Western scale
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    # Define the base frequency for A0
    C0 = 16.35

    # Initialize L_freq and H_freq
    L_freq = None
    H_freq = float('inf')

    # Calculate the frequency of each note in the Western scale
    western_scale = {}
    for octave in range(8):
        for i, note in enumerate(notes):
            key = note + str(octave)
            freq = C0 * 2**(i/12 + octave)
            if key == freq_low:
                L_freq = freq
            if key == freq_high:
                H_freq = freq
            if L_freq is not None and freq <= H_freq:
                western_scale[key] = freq
    return western_scale

def get_closest_note_from_ff(ff):
    """
    Finds the closest note to the fundamental frequency from the list C2, C3, or C4.
    """
    idx = np.argmin(np.abs(np.array([65.41, 130.81, 261.63]) - ff))

    if idx == 0:
        return 'C2'
    elif idx == 1:
        return 'C3'
    else:
        return 'C4'

def get_spec_mse(spec1, spec2):
    """
    Calculates the MSE between two spectrograms.
    """
    return np.mean((spec1 - spec2)**2)

def get_fundamental_frequency(orig_signal,fs,freq_low:str='C2',freq_high:str='C5'):
    """
    This function computes the likely fundamental frequency of the input signal.
    
    Parameters:
    orig_signal (ndarray): the input signal
    fs (float): the sample rate of the input signal, in Hz
    freq_low (str, optional): the lowest note in the frequency range to consider, in the format "C0", "C#0", "D0", etc. Default is "C1".
    freq_high (str, optional): the highest note in the frequency range to consider, in the format "C0", "C#0", "D0", etc. Default is "C5".
    
    Returns:
    Tuple[str,float]: the likely fundamental frequency of the input signal, in the format ("C0", float), etc.
    """
    # remove parts of the reference signal where there is no sound, within an absolute value tolerance
    reference_signal = orig_signal[np.abs(orig_signal) > 0.01]
    n_samples = len(reference_signal)
    time = np.linspace(0,n_samples/fs,n_samples)
    scale = get_western_scale(freq_low,freq_high)
    freq_range = list(scale.values())
    note_range = list(scale.keys())
    # sinusoid_specs = [wav2linearspec(np.sin(2*np.pi*f*time),fs,window_size=4096) for f in freq_range]
    ref_spec = wav2linearspec(reference_signal,fs,window_size=4096)
    # add normalization?
    # mse = []
    # for sin_spec in sinusoid_specs:
    #     mse.append(get_spec_mse(ref_spec,sin_spec))
    smoothed = np.median(ref_spec,axis=1)
    freqs = np.linspace(0,fs/2,smoothed.shape[0])
    max_freq_idx = np.argmax(smoothed)
    max_freq = freqs[max_freq_idx]
    # find the closest frequency in freq_range to max_freq
    idx = np.argmin(np.abs(np.array(freq_range) - max_freq))
    likely_note, likely_frequency = note_range[idx], freq_range[idx]
    print(f'Likely note: {likely_note}, likely frequency: {likely_frequency}')
    
    # window_size = 1
    # # smoothed = gaussian_filter1d(np.median(ref_spec,axis=1), sigma=window_size)
    # smoothed = np.median(ref_spec,axis=1)

    # # plot the first slice of the reference spectrogram as a spectrum, plot log frequency on x-axis
    # plt.figure(figsize=(10, 4))
    # plt.plot(np.log10(np.linspace(0,fs/2,smoothed.shape[0])),smoothed)
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Amplitude')
    # plt.title('Reference spectrogram slice')
    # plt.show()

    return likely_note, likely_frequency

def find_fundamental_frequency(periodic_signal:np.ndarray,f_s:int) -> float:
    periodic_signal = periodic_signal / np.max(periodic_signal)
    correlations = np.zeros(len(periodic_signal))
    for m in range(len(correlations)):
        shifted = np.roll(periodic_signal,m)
        correlation = np.corrcoef(periodic_signal,shifted)
        correlations[m] = correlation[0,1] # why?
    peak_idx = np.argmax(correlations)
    frequency = f_s / peak_idx
    return frequency


def parse_time_string(time_string:str):
    """
    Take a time string in the format of "minutes:seconds" and convert it to the equivalent
    number of seconds.

    Parameters:
    time_string (str): The input time string in the format of "minutes:seconds".

    Returns:
    int: The equivalent number of seconds.
    """
    time_parts = time_string.split(':')
    total_seconds = int(time_parts[0])*60 + int(time_parts[1])
    return total_seconds

def extract_wav_segment(wav_file:str,time_start:str,time_duration:float) -> dict():
    """
    Take a wav file, a time string for the start of the segment, and a duration in seconds and return
    a dictionary containing the truncated audio signal and the sample rate.

    Parameters:
    wav_file (str): The path to the wav file.
    time_start (str): The time string for the start of the segment in the format of "minutes:seconds".
    time_duration (float): The duration of the segment in seconds.

    Returns:
    dict: A dictionary containing two keys: 'signal' and 'fs'. The value of 'signal' is the truncated audio signal
    and the value of 'fs' is the sample rate.
    """
    signal, fs = lb.load(wav_file) # load the wav file
    start_time_sec = parse_time_string(time_start)
    end_time_sec = start_time_sec + time_duration
    truncated_signal = signal[start_time_sec*fs:end_time_sec*fs]
    out = {
        'signal':truncated_signal,
        'fs':fs
    }
    return out

def adsr_envelope(samples, attack_time, decay_time, sustain_level, release_time, sample_rate):
    """
    Generate an amplitude envelope following the ADSR (Attack-Decay-Sustain-Release) envelope pattern.
    
    Parameters:
    samples (np.ndarray): An array of waveform samples.
    attack_time (float): The time in seconds for the attack phase of the envelope.
    decay_time (float): The time in seconds for the decay phase of the envelope.
    sustain_level (float): The level (amplitude) for the sustain phase of the envelope.
    release_time (float): The time in seconds for the release phase of the envelope.
    sample_rate (int): The sample rate of the waveform samples, in samples per second.
    
    Returns:
    np.ndarray: An array of amplitude values following the ADSR envelope pattern, with the same length as `samples`.
    """
    attack_samples = int(attack_time * sample_rate)
    decay_samples = int(decay_time * sample_rate)
    release_samples = int(release_time * sample_rate)
    
    envelope = np.zeros(len(samples))
    envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    envelope[attack_samples:attack_samples + decay_samples] = np.linspace(1, sustain_level, decay_samples)
    envelope[-release_samples:] = np.linspace(sustain_level, 0, release_samples)
    
    return envelope

def generate_wave(frequency, end_time, amplitude, waveform_type, duty_cycle, attack_time, decay_time, sustain_level, release_time, envelope_type, filter_type, output_file):
    """
    This function generates a waveform based on the specified parameters.
    
    Args:
    frequency (float): frequency of the waveform in Hz
    end_time (float): end time of the waveform in seconds
    amplitude (float): amplitude of the waveform, between 0 and 1
    waveform_type (str): type of waveform to generate (sine, square, sawtooth)
    duty_cycle (float): duty cycle of the square waveform, between 0 and 1
    attack_time (float): attack time of the amplitude envelope in seconds
    decay_time (float): decay time of the amplitude envelope in seconds
    sustain_level (float): sustain level of the amplitude envelope, between 0 and 1
    release_time (float): release time of the amplitude envelope in seconds
    envelope_type (str): type of amplitude envelope to apply (ADSR)
    filter_type (str): type of filter to apply to the waveform (lowpass)
    output_file (str): name of the output WAV file

    Returns:
    None
    """
    # Generate time array
    sample_rate = daw_settings['SAMPLE_RATE']
    N_samples = sample_rate * end_time
    time_array = np.linspace(0, end_time, N_samples)
    
    # Generate waveform samples
    if waveform_type == "sine":
        waveform_samples = amplitude * np.sin(2 * np.pi * frequency * time_array)
    elif waveform_type == "square":
        waveform_samples = amplitude * signal.square(2 * np.pi * frequency * time_array, duty=duty_cycle)
    elif waveform_type == "sawtooth":
        waveform_samples = amplitude * signal.sawtooth(2 * np.pi * frequency * time_array)
    # Add more cases for different waveform types as needed
    
    wavfile.write('test.wav',sample_rate,waveform_samples)

    # Apply amplitude envelope
    if envelope_type == "ADSR":
        # Generate ADSR envelope samples
        envelope = adsr_envelope(waveform_samples, attack_time, decay_time, sustain_level, release_time,sample_rate)
        envelope_samples = waveform_samples * envelope
    # Add more cases for different envelope types as needed
    
    wavfile.write('envelope_samples.wav',sample_rate,envelope_samples)

    # Apply filter
    if filter_type == "lowpass":
        # Apply low-pass filter to waveform samples
        b, a = signal.butter(10, 0.125)
        filtered_samples = signal.filtfilt(b, a, envelope_samples)    # Add more cases for different filter types as needed
    
    wavfile.write('filtered_samples.wav',sample_rate,filtered_samples)

    out = {
        'signal':filtered_samples,
        'fs':sample_rate
    }

    return out