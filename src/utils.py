import numpy as np
import torch
import scipy.signal as signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
import librosa as lb
import librosa.display as lbd
import os
import random
import dawdreamer as daw
import json
import xmltodict
import difflib
import re
import sounddevice as sd

# make the same function below, but add documentation in the style of the rest of the code
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
        piano_note (str): A string representation of a piano note (e.g. 'C4').
    
    Returns:
        int: The MIDI note number corresponding to the input piano note.
    """
    # Define a dictionary that maps note names to their corresponding MIDI note numbers
    note_to_midi = {
        "C0": 12, "C#0": 13, "Db0": 13, "D0": 14, "D#0": 15, "Eb0": 15, "E0": 16, "F0": 17, "F#0": 18, "Gb0": 18, "G0": 19, "G#0": 20, "Ab0": 20, "A0": 21, "A#0": 22, "Bb0": 22, "B0": 23,
        "C1": 24, "C#1": 25, "Db1": 25, "D1": 26, "D#1": 27, "Eb1": 27, "E1": 28, "F1": 29, "F#1": 30, "Gb1": 30, "G1": 31, "G#1": 32, "Ab1": 32, "A1": 33, "A#1": 34, "Bb1": 34, "B1": 35,
        "C2": 36, "C#2": 37, "Db2": 37, "D2": 38, "D#2": 39, "Eb2": 39, "E2": 40, "F2": 41, "F#2": 42, "Gb2": 42, "G2": 43, "G#2": 44, "Ab2": 44, "A2": 45, "A#2": 46, "Bb2": 46, "B2": 47,
        "C3": 48, "C#3": 49, "Db3": 49, "D3": 50, "D#3": 51, "Eb3": 51, "E3": 52, "F3": 53, "F#3": 54, "Gb3": 54, "G3": 55, "G#3": 56, "Ab3": 56, "A3": 57, "A#3": 58, "Bb3": 58, "B3": 59,
        "C4": 60, "C#4": 61, "Db4": 61, "D4": 62, "D#4": 63, "Eb4": 63, "E4": 64, "F4": 65, "F#4": 66, "Gb4": 66, "G4": 67, "G#4": 68, "Ab4": 68, "A4": 69, "A#4": 70, "Bb4": 70, "B4": 71,
        "C5": 72, "C#5": 73, "Db5": 73, "D5": 74, "D#5": 75, "Eb5": 75, "E5": 76, "F5": 77, "F#5": 78, "Gb5": 78, "G5": 79, "G#5": 80, "Ab5": 80, "A5": 81, "A#5": 82, "Bb5": 82, "B5": 83, "C6": 84, "C#6": 85, "Db6": 85, "D6": 86, "D#6": 87, "Eb6": 87, "E6": 88, "F6": 89, "F#6": 90, "Gb6": 90, "G6": 91,
        "G#6": 92, "Ab6": 92, "A6": 93, "A#6": 94, "Bb6": 94, "B6": 95, "C7": 96, "C#7": 97, "Db7": 97, "D7": 98, "D#7": 99, "Eb7": 99, "E7": 100, "F7": 101, "F#7": 102, "Gb7": 102, "G7": 103,
        "G#7": 104, "Ab7": 104, "A7": 105, "A#7": 106, "Bb7": 106, "B7": 107, "C8": 108, "C#8": 109, "Db8": 109, "D8": 110, "D#8": 111, "Eb8": 111, "E8": 112, "F8": 113, "F#8": 114, "Gb8": 114, "G8": 115,
        "G#8": 116, "Ab8": 116, "A8": 117, "A#8": 118, "Bb8": 118, "B8": 119, "C9": 120, "C#9": 121, "Db9": 121, "D9": 122, "D#9": 123, "Eb9": 123, "E9": 124, "F9": 125, "F#9": 126, "Gb9": 126, "G9": 127
    }

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
            exceptions = {
                'volume':'master volume', 
                'octavetranspose':'master octave transpose',
                'adsrdecay':'decay',
                'adsrsustain':'sustain',
                'adsrrelease':'release',
                'chorus1enable':'chorus 1',
                'chorus2enable':'chorus 2',
                'midiclocksync':'clock sync',
                'miditriggerarp16sync':'trigger arp by midi channel 16'
                }

            if key.split('@')[-1] not in exceptions: # find the closest match automatically           
                # Find the closest match in the plugin parameter name list using max() and difflib.SequenceMatcher
                closest_match = max(param_name_to_index.keys(), key=lambda param_name: difflib.SequenceMatcher(None, key, param_name).ratio())

                if key.split('@')[-1][0] == closest_match[0]: # only continue if the first letters are the same and specified exceptions
                    if verbose:
                        print(f'match found for {key}; closest match: {closest_match}')
                    # Extract the value of the JSON key from the JSON string using regex
                    match_value = re.search(r'"{}":\s*"([\d.]+)"'.format(key), preset_settings)
                    if match_value:
                        param_value = float(match_value.group(1))
                        index = param_name_to_index[closest_match]
                        parameter_mapping[key] = {'match': closest_match, 'value': param_value, 'index': index}
                else:
                    if verbose:
                        print(f'no match found for {key}; closest match: {closest_match}')
            else:
                # map manually
                key_temp = key.split('@')[-1]

                # get closest_match from exceptions list
                closest_match = exceptions[key_temp]

                # Extract the value of the JSON key from the JSON string using regex
                match_value = re.search(r'"{}":\s*"([\d.]+)"'.format(key), preset_settings)
                if match_value:
                    param_value = float(match_value.group(1))
                    index = param_name_to_index[closest_match]

                parameter_mapping[key] = {'match': closest_match, 'value': param_value, 'index': index}
        
        
        with open(output_name, 'w') as outfile:
            json.dump(parameter_mapping, outfile)  

    return output_name      

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

def get_western_scale(octaves,freq_low:str,freq_high:str):
    """
    This function returns a dictionary that contains the musical notes in the Western scale in the frequency range specified by freq_low and freq_high.
    
    Parameters:
    octaves (int): the number of octaves in the Western scale
    freq_low (str): the lowest note in the frequency range, in the format "C0", "C#0", "D0", etc.
    freq_high (str): the highest note in the frequency range, in the format "C0", "C#0", "D0", etc.
    
    Returns:
    Dict[str, float]: a dictionary that maps musical notes to their corresponding frequencies in Hz
    """
    # Define the list of musical notes in the Western scale
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    # Define the base frequency for A0
    A0 = 27.5
    # Calculate the frequency of each note in the Western scale
    western_scale = {}
    for octave in range(8):
        for i, note in enumerate(notes):
            key = note + str(octave)
            freq = A0 * 2**(i/12 + octave)
            if key == freq_low:
                L_freq = freq
            elif key == freq_high:
                H_freq = freq
            if L_freq <= freq <= H_freq:
                western_scale[key] = freq
    return western_scale

def get_fundamental_frequency(reference_signal,fs,freq_low:str='C1',freq_high:str='C5'):
    """
    This function computes the likely fundamental frequency of the input signal.
    
    Parameters:
    reference_signal (ndarray): the input signal
    fs (float): the sample rate of the input signal, in Hz
    freq_low (str, optional): the lowest note in the frequency range to consider, in the format "C0", "C#0", "D0", etc. Default is "C1".
    freq_high (str, optional): the highest note in the frequency range to consider, in the format "C0", "C#0", "D0", etc. Default is "C5".
    
    Returns:
    Tuple[str,float]: the likely fundamental frequency of the input signal, in the format ("C0", float), etc.
    """
    n_samples = len(reference_signal)
    time = np.linspace(0,n_samples/fs,n_samples)
    scale = get_western_scale(8,freq_low,freq_high)
    freq_range = scale.values
    note_range = scale.keys
    sinusoid_specs = [spectrogram(np.sin(2*np.pi*f)) for f in freq_range]
    ref_spec = spectrogram(reference_signal)
    # add normalization?
    mse = mse(ref_spec,sinusoid_specs)
    idx = np.argmin(mse)
    likely_note, likely_frequency = note_range[idx], freq_range[idx]
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
    sample_rate = 44100
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