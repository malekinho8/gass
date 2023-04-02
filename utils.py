import numpy as np
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

def piano_note_to_midi_note(piano_note):
    # Define lists of piano note names and their corresponding MIDI note numbers
    piano_notes = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
    midi_notes = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

    # Extract the octave and note name from the piano note input
    octave = int(piano_note[-1])
    note_name = piano_note[:-1]

    # Find the index of the note name in the piano_notes list and add the corresponding MIDI note number
    note_num = piano_notes.index(note_name)
    midi_note = midi_notes[note_num] + (octave + 1) * 12

    return midi_note

def read_txt(path: str) -> str:
    with open(path, 'r') as file:
        txt = file.read()
    return txt

def get_xml_preset_settings(preset_path: str):
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

def make_json_parameter_mapping(plugin, preset_path:str):
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
                print(f'match found for {key}; closest match: {closest_match}')
                # Extract the value of the JSON key from the JSON string using regex
                match_value = re.search(r'"{}":\s*"([\d.]+)"'.format(key), preset_settings)
                if match_value:
                    param_value = float(match_value.group(1))
                    index = param_name_to_index[closest_match]
                    parameter_mapping[key] = {'match': closest_match, 'value': param_value, 'index': index}
            else:
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
    
    preset_name = preset_path.split(os.sep)[-1].split('.pjunoxl')[0]
    with open(f'TAL-UNO-{preset_name}-parameter-mapping.json', 'w') as outfile:
        json.dump(parameter_mapping, outfile)

def load_xml_preset(dawdreamer_plugin,parameter_mapping_json):
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
    preset_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(preset_ext):
                preset_files.append(os.path.join(root, file))
    return random.choice(preset_files) if preset_files else None

def audio2mel_spectrogram(audio_folder_path, plot_flag=False, window_size=2048, zero_padding_factor=1,
                           window_type='hann', gain_db=0.0, range_db=80.0, high_boost_db=0.0, f_min=0, f_max=20000, n_mels=256):
    """
    Convert a collection of audio files to mel-scaled spectrograms.

    Args:
    audio_folder_path: str
        The path to the folder containing the audio files.
        The audio files should be named in the format of "0.wav, 1.wav, 2.wav, ...".
    plot_flag: bool, default=False
        Whether to plot the mel-scaled spectrograms.
    window_size: int, default=2048
        The size of the FFT window to use.
    zero_padding_factor: int, default=1
        The amount of zero-padding to use in the FFT.
    window_type: str, default='hann'
        The type of window function to use in the FFT.
    gain_db: float, default=0.0
        The gain to apply to the audio signal in decibels.
    range_db: float, default=80.0
        The range of the mel-scaled spectrogram in decibels.
    high_boost_db: float, default=0.0
        The amount of high-frequency boost to apply to the mel-scaled spectrogram in decibels.
    f_min: int, default=0
        The minimum frequency to include in the spectrogram (Hz)
    f_max: int, default=20000
        The maximum frequency to include in the spectrogram (Hz)
    n_mels: int, default=256
        The number of mel frequency bins to include in the spectrogram

    Returns:
    list
        A list of mel-scaled spectrograms, where each element is a NumPy array.
    """

    # Get a list of audio file names in the folder
    audio_file_names = sorted(os.listdir(audio_folder_path))

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
            lbd.specshow(mel_spectrogram, x_axis='time', y_axis='mel',sr=sample_rate, fmin=f_min, fmax=f_max, hop_length=hop_length, cmap='jet', vmin=-range_db, vmax=mel_spectrogram.max() + high_boost_db)
            plt.colorbar(format='%+2.0f dB')
            plt.title('Mel spectrogram for {}'.format(file_name))
            plt.tight_layout()
            plt.show()
        
        mel_spectrograms.append(mel_spectrogram)

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