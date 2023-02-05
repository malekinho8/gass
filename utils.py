import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
import matplotlib.pyplot as plt

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