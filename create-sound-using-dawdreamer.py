import dawdreamer as daw
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from utils import audio2_mel_spectrogram

# create a RenderEngine object
fs = 44100
test = np.array([1,2,3,4,5])
engine = daw.RenderEngine(sample_rate=fs, block_size=128) # what does block_size do?

# create the plugin object
plugin_path = '/Library/Audio/Plug-Ins/VST3/TAL-U-NO-LX-V2.vst3'
plugin = engine.make_plugin_processor("test", plugin_path)

# generate a sound using the plugin
duration = 5  # in seconds
frequency = 440  # in Hz
samples = np.sin(2 * np.pi * frequency * np.arange(duration * fs))  # generate a sine wave
plugin.set_parameter(0, 1)  # set a parameter value for the plugin
plugin.add_midi_note(60, 60, 0.0, .25)
plugin.add_midi_note(64, 80, 0.5, .5)
plugin.add_midi_note(67, 127, 0.75, .5)

assert(plugin.n_midi_events == 3*2)  # multiply by 2 because of the off-notes.
engine.load_graph([(plugin, [])])

engine.render(duration)

audio = engine.get_audio()

# make the diectory if it does not exist
os.mkdir('test')

# write the output to a WAV file
write('test/0.wav', fs, audio[0,:])

# get the spectrogram
spec = audio2_mel_spectrogram(audio_folder_path='test',plot_flag=True,zero_padding_factor=1,range_db=80,gain_db=20)