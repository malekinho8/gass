import dawdreamer as daw
import numpy as np
from scipy.io.wavfile import write
from scipy.signal import savgol_filter
import os
import sys; sys.path.append(f'{os.sep}'.join(sys.path[0].split(os.sep)[0:-1]))
import librosa as lb
import torch
from musicnn.extractor import extractor
from src.utils import *

# define constants
SAMPLE_RATE = 22050
BUFFER_SIZE = 128 # Parameters will undergo automation at this buffer/block size.
PPQN = 960 # Pulses per quarter note.
SYNTH_PLUGIN = "/Library/Audio/Plug-Ins/VST3/TAL-U-NO-LX-V2.vst3"  # extensions: .dll, .vst3, .vst, .component
SYNTH_NAME = "TAL-Uno"
PRESET_FOLDER = "/Users/malek8/Library/Application Support/ToguAudioLine/TAL-U-No-LX/presets"
PRESET_EXT = ".pjunoxl"
DURATION = 5  # in seconds
NOTES = [f'{note}{i}' for i in range(1, 6) for note in ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']]
EXP_NAME = '04.03.2023-timbral-comparison-test'
SMOOTHING_WINDOW_LENGTH = 11  # Must be an odd integer
POLYORDER = 3

# create a RenderEngine object
engine = daw.RenderEngine(sample_rate=SAMPLE_RATE, block_size=BUFFER_SIZE) # what does block_size do?

# create the plugin object
plugin = engine.make_plugin_processor(SYNTH_NAME, SYNTH_PLUGIN)
assert plugin.get_name() == SYNTH_NAME

# randomly select a preset from the preset folder
preset_path = select_preset_path(PRESET_FOLDER,PRESET_EXT)

# create JSON parameter map and save it
make_json_parameter_mapping(plugin,preset_path)

# create JSON parameter map and save it
json_file_name = make_json_parameter_mapping(plugin,preset_path,verbose=False)

# apply the synth preset settings to the synth plugin processor object
loaded_preset_synth = load_xml_preset(plugin,json_file_name)

# intialize the outputs
mfcc_features = []
musicnn_features = []

for NOTE in NOTES:
    # convert the piano note to midi (0 to 127)
    midi_piano_note = piano_note_to_midi_note(NOTE)

    # generate a sound using the plugin
    loaded_preset_synth.add_midi_note(midi_piano_note, 60, 0.0, 3)

    engine.load_graph([(loaded_preset_synth, [])])

    # loaded_preset_synth.open_editor()
    engine.render(DURATION)

    audio = engine.get_audio()

    # make the diectory if it does not exist
    if not os.path.exists('output'):
        os.mkdir('output')

    # write the output to a WAV file
    out_name = json_file_name.split(os.sep)[-1].split('.json')[0].split('-parameter-mapping')[0]
    file_name = f'output{os.sep}{out_name}-d{DURATION}-{NOTE}.wav'
    write(file_name, SAMPLE_RATE, audio[0,:])

    # obtain timbral features from sound using MFCC
    mfcc = lb.feature.mfcc(y=audio[0,:],sr=SAMPLE_RATE)

    # obtain timbral features from sound using pre-trained neural network (pons 2018)
    taggram, tags, features = extractor(file_name, model='MTT_musicnn', extract_features=True)

    # Append the mean MFCC for each note
    mfcc_features.append(mfcc)

    # Append the mean musicnn feature for each note
    musicnn_features.append(features['penultimate'][0,:])

# stack the lists into an np array
mfcc_features_stacked = np.stack(mfcc_features, axis=0)
musicnn_features_stacked = np.stack(musicnn_features, axis=0)

# Normalize the MFCC and Musicnn feature datasets for fair comparison
mfcc_features_normalized = normalize_data(mfcc_features_stacked)
musicnn_features_normalized = normalize_data(musicnn_features_stacked)

# Compute the standard deviation of the elements in each list
mfcc_std = np.std(mfcc_features_normalized, axis=0, keepdims=True)[0]
musicnn_std = np.std(musicnn_features_normalized, axis=0, keepdims=True)[0, :]

# see which one has lower overall standard deviation
mfcc_total_diff = np.mean(mfcc_std)
musicnn_total_diff = np.mean(musicnn_std)

# plot the standard deviation surface/line
fig, axes = plt.subplots(2,1,figsize=(10,8))

for line, note in zip(musicnn_features, NOTES):
    smoothed_line = savgol_filter(line, SMOOTHING_WINDOW_LENGTH, POLYORDER)
    axes[0].plot(smoothed_line, alpha=0.2)
smoothed_musicnn_std = savgol_filter(musicnn_std, SMOOTHING_WINDOW_LENGTH, POLYORDER)
axes[0].plot(smoothed_musicnn_std, linewidth=2, label='Std. Deviation')
axes[0].legend()
axes[0].set_title('Musicnn Standard Deviation Across Notes (Smoothed)')
axes[0].set_ylim([0,1])
# Plot MFCC standard deviation surface
axes[1].imshow(mfcc_std[:,:],vmin=0,vmax=1)
axes[1].set_title('MFCC Standard Deviation Surface Across Notes')

# Add a colorbar for the MFCC plot
cbar = fig.colorbar(axes[1].get_images()[0], ax=axes[1])
cbar.set_label('Standard Deviation')

# Show the plot
plt.show(block=False)

# save experimental results
results = {
    'mfcc_total_diff':mfcc_total_diff,
    'musicnn_total_diff':musicnn_total_diff,
}
std_results ={
    'mfcc_std_surface':mfcc_std,
    'musicnn_std_line':musicnn_std,
    'mfcc_feature_vectors':mfcc_features,
    'musicnn_feature_vectors':musicnn_features,
}
save_name = os.path.join('exp','results',EXP_NAME + '.pt')
save_name2 = os.path.join('exp','results',EXP_NAME + '-feature-data' + '.pt')
torch.save(results,save_name)
torch.save(std_results,save_name2)

