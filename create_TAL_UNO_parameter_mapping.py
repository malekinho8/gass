import dawdreamer as daw
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from src.utils import *
import re
import difflib
import json

# define constants
SAMPLE_RATE = 44100
BUFFER_SIZE = 128 # Parameters will undergo automation at this buffer/block size.
PPQN = 960 # Pulses per quarter note.
SYNTH_PLUGIN = "/Library/Audio/Plug-Ins/VST3/TAL-U-NO-LX-V2.vst3"  # extensions: .dll, .vst3, .vst, .component
SYNTH_NAME = "TAL-Uno"
PRESET_FOLDER = "/Users/malek8/Library/Application Support/ToguAudioLine/TAL-U-No-LX/presets"
PRESET_EXT = ".pjunoxl"

# create a RenderEngine object
engine = daw.RenderEngine(sample_rate=SAMPLE_RATE, block_size=BUFFER_SIZE) # what does block_size do?

# create the plugin object
plugin = engine.make_plugin_processor(SYNTH_NAME, SYNTH_PLUGIN)
assert plugin.get_name() == SYNTH_NAME

# randomly select a preset from the preset folder
preset_path = select_preset_path(PRESET_FOLDER,PRESET_EXT)

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

with open('TAL-UNO-parameter-mapping.json', 'w') as outfile:
    json.dump(parameter_mapping, outfile)

