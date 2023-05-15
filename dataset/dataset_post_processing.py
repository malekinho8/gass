import torch
import pandas as pd
import matplotlib.pyplot as plt
import sys;sys.path.append('./')
# from src.utils import *

# load the preset dataset
df = pd.DataFrame(torch.load('./dataset/preset_dataset_musicnn.pt'))

# Make a copy of the dataframe
df_copy = df.copy()

df = 0 # reduce memory usage

# Verify the structure of the DataFrame
required_columns = ['preset_names', 'parameters', 'parameters_names']
if not all(col in df_copy.columns for col in required_columns):
    raise ValueError(f"DataFrame must contain the following columns: {required_columns}")

try:
    default_preset = df_copy[df_copy['preset_names'] == 'Default'].iloc[0]
except IndexError:
    raise ValueError("'Default' preset not found in DataFrame")

default_parameters = default_preset['parameters']
default_parameter_names = default_preset['parameters_names']
default_mapped_parameter_list = list(default_preset['mapped_parameter_names'])

# Check for non-unique parameter names in the default preset
if len(default_parameter_names) != len(set(default_parameter_names)):
    raise ValueError("Non-unique parameter names found in 'Default' preset")

def add_missing_parameters(row):
    if len(row['parameters']) < len(default_parameter_names) :
        # Update existing parameters
        new_parameters = [row['parameters'][row['parameters_names'].index(name)] if name in row['parameters_names'] else None for name in default_parameter_names]
        
        # Add missing parameters
        missing_parameters = set(default_parameter_names) - set(row['parameters_names'])
        for param_name in missing_parameters:
            new_parameters.append(default_parameters[default_parameter_names.index(param_name)])
            row['parameters_names'].append(param_name)
        
        row['parameters'] = new_parameters
    return row

# get a list containing the lengths of the parameters in each preset
preset_lengths = [len(x) for x in df_copy['parameters']]

# update every row of df_copy in a loop
# Add missing parameters to non-default presets
for i, row in df_copy.iterrows():
    missing_parameters = set(default_parameter_names) - set(row['parameters_names'])
    new_parameters = []
    new_parameter_names = []
    new_mapped_parameter_names = []
    for j, zipped in enumerate(zip(default_parameter_names, default_parameters)):
        param_name, param_value = zipped
        if param_name in row['parameters_names']:
            index = row['parameters_names'].index(param_name)
            new_parameters.append(row['parameters'][index])
            new_parameter_names.append(param_name)
        elif param_name in missing_parameters:
            new_parameters.append(param_value)
            new_parameter_names.append(param_name)
        mapped_name = default_mapped_parameter_list[j]['match']
        mapped_idx = default_mapped_parameter_list[j]['index']
        new_mapped_parameter_names.append({'tal-uno param name': param_name,'dawdreamer param name':mapped_name, 'value': new_parameters[-1], 'dawdreamer index': mapped_idx, 'tal-uno index': j})

    df_copy.at[i, 'parameters'] = new_parameters
    df_copy.at[i, 'parameters_names'] = new_parameter_names
    df_copy.at[i, 'mapped_parameter_names'] = new_mapped_parameter_names

# Display the result
print(df_copy)

# calcualate the length of all the presets in a list
preset_lengths = [len(x) for x in df_copy['parameters']]

# visualize the length of all the presets in a histogram
plt.hist(preset_lengths)
plt.show()

# check if all the presets have the same length
if len(set(preset_lengths)) != 1:
    raise ValueError("Not all presets have the same length")

# save the dataset
torch.save(df_copy, './dataset/processed_preset_dataset_musicnn.pt')

