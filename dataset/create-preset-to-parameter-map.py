import torch
import sys; sys.path.append('./')
from src.config import *


# load the processed preset dataset file
df = torch.load('./dataset/processed_preset_dataset_musicnn.pt')

# keep only the columns named 'preset_names' and 'parameters'
df = df[['preset_names', 'parameters']]

# make a list of the lengths of the parameters in each preset
preset_lengths = [len(x) for x in df['parameters']]

# ensure that all of the presets have exactly NUM_TAL_UNO_PARAMETERS parameters
if not all(x == NUM_TAL_UNO_PARAMETERS for x in preset_lengths):
    raise ValueError(f"Dataset contains presets with {NUM_TAL_UNO_PARAMETERS} parameters")

# save the dataset
torch.save(df, './dataset/tal-uno-preset-to-parameter-mapping.pt')
