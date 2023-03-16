import os, re
import numpy as np
import pandas as pd

'''
With the following code, the label mapping is performed. Modeling will be performed with
SNOMED CT Codes so all other different codes needs to be converted into them. The script updates
the original dataframe with a new column, e.g. 'SNOMEDCTCode'.

Note that if another csv file is stored, make sure to keep only one csv file in the data directory
so that creating csv files of data splits knows where to load the codes from. I.e., delete or move 
the ones that are not needed.

Note also that the columns should be similarly named in metadata file and mapping file.
'''

# ---- CSV FILE OF THE METADATA
csv_path = os.path.join(os.getcwd(), 'data', 'smoke_data', 'Shandong', 'metadata.csv')
csv_file = os.path.join(csv_path)

# ---- CSV FILE OF LABEL MAPPING
map_path = os.path.join(os.getcwd(), 'data', 'AHA_SNOMED_mapping.csv')
csv_map = os.path.join(map_path)

# --- FROM WHICH AND TO WHICH TO CONVERT
#     corresponds to the columns from where the codes are read from metadata and mapping file
from_code = 'AHA_Code'
to_code = 'SNOMEDCTCode'

# -------------------------------------------------------------------------------

# Read the csv file containing metadata
metadata = pd.read_csv(csv_file)

# Create an empty column to which to gather the SNOMED CT Codes
metadata[to_code] = np.nan

# Get label mapping
label_map = pd.read_csv(csv_map, sep=';').fillna(0)
label_map = label_map[(label_map[from_code] != 0) & (label_map[to_code] != 0)]
label_map = label_map[label_map[to_code].notna()]
label_map[[from_code, to_code]] = label_map[[from_code, to_code]].astype(int)

# Iterate over metadata file and add a SNOMED CT Code if possible
for i, aha in enumerate(metadata[from_code]):
    codes = re.findall('\d+', aha) # might be one or multiple
    codes = list(map(int, codes))
    
    # Check if any code in metadata found from the mapping csv file
    # If yes, find corresponding SNOMED CT Codes
    if any(c for c in codes if int(c) in label_map[from_code].values):
        found_codes = [c for c in codes if int(c) in label_map[from_code].values]

        # Gather codes
        snomed = []
        for dx in found_codes:
            snomed.append(label_map.loc[label_map[from_code] == dx, to_code].values[0])
        
        # If codes, convert into a string of codes and store using row index
        if snomed:
            metadata.loc[i, to_code] = ','.join(list(map(str, set(snomed))))

# Save the updated csv file
metadata.to_csv(csv_path, index=False)
