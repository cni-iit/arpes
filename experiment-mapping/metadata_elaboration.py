import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv('fits_metadata.csv')


# Set the first dataset's phi value to zero, shift the rest accordingly and convert to radians
df['Stage_Phi'] = np.radians((df['Stage_Phi'] - df['Stage_Phi'][0]))

# Identify rotations (phi coordinate) to correct X and Y
phi_motor_values = pd.to_numeric(df['Stage_Phi'], errors='coerce')
x_motor_values = pd.to_numeric(df['Stage_X'], errors='coerce')
y_motor_values = pd.to_numeric(df['Stage_Y'], errors='coerce')

for i in range(1, len(phi_motor_values)):
    phi_var = phi_motor_values.iloc[i] - phi_motor_values.iloc[i-1]
    
    # Check if there is a significant change in phi
    if np.degrees(abs(phi_var)):
        
        # Offset all stage X and Y values from now on to account for stage adjustment
        df.loc[i:,'Stage_X'] += x_motor_values.iloc[i-1] - x_motor_values.iloc[i]
        df.loc[i:,'Stage_Y'] += y_motor_values.iloc[i-1] - y_motor_values.iloc[i]
        
        # Apply reference frame rotation to stage X and Y coordinates from next dataset on
        df_old = df.copy()
        df.loc[i+1:,'Stage_X'] += (np.cos(phi_var)-1) * (df_old.loc[i+1:,'Stage_X']-df.loc[i,'Stage_X']) -  np.sin(phi_var)    * (df_old.loc[i+1:,'Stage_Y']-df.loc[i,'Stage_Y']) # type: ignore
        df.loc[i+1:,'Stage_Y'] +=  np.sin(phi_var)    * (df_old.loc[i+1:,'Stage_X']-df.loc[i,'Stage_X']) + (np.cos(phi_var)-1) * (df_old.loc[i+1:,'Stage_Y']-df.loc[i,'Stage_Y']) # type: ignore
        
        # Apply reference frame rotation to capillary X and Y coordinates from now on
        df.loc[i:,'Capillary_X'] = np.cos(phi_var) * df_old.loc[i:,'Capillary_X'] - np.sin(phi_var) * df_old.loc[i:,'Capillary_Y']
        df.loc[i:,'Capillary_Y'] = np.sin(phi_var) * df_old.loc[i:,'Capillary_X'] + np.cos(phi_var) * df_old.loc[i:,'Capillary_Y']


# Center the physical stage coordinates around the first dataset
df['Stage_X'] = df['Stage_X'] - df['Stage_X'][0]
df['Stage_Y'] = df['Stage_Y'] - df['Stage_Y'][0]


# Filter rows based on the 'Measurement_type' column
coarse_scans = df[df['Measurement_type'].isin(['XY Scan Coarse'])]
fine_scans = df[df['Measurement_type'].isin(['XY Scan Fine'])]
single_spot_measurements = df[~df['Measurement_type'].isin(['XY Scan Fine', 'XY Scan Coarse', 'Focus Scan Fine'])]

# Prepare data for [COARSE SCANS] section
coarse_scans_list = []
for _, dataset in coarse_scans.iterrows():
    x_stage = round(dataset['Stage_X'] + dataset['Capillary_X'], 3)  # Physical stage x + capillary correction (fixed offset during scan)
    y_stage = round(dataset['Stage_Y'] + dataset['Capillary_Y'], 3)  # Physical stage y + capillary correction (fixed offset during scan)
    x_min = round(dataset['Scan_start_x'] - dataset['Logical_stage_X'], 1)  # Logical stage starting x - logical stage x
    x_max = round(dataset['Scan_end_x'] - dataset['Logical_stage_X'], 1)  # Logical stage ending x - logical stage x
    y_min = round(dataset['Scan_start_y'] - dataset['Logical_stage_Y'], 1)  # Logical stage starting y - logical stage y
    y_max = round(dataset['Scan_end_y'] - dataset['Logical_stage_Y'], 1)  # Logical stage ending y - logical stage y
    phi = round(dataset['Stage_Phi'], 1)  # Stage phi
    label = dataset['Filename'].split('_', 1)[-1].rsplit('.', 1)[0].lstrip('0')
    coarse_scans_list.append(f"{x_stage}, {y_stage}, ({x_min}, {x_max}), ({y_min}, {y_max}), {phi}, '{label}'")

# Prepare data for [FINE SCANS] section
fine_scans_list = []
for _, dataset in fine_scans.iterrows():
    x_stage = round(dataset['Stage_X'], 3)  # Physical stage x (no need for capillary correction)
    y_stage = round(dataset['Stage_Y'], 3)  # Physical stage y (no need for capillary correction)
    x_min = dataset['Scan_start_x']  # Capillary starting x
    x_max = dataset['Scan_end_x']  # Capillary ending x
    y_min = dataset['Scan_start_y']  # Capillary starting y
    y_max = dataset['Scan_end_y']  # Capillary ending y
    phi = round(dataset['Stage_Phi'], 1)  # Stage phi
    label = dataset['Filename'].split('_', 1)[-1].rsplit('.', 1)[0].lstrip('0')
    fine_scans_list.append(f"{x_stage}, {y_stage}, ({x_min}, {x_max}), ({y_min}, {y_max}), {phi}, '{label}'")

# Prepare data for [SINGLE-SPOT MEASUREMENTS] section
single_spot_measurements_list = []
for _, dataset in single_spot_measurements.iterrows():
    x_stage = round(dataset['Stage_X'], 3)  # Physical stage x
    y_stage = round(dataset['Stage_Y'], 3)  # Physical stage y
    x_scan = round(dataset['Capillary_X'], 3)  # Capillary x
    y_scan = round(dataset['Capillary_Y'], 3)  # Capillary y
    label = dataset['Filename'].split('_', 1)[-1].rsplit('.', 1)[0].lstrip('0')
    single_spot_measurements_list.append(f"{x_stage}, {y_stage}, {x_scan}, {y_scan}, '{label}'")

# Write the results to 'MeasureCoords.txt'
with open('MeasureCoords.txt', 'w') as f:
    f.write("[COARSE SCANS]\n")
    f.write("# Format: x_stage, y_stage, (x_min, x_max), (y_min, y_max), phi, label\n")
    for scan in coarse_scans_list:
        f.write(scan + "\n")
    f.write("\n[FINE SCANS]\n")
    f.write("# Format: x_stage, y_stage, (x_min, x_max), (y_min, y_max), phi, label\n")
    for scan in fine_scans_list:
        f.write(scan + "\n")
    f.write("\n[SINGLE-SPOT MEASUREMENTS]\n")
    f.write("# Format: x_stage, y_stage, x_scan, y_scan, label\n")
    for measurement in single_spot_measurements_list:
        f.write(measurement + "\n")

print("The file 'MeasureCoords.txt' has been created successfully.")