import pandas as pd
import numpy as np

# Load the CSV file
df = pd.read_csv('fits_metadata.csv')


# Set the first dataset's phi value to zero, shift the rest accordingly and convert to radians
df['PMOTOR4'] = (df['PMOTOR4'] - df['PMOTOR4'][0])*np.pi/180.

# Identify rotations (phi coordinate) to correct X and Y
phi_motor_values = pd.to_numeric(df['PMOTOR4'], errors='coerce')
x_motor_values = pd.to_numeric(df['PMOTOR0'], errors='coerce')
y_motor_values = pd.to_numeric(df['PMOTOR1'], errors='coerce')

for i in range(1, len(phi_motor_values)):
    phi_var = phi_motor_values.iloc[i] - phi_motor_values.iloc[i-1]
    
    # Check if there is a significant change in phi
    if abs(phi_var)*180/np.pi>1:
        
        # Offset all stage X and Y values from now on to account for stage adjustment
        df.loc[i:,'PMOTOR0'] += x_motor_values.iloc[i-1] - x_motor_values.iloc[i]
        df.loc[i:,'PMOTOR1'] += y_motor_values.iloc[i-1] - y_motor_values.iloc[i]
        
        # Apply reference frame rotation to stage X and Y coordinates from next dataset on
        df_old = df.copy()
        df.loc[i+1:,'PMOTOR0'] += (np.cos(phi_var)-1) * (df_old.loc[i+1:,'PMOTOR0']-df.loc[i,'PMOTOR0']) -  np.sin(phi_var)    * (df_old.loc[i+1:,'PMOTOR1']-df.loc[i,'PMOTOR1']) # type: ignore
        df.loc[i+1:,'PMOTOR1'] +=  np.sin(phi_var)    * (df_old.loc[i+1:,'PMOTOR0']-df.loc[i,'PMOTOR0']) + (np.cos(phi_var)-1) * (df_old.loc[i+1:,'PMOTOR1']-df.loc[i,'PMOTOR1']) # type: ignore
        
        # Apply reference frame rotation to capillary X and Y coordinates from now on
        df.loc[i:,'LMOTOR10'] = np.cos(phi_var) * df_old.loc[i:,'LMOTOR10'] - np.sin(phi_var) * df_old.loc[i:,'LMOTOR11']
        df.loc[i:,'LMOTOR11'] = np.sin(phi_var) * df_old.loc[i:,'LMOTOR10'] + np.cos(phi_var) * df_old.loc[i:,'LMOTOR11']


# Center the physical stage coordinates around the first dataset
df['PMOTOR0'] = df['PMOTOR0'] - df['PMOTOR0'][0]
df['PMOTOR1'] = df['PMOTOR1'] - df['PMOTOR1'][0]


# Filter rows based on the 'LWLVNM' column
coarse_scans = df[df['LWLVNM'].isin(['XY Scan Coarse'])]
fine_scans = df[df['LWLVNM'].isin(['XY Scan Fine'])]
single_spot_measurements = df[~df['LWLVNM'].isin(['XY Scan Fine', 'XY Scan Coarse', 'Focus Scan Fine'])]

# Prepare data for [COARSE SCANS] section
coarse_scans_list = []
for _, row in coarse_scans.iterrows():
    x_stage = round(row['PMOTOR0'] + row['LMOTOR10'], 3)  # Physical stage x + capillary correction (fixed offset during scan)
    y_stage = round(row['PMOTOR1'] + row['LMOTOR11'], 3)  # Physical stage y + capillary correction (fixed offset during scan)
    x_min = round(row['ST_0_0'] - row['LMOTOR0'], 1)  # Logical stage starting x - logical stage x
    x_max = round(row['EN_0_0'] - row['LMOTOR0'], 1)  # Logical stage ending x - logical stage x
    y_min = round(row['ST_0_1'] - row['LMOTOR1'], 1)  # Logical stage starting y - logical stage y
    y_max = round(row['EN_0_1'] - row['LMOTOR1'], 1)  # Logical stage ending y - logical stage y
    phi = round(row['PMOTOR4'], 1)  # Stage phi
    label = row['FILENAME'].split('_', 1)[-1].rsplit('.', 1)[0].lstrip('0')
    coarse_scans_list.append(f"{x_stage}, {y_stage}, ({x_min}, {x_max}), ({y_min}, {y_max}), {phi}, '{label}'")

# Prepare data for [FINE SCANS] section
fine_scans_list = []
for _, row in fine_scans.iterrows():
    x_stage = round(row['PMOTOR0'], 3)  # Physical stage x (no need for capillary correction)
    y_stage = round(row['PMOTOR1'], 3)  # Physical stage y (no need for capillary correction)
    x_min = row['ST_0_0']  # Capillary starting x
    x_max = row['EN_0_0']  # Capillary ending x
    y_min = row['ST_0_1']  # Capillary starting y
    y_max = row['EN_0_1']  # Capillary ending y
    phi = round(row['PMOTOR4'], 1)  # Stage phi
    label = row['FILENAME'].split('_', 1)[-1].rsplit('.', 1)[0].lstrip('0')
    fine_scans_list.append(f"{x_stage}, {y_stage}, ({x_min}, {x_max}), ({y_min}, {y_max}), {phi}, '{label}'")

# Prepare data for [SINGLE-SPOT MEASUREMENTS] section
single_spot_measurements_list = []
for _, row in single_spot_measurements.iterrows():
    x_stage = round(row['PMOTOR0'], 3)  # Physical stage x
    y_stage = round(row['PMOTOR1'], 3)  # Physical stage y
    x_scan = round(row['LMOTOR10'], 3)  # Capillary x
    y_scan = round(row['LMOTOR11'], 3)  # Capillary y
    label = row['FILENAME'].split('_', 1)[-1].rsplit('.', 1)[0].lstrip('0')
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