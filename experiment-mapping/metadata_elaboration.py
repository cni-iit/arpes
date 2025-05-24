import pandas as pd

# Load the CSV file
df = pd.read_csv('fits_metadata.csv')


# Identify offsets in X and Y coordinates and correct
print(f"Identifying offsets in X and Y motor coordinates...")

x_motor_values = pd.to_numeric(df['LMOTOR0'], errors='coerce')
y_motor_values = pd.to_numeric(df['LMOTOR1'], errors='coerce')

x_jumps = []
x_jump_magnitudes = []
y_jumps = []
y_jump_magnitudes = []

for i in range(1, len(x_motor_values)):
    prev_xval = x_motor_values.iloc[i-1]
    curr_xval = x_motor_values.iloc[i]
    prev_yval = y_motor_values.iloc[i-1]
    curr_yval = y_motor_values.iloc[i]
    
    # Check if there is a significant jump back towards 0
    # Conditions:
    # 1. Current value is close to zero
    # 2. Current value is negligible compared to the previous one
    if (abs(curr_xval) < 2 and 
        abs(curr_xval) < abs(prev_xval) * 0.01):
        
        x_jumps.append(i)
        x_jump_magnitudes.append(prev_xval-curr_xval)
        y_jumps.append(i)
        y_jump_magnitudes.append(prev_yval-curr_yval)
        print(f"Jump detected at index {i} in X: {prev_xval:.2f} -> {curr_xval:.2f}")
    
    if (abs(curr_yval) < 5 and 
        abs(curr_yval) < abs(prev_yval) * 0.01):
        
        y_jumps.append(i)
        y_jump_magnitudes.append(prev_yval-curr_yval)
        x_jumps.append(i)
        x_jump_magnitudes.append(prev_xval-curr_xval)
        print(f"Jump detected at index {i} in Y: {prev_yval:.2f} -> {curr_yval:.2f}")

# Prepare for correction of X coordinates
cumulative_correction = 0

for jump_idx, jump_mag in zip(x_jumps, x_jump_magnitudes):
    cumulative_correction += jump_mag
    print(f"Applying correction in X of {cumulative_correction:.2f} from index {jump_idx} onward")
    
    # Apply correction from this point forward
    df.loc[jump_idx:, 'LMOTOR0'] += cumulative_correction
    df.loc[jump_idx:, 'PMOTOR0'] += cumulative_correction

# Prepare for correction of Y coordinates
cumulative_correction = 0

for jump_idx, jump_mag in zip(y_jumps, y_jump_magnitudes):
    cumulative_correction += jump_mag
    print(f"Applying correction in Y of {cumulative_correction:.2f} from index {jump_idx} onward")
    
    # Apply correction from this point forward
    df.loc[jump_idx:, 'LMOTOR1'] += cumulative_correction
    df.loc[jump_idx:, 'PMOTOR1'] += cumulative_correction


# Center the physical stage coordinates around 0
df['PMOTOR0'] = df['PMOTOR0'] - df['PMOTOR0'].median()
df['PMOTOR1'] = df['PMOTOR1'] - df['PMOTOR1'].median()

# Filter rows based on the 'LWLVNM' column
coarse_scans = df[df['LWLVNM'].isin(['XY Scan Coarse'])]
fine_scans = df[df['LWLVNM'].isin(['XY Scan Fine'])]
single_spot_measurements = df[~df['LWLVNM'].isin(['XY Scan Fine', 'XY Scan Coarse', 'Focus Scan Fine'])]

# Prepare data for [COARSE SCANS] section
coarse_scans_list = []
for _, row in coarse_scans.iterrows():
    x_stage = round(row['PMOTOR0'] + row['PMOTOR10'], 3)  # Physical stage x + capillary correction (fixed offset during scan)
    y_stage = round(row['PMOTOR1'] + row['PMOTOR11'], 3)  # Physical stage y + capillary correction (fixed offset during scan)
    x_min = round(row['ST_0_0'] - row['LMOTOR0'])  # Logical stage starting x - logical stage x
    x_max = round(row['EN_0_0'] - row['LMOTOR0'])  # Logical stage ending x - logical stage x
    y_min = round(row['ST_0_1'] - row['LMOTOR1'])  # Logical stage starting y - logical stage y
    y_max = round(row['EN_0_1'] - row['LMOTOR1'])  # Logical stage ending y - logical stage y
    label = row['FILENAME'].split('_', 1)[-1].rsplit('.', 1)[0].lstrip('0')
    coarse_scans_list.append(f"{x_stage}, {y_stage}, ({x_min}, {x_max}), ({y_min}, {y_max}), '{label}'")

# Prepare data for [FINE SCANS] section
fine_scans_list = []
for _, row in fine_scans.iterrows():
    x_stage = round(row['PMOTOR0'], 3)  # Physical stage x (no need for capillary correction)
    y_stage = round(row['PMOTOR1'], 3)  # Physical stage y (no need for capillary correction)
    x_min = row['ST_0_0']  # Capillary starting x
    x_max = row['EN_0_0']  # Capillary ending x
    y_min = row['ST_0_1']  # Capillary starting y
    y_max = row['EN_0_1']  # Capillary ending y
    label = row['FILENAME'].split('_', 1)[-1].rsplit('.', 1)[0].lstrip('0')
    fine_scans_list.append(f"{x_stage}, {y_stage}, ({x_min}, {x_max}), ({y_min}, {y_max}), '{label}'")

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
    f.write("# Format: x_stage, y_stage, (x_min, x_max), (y_min, y_max), label\n")
    for scan in coarse_scans_list:
        f.write(scan + "\n")
    f.write("\n[FINE SCANS]\n")
    f.write("# Format: x_stage, y_stage, (x_min, x_max), (y_min, y_max), label\n")
    for scan in fine_scans_list:
        f.write(scan + "\n")
    f.write("\n[SINGLE-SPOT MEASUREMENTS]\n")
    f.write("# Format: x_stage, y_stage, x_scan, y_scan, label\n")
    for measurement in single_spot_measurements_list:
        f.write(measurement + "\n")

print("The file 'MeasureCoords.txt' has been created successfully.")