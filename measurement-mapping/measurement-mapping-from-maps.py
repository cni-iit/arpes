import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import re
import ast

def read_measure_coords(file_path):
    """
    Read points and rectangles from MeasureCoords.txt file.
    
    The file format should be:
    
    [POINTS]
    x, y, label
    x, y, label
    ...
    
    [RECTANGLES]
    x_stage, y_stage, (x_min, x_max), (y_min, y_max), label
    x_stage, y_stage, (x_min, x_max), (y_min, y_max), label
    ...
    
    Returns:
        tuple: (points, rectangles) where:
            - points is a list of tuples (x, y, label)
            - rectangles is a list of tuples (x_stage, y_stage, x_scan_exten, y_scan_exten, label)
    """
    try:
        point_measurements = []
        fine_scans = []
        
        current_section = None
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines
                if not line or line.startswith('#'):
                    continue
                
                # Check for section headers
                if line.upper() == '[POINT MEASUREMENTS]':
                    current_section = 'point measurements'
                    continue
                elif line.upper() == '[FINE SCANS]':
                    current_section = 'fine scans'
                    continue
                
                # Process data based on current section
                if current_section == 'point measurements':
                    try:
                        # Split the line by comma and strip whitespace
                        parts = [p.strip() for p in line.split(',', 4)]
                        if len(parts) == 5:
                            x = float(parts[0]) + float(parts[2])  # Combine x_stage and x_scan
                            y = float(parts[1]) + float(parts[3])  # Combine y_stage and y_scan
                            label = parts[4].strip("'\"")  # Remove quotes if present
                            point_measurements.append((x, y, label))
                    except Exception as e:
                        print(f"! Error parsing point: {line}. Error: {str(e)}")
                
                elif current_section == 'fine scans':
                    try:
                        # Use regex to find all parts
                        matches = re.match(r'(.*?),\s*(.*?),\s*(\(.*?\)),\s*(\(.*?\)),\s*(.*?)$', line)
                        if matches:
                            x_stage = float(eval(matches.group(1)))
                            y_stage = float(eval(matches.group(2)))
                            x_scan_exten = ast.literal_eval(matches.group(3))
                            y_scan_exten = ast.literal_eval(matches.group(4))
                            label = matches.group(5).strip("'\"")
                            fine_scans.append((x_stage, y_stage, x_scan_exten, y_scan_exten, label))
                    except Exception as e:
                        print(f"! Error parsing rectangle: {line}. Error: {str(e)}")
        
        return point_measurements, fine_scans
    
    except Exception as e:
        print(f"! Error reading measure coordinates file: {str(e)}")
        return [], []

def plot_graph_from_file(file_path, point_measurements, fine_scans, save_path=None):
    """
    Plot intensity map from a txt file and add point measurements and fine scans based on specified rules.
    
    Args:
        file_path: Path to the txt file containing intensity data
        point_measurements: List of point measurements as tuples (x, y, label)
        fine_scans: List of fine scans as tuples (x_stage, y_stage, x_scan_exten, y_scan_exten, label)
        save_path: If provided, save the plot to this path instead of displaying it
    """
    try:
        # Load data from file, handling empty cells and non-numeric data
        data = np.genfromtxt(file_path, delimiter='\t', filling_values=np.nan)
        
        # Extract x and y values
        x_values = data[1:, 0]
        y_values = data[0, 1:]
        
        # Extract intensity values
        intensity_values = data[1:, 1:].T
        
        # Get data limits
        x_min, x_max = np.min(x_values), np.max(x_values)
        y_min, y_max = np.min(y_values), np.max(y_values)
        
        # Create a new figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot intensity values as faded background
        c = ax.pcolormesh(x_values, y_values, intensity_values, shading='auto', cmap='Greys_r', alpha=0.8)
        # fig.colorbar(c, ax=ax)
        
        # Ensure units on both axes have the same length in pixels
        ax.set_aspect('equal', adjustable='box')
        
        # Filter and plot point measurements that are within the data limits
        valid_points = []
        for point in point_measurements:
            x, y, label = point
            
            if x_min <= x <= x_max and y_min <= y <= y_max:
                valid_points.append((x, y, label))
                ax.plot(x, y, 'o', label=label)
        
        # Filter and plot fine scans with at least two vertices inside limits
        for rect in fine_scans:
            x_stage, y_stage, x_scan_exten, y_scan_exten, label = rect
            
            # Calculate rectangle corners
            x_left = x_stage + x_scan_exten[0]
            y_bottom = y_stage + y_scan_exten[0]
            x_right = x_stage + x_scan_exten[1]
            y_top = y_stage + y_scan_exten[1]
            
            # Check if at least two vertices are within the data limits
            vertices = [
                (x_left, y_bottom),  # Bottom-left
                (x_right, y_bottom), # Bottom-right
                (x_right, y_top),    # Top-right
                (x_left, y_top)      # Top-left
            ]
            
            vertices_in_bounds = sum(1 for vx, vy in vertices if 
                                     x_min <= vx <= x_max and y_min <= vy <= y_max)
            
            if vertices_in_bounds >= 2:
                # Plot the rectangle
                rect_patch = plt.Rectangle((x_left, y_bottom), 
                                          x_right - x_left, 
                                          y_top - y_bottom, 
                                          linestyle='--', 
                                          edgecolor='black', 
                                          facecolor='none')
                ax.add_patch(rect_patch)
                # Add label
                ax.text(x_left, y_bottom, f' {label}', 
                        verticalalignment='bottom', horizontalalignment='left')
        
        # Add legend if there are valid points
        if valid_points:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Set title with the filename
        plt.title(f"Map from file \"{os.path.basename(file_path)}\"")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
        return True
    except Exception as e:
        print(f"! Error processing {file_path}: {str(e)}")
        return False

def create_overview_plot(point_measurements, fine_scans, save_path=None):
    """
    Create an overview plot with all point measurements and fine scans
    when no intensity map files are available.
    
    Args:
        point_measurements: List of point measurements
        fine_scans: List of fine scans
        save_path: Optional path to save the plot
    """
    # Calculate the overall boundaries for the plot
    all_x_coords = []
    all_y_coords = []
    
    # Add points
    for point in point_measurements:
        x, y, _ = point
        all_x_coords.append(x)
        all_y_coords.append(y)
    
    # Add rectangle corners
    for rect in fine_scans:
        x_stage, y_stage, x_scan_exten, y_scan_exten, _ = rect
        x_left = x_stage + x_scan_exten[0]
        y_bottom = y_stage + y_scan_exten[0]
        x_right = x_stage + x_scan_exten[1]
        y_top = y_stage + y_scan_exten[1]
        
        all_x_coords.extend([x_left, x_right])
        all_y_coords.extend([y_bottom, y_top])
    
    if not all_x_coords or not all_y_coords:
        print("! No points or rectangles to display.\n")
        return False
    
    # Calculate bounds with a margin
    margin_factor = 0.1  # 10% margin
    x_min, x_max = min(all_x_coords), max(all_x_coords)
    y_min, y_max = min(all_y_coords), max(all_y_coords)
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    x_min -= x_range * margin_factor
    x_max += x_range * margin_factor
    y_min -= y_range * margin_factor
    y_max += y_range * margin_factor
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot all point measurements
    for point in point_measurements:
        x, y, label = point
        ax.plot(x, y, 'o', label=label)
    
    # Plot all fine scans
    for rect in fine_scans:
        x_stage, y_stage, x_scan_exten, y_scan_exten, label = rect
        
        # Calculate rectangle corners
        x_left = x_stage + x_scan_exten[0]
        y_bottom = y_stage + y_scan_exten[0]
        width = x_scan_exten[1] - x_scan_exten[0]
        height = y_scan_exten[1] - y_scan_exten[0]
        
        # Plot the rectangle
        rect_patch = plt.Rectangle((x_left, y_bottom), width, height,
                                  linestyle='--', edgecolor='black', facecolor='none')
        ax.add_patch(rect_patch)
        # Add label
        ax.text(x_left, y_bottom, f' {label}',
                verticalalignment='bottom', horizontalalignment='left')
    
    # Set the axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    # Ensure units on both axes have the same length in pixels
    ax.set_aspect('equal', adjustable='box')
    
    # Add legend for point measurements
    if point_measurements:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Set title
    plt.title("Overview of All Point Measurements and Fine Scans")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
        print(f"Overview plot saved to: {save_path}\n")
    else:
        plt.show()
    
    return True

def process_all_txt_files(coords_file_name="MeasureCoords.txt", output_dir=None):
    """
    Process all txt files in the script's directory and generate plots.
    
    Args:
        coords_file_name: Name of the file containing point measurements and fine scans coordinates
        output_dir: Directory to save output plots (optional)
    """
    # Print an empty line to begin with
    print()
    
    # Get the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Path to the coordinates file
    coords_file_path = os.path.join(script_dir, coords_file_name)
    
    # Check if coordinates file exists
    if not os.path.isfile(coords_file_path):
        print(f"Coordinates file '{coords_file_name}' not found in {script_dir}")
        print(f" -> Creating a template coordinates file at {coords_file_path}")
        
        # Create a template file
        with open(coords_file_path, 'w') as f:
            f.write("[POINT MEASUREMENTS]\n")
            f.write("# Format: x_stage, y_stage, x_scan, y_scan, label\n")
            f.write("418.6, 4280.7, 20.5, -6.8, 'f01744'\n")
            f.write("418.6, 4280.9, -5.0, -5.2, 'fo1751'\n")
            f.write("\n")
            f.write("[FINE SCANS]\n")
            f.write("# Format: x_stage, y_stage, (x_min, x_max), (y_min, y_max), label\n")
            f.write("448.8, 4260.7, (-40, 40), (-30, 30), 'f01742'\n")
            f.write("418.7, 4280.7, (-40, 40), (-30, 30), 'f01743'\n")
        
        print(f"Please edit {coords_file_name} with your coordinates and run the script again.\n")
        return
    
    # Read point measurements and fine scans from the coordinates file
    point_measurements, fine_scans = read_measure_coords(coords_file_path)
    
    if not point_measurements and not fine_scans:
        print("No valid point measurements or fine scans found in the coordinates file.\n")
        return
    
    print(f"Loaded {len(point_measurements)} point measurements and {len(fine_scans)} fine scans from {coords_file_name}")
    
    # Find all txt files in the directory (excluding the coordinates file)
    txt_files = [f for f in glob.glob(os.path.join(script_dir, "*.txt")) 
                if os.path.basename(f).lower() != coords_file_name.lower()]
    
    if not txt_files:
        print("No .txt-format maps data found in the directory.")
        print(" -> Creating an overview plot with all point measurements and fine scans...")
        
        if output_dir:
            save_path = os.path.join(output_dir, "overview_plot.png")
        else:
            save_path = None
            
        create_overview_plot(point_measurements, fine_scans, save_path)
        return
    
    print(f"Found {len(txt_files)} .txt-format maps files to process.")
    
    for txt_file in txt_files:
        file_name = os.path.basename(txt_file)
        print(f" -> Processing: {file_name}")
        
        if output_dir:
            save_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.png")
        else:
            save_path = None
            
        success = plot_graph_from_file(txt_file, point_measurements, fine_scans, save_path)
        
        if success and save_path:
            print(f"Plot saved to: {save_path}")
    print()

if __name__ == "__main__":

    output_dir = "Experiment maps"   # Comment to display plots instead of saving them
    
    # Process all txt files and generate plots
    # This will read coordinates from "MeasureCoords.txt" by default
    process_all_txt_files(coords_file_name="MeasureCoords.txt", output_dir=output_dir)