import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import re
import ast

def plot_graph_from_file(file_path, points, rectangles, save_path=None):
    """
    Plot intensity map from a txt file and add points and rectangles based on specified rules.
    
    Args:
        file_path: Path to the txt file containing intensity data
        points: List of points [x, y, label]
        rectangles: List of rectangles [x_stage, y_stage, x_scan_exten, y_scan_exten, label]
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
        
        # Filter and plot points that are within the data limits
        valid_points = []
        for point in points:
            x, y, label = point
            if x_min <= x <= x_max and y_min <= y <= y_max:
                valid_points.append(point)
                ax.plot(x, y, 'o', label=label)
        
        # Filter and plot rectangles with at least two vertices inside limits
        for rect in rectangles:
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
        plt.title(f"Intensity Map: {os.path.basename(file_path)}")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
        return True
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False

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
        points = []
        rectangles = []
        
        current_section = None
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Check for section headers
                if line.upper() == '[POINTS]':
                    current_section = 'points'
                    continue
                elif line.upper() == '[RECTANGLES]':
                    current_section = 'rectangles'
                    continue
                
                # Process data based on current section
                if current_section == 'points':
                    try:
                        # Split the line by comma and strip whitespace
                        parts = [p.strip() for p in line.split(',', 2)]
                        if len(parts) == 3:
                            x = float(eval(parts[0]))  # Allow expressions like 418.6+20.5
                            y = float(eval(parts[1]))  # Allow expressions like 4280.7-6.8
                            label = parts[2].strip("'\"")  # Remove quotes if present
                            points.append((x, y, label))
                    except Exception as e:
                        print(f"Error parsing point: {line}. Error: {str(e)}")
                
                elif current_section == 'rectangles':
                    try:
                        # Use regex to find all parts
                        matches = re.match(r'(.*?),\s*(.*?),\s*(\(.*?\)),\s*(\(.*?\)),\s*(.*?)$', line)
                        if matches:
                            x_stage = float(eval(matches.group(1)))
                            y_stage = float(eval(matches.group(2)))
                            x_scan_exten = ast.literal_eval(matches.group(3))
                            y_scan_exten = ast.literal_eval(matches.group(4))
                            label = matches.group(5).strip("'\"")
                            rectangles.append((x_stage, y_stage, x_scan_exten, y_scan_exten, label))
                    except Exception as e:
                        print(f"Error parsing rectangle: {line}. Error: {str(e)}")
        
        return points, rectangles
    
    except Exception as e:
        print(f"Error reading measure coordinates file: {str(e)}")
        return [], []

def process_all_txt_files(coords_file_name="MeasureCoords.txt", output_dir=None):
    """
    Process all txt files in the script's directory and generate plots.
    
    Args:
        coords_file_name: Name of the file containing points and rectangles coordinates
        output_dir: Directory to save output plots (optional)
    """
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
        print(f"Creating a template coordinates file at {coords_file_path}")
        
        # Create a template file
        with open(coords_file_path, 'w') as f:
            f.write("[POINTS]\n")
            f.write("# Format: x, y, label\n")
            f.write("418.6+20.5, 4280.7-6.8, '1744-50,52-56,60'\n")
            f.write("418.6-5.0, 4280.9-5.2, '1751,57-59,61'\n")
            f.write("\n")
            f.write("[RECTANGLES]\n")
            f.write("# Format: x_stage, y_stage, (x_min, x_max), (y_min, y_max), label\n")
            f.write("448.8, 4260.7, (-40, 40), (-30, 30), '1742'\n")
            f.write("418.7, 4280.7, (-40, 40), (-30, 30), '1743'\n")
        
        print(f"Please edit {coords_file_name} with your coordinates and run the script again.")
        return
    
    # Read points and rectangles from the coordinates file
    points, rectangles = read_measure_coords(coords_file_path)
    
    if not points and not rectangles:
        print("No valid points or rectangles found in the coordinates file.")
        return
    
    print(f"Loaded {len(points)} points and {len(rectangles)} rectangles from {coords_file_name}")
    
    # Find all txt files in the directory (excluding the coordinates file)
    txt_files = [f for f in glob.glob(os.path.join(script_dir, "*.txt")) 
                if os.path.basename(f).lower() != coords_file_name.lower()]
    
    if not txt_files:
        print("No data .txt files found in the directory.")
        return
    
    print(f"Found {len(txt_files)} data .txt files to process.")
    
    for txt_file in txt_files:
        file_name = os.path.basename(txt_file)
        print(f"Processing: {file_name}")
        
        if output_dir:
            save_path = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.png")
        else:
            save_path = None
            
        success = plot_graph_from_file(txt_file, points, rectangles, save_path)
        
        if success and save_path:
            print(f"Saved plot to: {save_path}")

if __name__ == "__main__":
    # Set to None to display plots instead of saving them
    # or specify a directory path to save all plots
    output_dir = "intensity_maps"
    
    # Process all txt files and generate plots
    # This will read coordinates from "MeasureCoords.txt" by default
    process_all_txt_files(coords_file_name="MeasureCoords.txt", output_dir=output_dir)