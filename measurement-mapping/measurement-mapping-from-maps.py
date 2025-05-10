import matplotlib.pyplot as plt
import numpy as np
import os
import glob

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

def process_all_txt_files(points, rectangles, output_dir=None):
    """
    Process all txt files in the script's directory and generate plots.
    
    Args:
        points: List of points [x, y, label]
        rectangles: List of rectangles [x_stage, y_stage, x_scan_exten, y_scan_exten, label]
        output_dir: Directory to save output plots (optional)
    """
    # Get the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Find all txt files in the directory
    txt_files = glob.glob(os.path.join(script_dir, "*.txt"))
    
    if not txt_files:
        print("No .txt files found in the directory.")
        return
    
    print(f"Found {len(txt_files)} .txt files to process.")
    
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

# Define points and rectangles (same as in your example)
points = [
    (418.6 +20.5, 4280.7 - 6.8, '1744-50,52-56,60'), 
    (418.6 - 5.0, 4280.9 - 5.2, '1751,57-59,61'),
    (418.6 - 5.0, 4280.9 +11.6, '1762,66'),
    (418.6 +20.3, 4280.9 -18.0, '1763,67,68'),
    (418.6 -33.0, 4280.9 +20.0, '1764,65')
]

rectangles = [
    (448.8, 4260.7, (-40,40), (-30,30), '1742'),
    (418.7, 4280.7, (-40,40), (-30,30), '1743'),
    (418.6, 4280.9, (- 5,20), (-18,11), '1769')
]

if __name__ == "__main__":
    # Set to None to display plots instead of saving them
    # or specify a directory path to save all plots
    output_dir = "intensity_maps"
    
    # Process all txt files and generate plots
    process_all_txt_files(points, rectangles, output_dir)