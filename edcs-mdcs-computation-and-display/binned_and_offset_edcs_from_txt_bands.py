import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from pathlib import Path

class ARPESAnalyzer:
    def __init__(self, data_folder):
        """
        Initialize ARPES analyzer
        
        Parameters:
        data_folder (str): Path to folder containing .txt files
        """
        self.data_folder = data_folder
        self.data_files = []
        self.data_dict = {}
        self.energies = None
        self.wavenumbers = None
        
    def load_data(self):
        """Load all tab-delimited .txt files from the specified folder"""
        # Find all .txt files in the folder
        txt_files = glob.glob(os.path.join(self.data_folder, "*.txt"))
        txt_files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        
        if not txt_files:
            raise ValueError(f"No .txt files found in {self.data_folder}")
        
        print(f"Found {len(txt_files)} .txt files")
        
        # Load the first file to extract energy and wavenumber axes
        first_file = True
        
        for file_path in txt_files:
            filename = Path(file_path).stem
            try:
                # Load tab-delimited file, skip first row (header)
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                # Skip header (first line)
                # Second line contains energy values
                energy_line = lines[1].strip().split('\t')
                # Remove first element (empty or label) and convert to float
                energies_from_file = np.array([float(x) for x in energy_line[0:] if x.strip()])
                
                # Load the rest of the data (wavenumbers in first column, intensities in the rest)
                data_lines = lines[2:]  # Skip header and energy line
                
                wavenumbers_from_file = []
                intensity_data = []
                
                for line in data_lines:
                    if line.strip():  # Skip empty lines
                        values = line.strip().split('\t')
                        wavenumbers_from_file.append(float(values[0]))
                        intensity_row = [float(x) for x in values[1:] if x.strip()]
                        intensity_data.append(intensity_row)
                
                wavenumbers_from_file = np.array(wavenumbers_from_file)
                intensity_data = np.array(intensity_data)
                
                # Store the intensity data
                self.data_dict[filename] = intensity_data
                self.data_files.append(filename)
                
                # Set energy and wavenumber axes from the first file (assuming all files share the same axes)
                if first_file:
                    self.energies = energies_from_file
                    self.wavenumbers = wavenumbers_from_file
                    first_file = False
                    print(f"Energy range: {self.energies[0]:.3f} to {self.energies[-1]:.3f} eV ({len(self.energies)} points)")
                    print(f"Wavenumber range: {self.wavenumbers[0]:.3f} to {self.wavenumbers[-1]:.3f} Å⁻¹ ({len(self.wavenumbers)} points)")
                
                print(f"Loaded {filename}: shape {intensity_data.shape}")
                
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                # Remove from data_files if it was added
                if filename in self.data_files:
                    self.data_files.remove(filename)
            
    def plot_tiled_layout(self, output_dir, figsize=None, cmap='viridis'):
        """
        Plot all ARPES data in a flexible tiled layout (4 columns, variable rows)
        
        Parameters:
        figsize (tuple): Figure size (auto-calculated if None)
        cmap (str): Colormap for the plots
        """
        n_files = len(self.data_files)
        if n_files == 0:
            print("No data to plot")
            return
        
        # Calculate grid dimensions: 4 columns, variable rows
        n_cols = 4
        n_rows = int(np.ceil(n_files / n_cols))
        
        # Auto-calculate figure size if not provided
        if figsize is None:
            figsize = (4 * n_cols, 4 * n_rows)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=True, sharey=True)
        
        # Handle case where there's only one row
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        # Handle case where there's only one column (shouldn't happen with n_cols=4, but for safety)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        axes = axes.flatten()
        
        for i, filename in enumerate(self.data_files):
            data = self.data_dict[filename]
            
            im = axes[i].imshow(data.T, aspect='auto', origin='lower', 
                            extent=[self.wavenumbers[0], self.wavenumbers[-1],
                                    self.energies[0], self.energies[-1]],
                            cmap=cmap)
            axes[i].set_title(filename, fontsize=10)
            
            # Add x-axis labels to bottom row
            if i >= (n_rows - 1) * n_cols:
                axes[i].set_xlabel('kx (Å⁻¹)')
            # Add y-axis labels to leftmost column
            if i % n_cols == 0:
                axes[i].set_ylabel('Energy (eV)')
        
        # Hide unused subplots
        for i in range(n_files, n_rows * n_cols):
            axes[i].set_visible(False)
        
        # plt.tight_layout()
        
        # Add colorbar
        plt.colorbar(im, ax=axes[:n_files], shrink=0.8, label='Intensity (counts)')
        plt.suptitle(f'ARPES Data - {n_files} Files ({n_rows}×{n_cols} Layout)', fontsize=16, y=0.98)
        
        save_path = os.path.join(output_dir, "bands_gallery.png")
        plt.savefig(save_path)
        plt.close()
    
    def compute_edc(self, kx_target, kx_bin_width=0.05):
        """
        Compute EDCs at a specific wavenumber with optional binning
        
        Parameters:
        kx_target (float): Target wavenumber for EDC extraction
        kx_bin_width (float): Binning width around target wavenumber
        
        Returns:
        dict: Dictionary containing EDCs for each file
        """
        edcs = {}
        
        # Find indices for binning around target kx
        kx_min = kx_target - kx_bin_width/2
        kx_max = kx_target + kx_bin_width/2
        
        kx_indices = np.where((self.wavenumbers >= kx_min) & 
                             (self.wavenumbers <= kx_max))[0]
        
        if len(kx_indices) == 0:
            raise ValueError(f"No data points found around kx = {kx_target}")
        
        print(f"Computing EDCs at kx = {kx_target:.3f} ± {kx_bin_width/2:.3f}")
        print(f"Using {len(kx_indices)} wavenumber points for binning")
        
        for filename in self.data_files:
            data = self.data_dict[filename]
            # Average over the binned wavenumber range
            edc = np.mean(data[kx_indices, :], axis=0)
            edcs[filename] = edc
            
        return edcs
    
    def plot_edcs(self, edcs, output_dir, energy_range=(-2.0, 0.5), y_offset=None, 
                  figsize=(4, 8), colors=None):
        """
        Plot EDCs with y-axis offset
        
        Parameters:
        edcs (dict): Dictionary of EDCs from compute_edc()
        energy_range (tuple): Energy range to plot (eV)
        y_offset (float): Vertical offset between curves (auto if None)
        figsize (tuple): Figure size
        colors (list): List of colors for each EDC
        """
        # Filter energy range
        e_mask = (self.energies >= energy_range[0]) & (self.energies <= energy_range[1])
        e_plot = self.energies[e_mask]
        
        # Auto-calculate y_offset if not provided
        if y_offset is None:
            max_intensities = [np.max(edc[e_mask]) for edc in edcs.values()]
            y_offset = np.mean(max_intensities) * 0.3
        
        plt.figure(figsize=figsize)
        
        # Set up colors
        if colors is None:
            colors = plt.cm.tab10(np.linspace(0, 1, len(edcs)))
        
        for i, (filename, edc) in enumerate(edcs.items()):
            edc_plot = edc[e_mask]
            offset_edc = edc_plot + i * y_offset
            
            # plt.plot(e_plot, offset_edc, label=filename, 
            #         color=colors[i % len(colors)], linewidth=1.5)
            plt.plot(e_plot, offset_edc, label=filename, 
                    color='0', linewidth=1.5)
        
        plt.xlabel('Energy (eV)', fontsize=12)
        plt.ylabel('Intensity (counts) + offset', fontsize=12)
        plt.title('Energy Distribution Curves (EDCs)', fontsize=14)
        # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, "edcs.png")
        plt.savefig(save_path)
        plt.close()
    
    def plot_intensity_vs_files(self, edcs, output_dir, target_energy, energy_tolerance=0.05,
                               figsize=(12, 6), marker='o', line_style='-'):
        """
        Plot intensity values from EDCs at a specific energy against filenames
        
        Parameters:
        edcs (dict): Dictionary of EDCs from compute_edc()
        target_energy (float): Target energy value (eV)
        energy_tolerance (float): Tolerance for energy matching (eV)
        figsize (tuple): Figure size
        marker (str): Marker style for data points
        line_style (str): Line style connecting points
        """
        if not edcs:
            print("No EDC data provided")
            return
        
        # Find the energy index closest to target_energy
        energy_diff = np.abs(self.energies - target_energy)
        
        # Check if target energy is within the available range
        if np.min(energy_diff) > energy_tolerance:
            print(f"Warning: Target energy {target_energy:.3f} eV is not within tolerance "
                  f"({energy_tolerance:.3f} eV) of available energies.")
            print(f"Closest available energy: {self.energies[np.argmin(energy_diff)]:.3f} eV")
        
        target_idx = np.argmin(energy_diff)
        actual_energy = self.energies[target_idx]
        
        # Extract intensities at the target energy for all files
        filenames = list(edcs.keys())
        intensities = []
        
        for filename in filenames:
            edc = edcs[filename]
            intensity = edc[target_idx]
            intensities.append(intensity)
        
        intensities = np.array(intensities)
        
        # Create the plot
        plt.figure(figsize=figsize)
        
        # Plot points and connecting line
        x_positions = np.arange(len(filenames))
        plt.plot(x_positions, intensities, marker=marker, linestyle=line_style, 
                linewidth=2, markersize=8, markerfacecolor='red', markeredgecolor='darkred')
        
        # Customize the plot
        plt.xlabel('Files', fontsize=12)
        plt.ylabel('Intensity (counts)', fontsize=12)
        plt.title(f'Intensity vs Files at E = {actual_energy:.3f} eV', fontsize=14)
        
        # Set x-axis labels
        plt.xticks(x_positions, filenames, rotation=45, ha='right')
        
        # Add grid for better readability
        plt.grid(True, alpha=0.3)
        
        # Add value labels on each point
        for i, (filename, intensity) in enumerate(zip(filenames, intensities)):
            plt.annotate(f'{intensity:.1f}', 
                        (i, intensity), 
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center', 
                        fontsize=9)
        
        # Add statistics text box
        stats_text = f'Statistics:\nMean: {np.mean(intensities):.1f}\nStd: {np.std(intensities):.1f}\nMin: {np.min(intensities):.1f}\nMax: {np.max(intensities):.1f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, "intensity_analysis.png")
        plt.savefig(save_path)
        plt.close()
        
        # Print summary
        print(f"\nIntensity analysis at E = {actual_energy:.3f} eV:")
        print(f"{'Filename':<20} {'Intensity':<10}")
        print("-" * 35)
        for filename, intensity in zip(filenames, intensities):
            print(f"{filename:<20} {intensity:<10.2f}")
        print(f"\nStatistics: Mean = {np.mean(intensities):.2f}, Std = {np.std(intensities):.2f}")

# Example usage
def main():
    # Set your data folder path here
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = script_dir  # Change this to another path if data are not with the script
    
    # Set your plot save path here
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = ARPESAnalyzer(data_folder)
    
    try:
        # Print an empty line to begin with
        print()
        
        # Load all .txt files
        analyzer.load_data()
        
        # Plot tiled layout of all files
        analyzer.plot_tiled_layout(output_dir)
        
        # Compute EDCs at specific wavenumber
        kx_target = 584.541  # Target wavenumber (Å⁻¹)
        kx_bin_width = 41  # Binning width (Å⁻¹)
        
        edcs = analyzer.compute_edc(kx_target=kx_target, 
                                   kx_bin_width=kx_bin_width)
        
        # Plot EDCs with offset
        analyzer.plot_edcs(edcs, output_dir, energy_range=(-1.4, 0.2), y_offset=None)
        
        # Plot intensity vs files at a specific energy
        target_energy = -0.0626537  # Energy value (eV) where you want to compare intensities
        analyzer.plot_intensity_vs_files(edcs, output_dir, target_energy=target_energy,
                                          energy_tolerance=0.05, )        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

# Additional utility functions for advanced analysis

def batch_edc_analysis(data_folder, kx_values, kx_bin_width=0.05, 
                      energy_range=(-2.0, 0.5)):
    """
    Perform EDC analysis for multiple kx values
    
    Parameters:
    data_folder (str): Path to data folder
    kx_values (list): List of kx values to analyze
    kx_bin_width (float): Binning width
    energy_range (tuple): Energy range for plotting
    """
    analyzer = ARPESAnalyzer(data_folder)
    analyzer.load_data()
    
    for kx in kx_values:
        print(f"\n--- Analysis for kx = {kx:.3f} ---")
        edcs = analyzer.compute_edc(kx_target=kx, kx_bin_width=kx_bin_width)
        analyzer.plot_edcs(edcs, energy_range=energy_range)

def export_edcs_to_csv(edcs, energies, output_file="edcs_export.csv"):
    """
    Export EDCs to CSV file
    
    Parameters:
    edcs (dict): Dictionary of EDCs
    energies (array): Energy array
    output_file (str): Output CSV filename
    """
    import pandas as pd
    
    # Create DataFrame
    data = {'Energy_eV': energies}
    for filename, edc in edcs.items():
        data[filename] = edc
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    print(f"EDCs exported to {output_file}")