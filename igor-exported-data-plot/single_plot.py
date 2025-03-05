#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ARPES Data Plotting Script
This script provides functions to read, process, and visualize ARPES (Angle-Resolved Photoemission Spectroscopy) data.
It handles both k-E (energy dispersion) and k-k (momentum mapping) data formats.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path
from tqdm import tqdm


# === k-E data functions ===

def read_spectral_data_kE(
        file_path, 
        min_wavevector=-0.2, 
        max_wavevector=0.2, 
        min_energy=-1, 
        max_energy=0.1
        ):
    """
    Read spectral data from a txt file and return it as a DataFrame with proper indexing.
    
    Parameters:
    file_path : str or Path
        Path to the data file
    min_wavevector : float
        Minimum wavevector value to include in the output DataFrame
    max_wavevector : float
        Maximum wavevector value to include in the output DataFrame
    min_energy : float
        Minimum energy value to include in the output DataFrame
    max_energy : float
        Maximum energy value to include in the output DataFrame
    
    Returns:
    tuple: (spectrum_name, data_df)
        spectrum_name: name of the spectrum
        data_df: DataFrame with wavevectors as index and energies as columns
    """
    # Read the first line to get spectrum name
    with open(file_path, 'r') as f:
        spectrum_name = f.readline().strip()
    
    # Read the rest of the data
    data = pd.read_csv(file_path, sep='\t', skiprows=1)
    
    # Extract and filter energy values
    energy_values = pd.to_numeric(data.columns[1:])
    energy_mask = (energy_values <= max_energy) & (min_energy <= energy_values)
    energy_values = energy_values[energy_mask]
    
    # Extract and filter wavevector values
    wavevector_values = data.iloc[:, 0].values
    wavevector_mask = (min_wavevector <= wavevector_values) & (wavevector_values <= max_wavevector)
    wavevector_values = wavevector_values[wavevector_mask]
    
    # Create DataFrame with proper indexing
    intensity_data = data.iloc[wavevector_mask, 1:].values[:, energy_mask]
    data_df = pd.DataFrame(
        intensity_data,
        index=wavevector_values,
        columns=energy_values
        )
    
    # Appropriately label rows and columns of the DataFrame
    data_df.index.name = 'Wavevector'
    data_df.columns.name = 'Energy'
    
    return spectrum_name, data_df


def plot_spectrum_kE(
        spectrum_name, 
        data_df, 
        vertical_size=6, 
        aspect_ratio=1.5,
        colormap='viridis', 
        normalize=True
        ):
    """
    Create and display visualizations of the spectral data.
    
    Parameters:
    spectrum_name : str
        Name of the spectrum
    data_df : pandas.DataFrame
        DataFrame containing the intensity data with wavevectors as index and energies as columns
    vertical_size : int, optional
        The height of the final pictures
    aspect_ratio : float, optional
        The aspect ratio of the final pictures, multiplies 'vertical_size' to compute the width
    colormap : str
        Matplotlib colormap to use
    normalize : bool
        Whether to normalize intensity values to [0,1]
    
    Returns:
    figure: matplotlib figure object
    """
    # Normalize data if requested
    if normalize:
        plot_data = (data_df - data_df.min().min()) / (data_df.max().max() - data_df.min().min())
    else:
        plot_data = data_df
    
    # Create the coordinates for the 2D plots
    X, Y = np.meshgrid(plot_data.index, plot_data.columns)
    
    # Create the figure with specified size
    fig, ax = plt.subplots(
        figsize=(np.floor(vertical_size*aspect_ratio), vertical_size)
        )
    
    # Plot the data
    pcm = ax.pcolormesh(
        X, 
        Y, 
        plot_data.T, 
        cmap=colormap, 
        shading='gouraud'
        )
    
    # Complement the plot with colorbar, axes labels and title
    plt.colorbar(pcm, label='Intensity (arb. u.)')
    ax.set_xlabel(r'k$_{//}$ ($\AA^{-1}$)')
    ax.set_ylabel(r'E-E$_F$ (eV)')
    ax.set_title(f'{spectrum_name}')
    
    return fig


def process_spectral_data_kE(
        data_dir=None, 
        output_dir=None, 
        min_wavevector=-0.2, 
        max_wavevector=0.2, 
        min_energy=-1, 
        max_energy=0.1,
        store_data=False, 
        vertical_size=7, 
        aspect_ratio=1.5, 
        colormap='viridis', 
        normalize=True, 
        fontsize=15,
        show_plot=False,
        img_format='png',
        dpi=300,
        ):
    """
    Process all spectral data files in the specified directory.
    
    Parameters:
    data_dir : str or Path, optional
        Directory containing the data files. Defaults to current directory.
    output_dir : str or Path, optional
        Directory for saving plots. Defaults to None (plots only displayed inline)
    min_wavevector : float, optional
        Minimum wavevector value to include in the output DataFrame
    max_wavevector : float, optional
        Maximum wavevector value to include in the output DataFrame
    min_energy : float, optional
        Minimum energy value to include in the output DataFrame
    max_energy : float, optional
        Maximum energy value to include in the output DataFrame
    store_data : bool, optional
        Whether to store data for subsequent custom elaboration
    vertical_size : int, optional
        The height of the final pictures
    aspect_ratio : float, optional
        The aspect ratio of the final pictures, multiplies 'vertical_size' to compute the width
    colormap : str, optional
        Matplotlib colormap to use
    normalize : bool, optional
        Whether to normalize intensity values
    show_plot : bool, optional
        Whether to show the plot in a pop-up window, pausing the execution
    img_format : str, optional
        String specifying in which format to save the plot
    dpi : int, optional
        The quality of the saved plot file, useful only for raster formats
    
    Returns:
    dict: Dictionary mapping file names to their processed DataFrames
    """
    # Set up directories
    data_dir = Path(data_dir) if data_dir else Path.cwd()
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    
    # Get all txt files
    txt_files = list(data_dir.glob('*.txt'))
    if not txt_files:
        print("No .txt files found in the specified directory.")
        return {}
    
    print(f"Found {len(txt_files)} files...")
    
    # Update all font sizes
    rcParams['font.size'] = fontsize
    
    # Process each file
    processed_data = {}
    for file_path in tqdm(txt_files, desc="Processing spectra"):
        try:
            # Read and process the data
            spectrum_name, data_df = read_spectral_data_kE(
                file_path, 
                min_wavevector=min_wavevector, max_wavevector=max_wavevector, 
                min_energy=min_energy, max_energy=max_energy
                )
            
            # Optionally store processed data
            if store_data:
                processed_data[spectrum_name] = data_df
            
            # Create and display plots
            fig = plot_spectrum_kE(
                spectrum_name,
                data_df,
                vertical_size=vertical_size,
                aspect_ratio=aspect_ratio,
                colormap=colormap,
                normalize=normalize
                )
            
            # Save the plot, if output directory is provided
            if output_dir:
                output_path = os.path.join(output_dir, f'{spectrum_name}.{img_format}')
                fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
                print(f"\nPlot saved as: {output_path}")
            
            # Optionally show the figure
            if show_plot:
                plt.show()
            
            plt.close(fig)
        
        except Exception as e:
            print(f"\nError processing {file_path.name}: {str(e)}")
    
    return processed_data


# === k-k data functions ===

def read_spectral_data_kk(
        file_path,
        min_kx=-0.5, max_kx=0.5, 
        min_ky=-0.5, max_ky=0.5
        ):
    """
    Read spectral data from a txt file and return it as a DataFrame with proper indexing.
    
    Parameters:
    file_path : str or Path
        Path to the data file
    min_kx : float
        Minimum wavevector in the x direction to include in the output DataFrame
    max_kx : float
        Maximum wavevector in the x direction to include in the output DataFrame
    min_ky : float
        Minimum wavevector in the y direction to include in the output DataFrame
    max_ky : float
        Maximum wavevector in the y direction to include in the output DataFrame
    
    Returns:
    tuple: (spectrum_name, data_df)
        spectrum_name: name of the spectrum
        data_df: DataFrame with wavevectors as index and energies as columns
    """
    # Read the first line to get spectrum name
    with open(file_path, 'r') as f:
        spectrum_name = f.readline().strip()
    
    # Read the rest of the data
    data = pd.read_csv(file_path, sep='\t', skiprows=1)
    
    # Extract and filter kx values
    kx_values = data.iloc[:, 0].values
    kx_mask = (min_kx <= kx_values) & (kx_values <= max_kx)
    kx_values = kx_values[kx_mask]
    
    # Extract and filter ky values
    ky_values = pd.to_numeric(data.columns[1:])
    ky_mask = (min_ky <= ky_values) & (ky_values <= max_ky)
    ky_values = ky_values[ky_mask]

    # Create DataFrame with proper indexing
    intensity_data = data.iloc[kx_mask, 1:].values[:, ky_mask]
    data_df = pd.DataFrame(
        intensity_data,
        index=kx_values,
        columns=ky_values
        )
    
    # Appropriately label rows and columns of the DataFrame
    data_df.index.name = 'kx'
    data_df.columns.name = 'ky'
    
    return spectrum_name, data_df


def plot_spectrum_kk(
        spectrum_name, 
        data_df, 
        vertical_size=6, aspect_ratio=1.5, 
        colormap='viridis', 
        normalize=True
        ):
    """
    Create and display visualizations of the spectral data.
    
    Parameters:
    spectrum_name : str
        Name of the spectrum
    data_df : pandas.DataFrame
        DataFrame containing the intensity data with wavevectors as index and energies as columns
    vertical_size : int, optional
        The height of the final pictures
    aspect_ratio : float, optional
        The aspect ratio of the final pictures, multiplies 'vertical_size' to compute the width
    colormap : str
        Matplotlib colormap to use
    normalize : bool
        Whether to normalize intensity values to [0,1]
    
    Returns:
    figure: matplotlib figure object
    """
    # Normalize data if requested
    if normalize:
        plot_data = (data_df - data_df.min().min()) / (data_df.max().max() - data_df.min().min())
    else:
        plot_data = data_df
    
    # Create the coordinates for the 2D plots
    X, Y = np.meshgrid(plot_data.index, plot_data.columns)
    
    # Create the figure with specified size
    fig, ax = plt.subplots(
        figsize=(np.floor(vertical_size*aspect_ratio), vertical_size)
        )
    
    # Plot the data
    pcm = ax.pcolormesh(
        X, 
        Y, 
        plot_data.T, 
        cmap=colormap, 
        shading='gouraud'
        )
    
    # Complement the plot with colorbar, axes labels and title
    plt.colorbar(pcm, label='Intensity')
    ax.set_xlabel(r'k$_{x}$ ($\AA^{-1}$)')
    ax.set_ylabel(r'k$_{y}$ ($\AA^{-1}$)')
    ax.set_title(f'{spectrum_name}')
    
    return fig


def process_spectral_data_kk(
        data_dir=None, output_dir=None, 
        min_kx=-0.5, max_kx=0.5, 
        min_ky=-0.5, max_ky=0.5,
        store_data=False, 
        vertical_size=7, aspect_ratio=1.5, 
        colormap='viridis', 
        normalize=True,
        fontsize=15,
        show_plot=False,
        img_format='png',
        dpi=300,
        ):
    """
    Process all spectral data files in the specified directory.
    
    Parameters:
    data_dir : str or Path, optional
        Directory containing the data files. Defaults to current directory.
    output_dir : str or Path, optional
        Directory for saving plots. Defaults to None (plots only displayed inline)
    min_kx : float
        Minimum wavevector in the x direction to include in the output DataFrame
    max_kx : float
        Maximum wavevector in the x direction to include in the output DataFrame
    min_ky : float
        Minimum wavevector in the y direction to include in the output DataFrame
    max_ky : float
        Maximum wavevector in the y direction to include in the output DataFrame
    store_data : bool, optional
        Whether to store processed data
    vertical_size : int, optional
        The height of the final pictures
    aspect_ratio : float, optional
        The aspect ratio of the final pictures, multiplies 'vertical_size' to compute the width
    colormap : str, optional
        Matplotlib colormap to use
    normalize : bool, optional
        Whether to normalize intensity values
    show_plot : bool, optional
        Whether to show the plot in a pop-up window, pausing the execution
    img_format : str, optional
        String specifying in which format to save the plot
    dpi : int, optional
        The quality of the saved plot file, useful only for raster formats
    
    Returns:
    dict: Dictionary mapping file names to their processed DataFrames
    """
    # Set up directories
    data_dir = Path(data_dir) if data_dir else Path.cwd()
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
    
    # Get all txt files
    txt_files = list(data_dir.glob('*.txt'))
    if not txt_files:
        print("No .txt files found in the specified directory.")
        return {}
    
    print(f"Found {len(txt_files)} files...")
    
    # Update all font sizes
    rcParams['font.size'] = fontsize
    
    # Process each file
    processed_data = {}
    for file_path in tqdm(txt_files, desc="Processing spectra"):
        try:
            # Read and process the data
            spectrum_name, data_df = read_spectral_data_kk(
                file_path,
                min_kx=min_kx, max_kx=max_kx, 
                min_ky=min_ky, max_ky=max_ky,
                )
            
            # Optionally store processed data
            if store_data:
                processed_data[spectrum_name] = data_df
            
            # Create and display plots
            fig = plot_spectrum_kk(
                spectrum_name,
                data_df,
                vertical_size=vertical_size,
                aspect_ratio=aspect_ratio,
                colormap=colormap,
                normalize=normalize
                )
            
            # Save the plot, if output directory is provided
            if output_dir:
                output_path = os.path.join(output_dir, f'{spectrum_name}.{img_format}')
                fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
                print(f"\nPlot saved as: {output_path}")
            
            # Optionally show the figure
            if show_plot:
                plt.show()
            
            plt.close(fig)
        
        except Exception as e:
            print(f"\nError processing {file_path.name}: {str(e)}")
    
    return processed_data


# === Main script ===

if __name__ == "__main__":

    # Analyze and plot k-E data
    print("\n----- Processing k-E data -----")
    results_kE = process_spectral_data_kE(
        data_dir='k-E_data_to_process',
        output_dir='plots',  # Activate to save plots
        min_wavevector= -0.3,
        max_wavevector=  2.0, 
        min_energy= -0.75,
        max_energy=  0.1,
        # store_data=True,     # Activate to store the data
        # vertical_size=10,
        aspect_ratio=0.8,
        colormap='plasma',
        # normalize=False,   # Keep commented to normalize the plot (default)
        fontsize=12,         # Defaults to 15
        show_plot=True,      # Activate to show plots as pop-ups
        # img_format='svg',    # Activate to save vectorial images (defaults to png)
        # dpi=600,             # Activate to save high-quality raster (defaults to 300)
        )
    
    # Analyze and plot isoenergetic k-k data
    print("\n----- Processing k-k data -----")
    results_kk = process_spectral_data_kk(
        data_dir='k-k_data_to_process',
        output_dir='plots',  # Activate to save plots
        min_kx= -0.5, 
        max_kx=  0.5, 
        min_ky= -0.5, 
        max_ky=  2.0,
        # store_data=True,     # Activate to store the data
        # vertical_size=10,
        aspect_ratio=0.8,
        colormap='plasma',
        # normalize=False,   # Keep commented to normalize the plot (default)
        fontsize=12,         # Defaults to 15
        show_plot=True,      # Activate to show plots as pop-ups
        # img_format='svg',    # Activate to save vectorial images (defaults to png)
        # dpi=600,             # Activate to save high-quality raster (defaults to 300)
        )
    
    print("\nProcessing complete!\n")