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
from pathlib import Path
from tqdm import tqdm


# === k-E data functions ===

def read_spectral_data_kE(file_path,  
                          min_e=-1, max_e=0.0):
    """
    Read spectral data from a txt file and return it as a DataFrame with proper indexing.
    
    Parameters:
    file_path : str or Path
        Path to the data file
    min_e : float
        Minimum energy value to include
    max_e : float
        Maximum energy value to include
    
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
    energy_mask = energy_values <= max_energy
    energy_values = energy_values[energy_mask]
    
    # Extract wavevector values
    wavevector_values = data.iloc[:, 0].values
    
    # Create DataFrame with proper indexing
    intensity_data = data.iloc[:, 1:].values[:, energy_mask]
    data_df = pd.DataFrame(
        intensity_data,
        index=wavevector_values,
        columns=energy_values
    )
    data_df.index.name = 'Wavevector'
    data_df.columns.name = 'Energy'
    
    return spectrum_name, data_df


def plot_spectrum_kE(spectrum_name, data_df, output_dir=None, size=7, aspect_ratio=1.5,
                    plot_type='heatmap', colormap='viridis', normalize=True):
    """
    Create and display visualizations of the spectral data.
    
    Parameters:
    spectrum_name : str
        Name of the spectrum
    data_df : pandas.DataFrame
        DataFrame containing the intensity data with wavevectors as index and energies as columns
    output_dir : str or Path, optional
        Directory for saving plots. If None, plots are only displayed inline
    size : int, optional
        The height of the final pictures
    aspect_ratio : float, optional
        The aspect ratio of the final pictures, multiplies 'size' to compute the width
    plot_type : str
        Type of plot to create ('heatmap', 'contour', '3d_surface', or 'all')
    colormap : str
        Matplotlib colormap to use
    normalize : bool
        Whether to normalize intensity values to [0,1]
    
    Returns:
    dict: Dictionary of created figures
    """
    # Normalize data if requested
    if normalize:
        plot_data = (data_df - data_df.min().min()) / (data_df.max().max() - data_df.min().min())
    else:
        plot_data = data_df
    
    # Create the coordinates for the 2D plots
    X, Y = np.meshgrid(plot_data.index, plot_data.columns)

    def create_heatmap():
        fig = plt.figure(figsize=(np.floor(size*aspect_ratio), size))
        plt.pcolormesh(X, Y, plot_data.T,
                      cmap=colormap)
        plt.colorbar(label='Intensity')
        plt.xlabel(r'k$_{//}$ ($\AA^{-1}$)')
        plt.ylabel(r'E-E$_F$ (eV)')
        plt.title(f'Spectrum: {spectrum_name} (Heatmap)')
        return fig
    
    def create_contour():
        fig = plt.figure(figsize=(np.floor(size*aspect_ratio), size))
        plt.contourf(X, Y, plot_data.T, levels=40, cmap=colormap)
        plt.colorbar(label='Intensity')
        plt.xlabel(r'k$_{//}$ ($\AA^{-1}$)')
        plt.ylabel(r'E-E$_F$ (eV)')
        plt.title(f'Spectrum: {spectrum_name} (Contour)')
        return fig
    
    def create_3d_surface():
        fig = plt.figure(figsize=(np.floor(size*aspect_ratio), size))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, plot_data.T,
                              cmap=colormap,
                              linewidth=0,
                              antialiased=True)
        fig.colorbar(surf, label='Intensity')
        ax.set_xlabel(r'k$_{//}$ ($\AA^{-1}$)')
        ax.set_ylabel(r'E-E$_F$ (eV)')
        ax.set_zlabel('Intensity')
        plt.title(f'Spectrum: {spectrum_name} (3D Surface)')
        return fig

    # Create links dictionary to plot functions
    plot_functions = {
        'heatmap': create_heatmap,
        'contour': create_contour,
        '3d_surface': create_3d_surface
    }

    # Prepare iterable with required plots
    if plot_type == 'all':
        plot_types = list(plot_functions.keys())
    else:
        plot_types = [plot_type]
    
    # Loop through all plots
    figures = {}
    for pt in plot_types:
        fig = plot_functions[pt]()
        figures[pt] = fig
        
        # Save if output directory is provided
        if output_dir:
            output_path = os.path.join(output_dir, f'{spectrum_name}_spectrum_{pt}.png')
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved as: {output_path}")
        
        # Show the figure in non-notebook environment
        plt.show()
        plt.close(fig)
    
    return figures


def process_spectral_data_kE(data_dir=None, output_dir=None, store_data=False, size=7, aspect_ratio=1.5,
                           plot_type='heatmap', colormap='viridis', normalize=True, max_energy=0.0):
    """
    Process all spectral data files in the specified directory.
    
    Parameters:
    data_dir : str or Path, optional
        Directory containing the data files. Defaults to current directory.
    output_dir : str or Path, optional
        Directory for saving plots. Defaults to None (plots only displayed inline)
    store_data : bool, optional
        Whether to store data for subsequent custom elaboration
    size : int, optional
        The height of the final pictures
    aspect_ratio : float, optional
        The aspect ratio of the final pictures, multiplies 'size' to compute the width
    plot_type : str, optional
        Type of plot to create ('heatmap', 'contour', '3d_surface', or 'all')
    colormap : str, optional
        Matplotlib colormap to use
    normalize : bool, optional
        Whether to normalize intensity values
    max_energy : float, optional
        Maximum energy value to include
        
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
    
    print(f"Processing {len(txt_files)} files...")
    
    # Process each file
    processed_data = {}
    for file_path in tqdm(txt_files, desc="Processing spectra"):
        try:
            # Read and process the data
            spectrum_name, data_df = read_spectral_data_kE(file_path, max_energy=max_energy)
            
            # Store processed data
            if store_data:
                processed_data[file_path.name] = {
                    'name': spectrum_name,
                    'data': data_df
                }
            
            # Create and display plots
            plot_spectrum_kE(
                spectrum_name,
                data_df,
                output_dir=output_dir,
                size=size,
                aspect_ratio=aspect_ratio,
                plot_type=plot_type,
                colormap=colormap,
                normalize=normalize
            )
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")
    
    return processed_data


# === k-k data functions ===

def read_spectral_data_kk(file_path):
    """
    Read spectral data from a txt file and return it as a DataFrame with proper indexing.
    
    Parameters:
    file_path : str or Path
        Path to the data file
    
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
    
    # Extract kx and ky values
    ky_values = pd.to_numeric(data.columns[1:])
    kx_values = data.iloc[:, 0].values
    
    # Create DataFrame with proper indexing
    intensity_data = data.iloc[:, 1:].values
    data_df = pd.DataFrame(
        intensity_data,
        index=kx_values,
        columns=ky_values
    )
    data_df.index.name = 'kx'
    data_df.columns.name = 'ky'
    
    return spectrum_name, data_df


def plot_spectrum_kk(spectrum_name, data_df, output_dir=None, aspect_ratio=(10,8),
                    plot_type='contour', colormap='viridis', normalize=True):
    """
    Create and display visualizations of the spectral data.
    
    Parameters:
    spectrum_name : str
        Name of the spectrum
    data_df : pandas.DataFrame
        DataFrame containing the intensity data with wavevectors as index and energies as columns
    output_dir : str or Path, optional
        Directory for saving plots. If None, plots are only displayed inline
    plot_type : str
        Type of plot to create ('heatmap', 'contour', '3d_surface', or 'all')
    colormap : str
        Matplotlib colormap to use
    normalize : bool
        Whether to normalize intensity values to [0,1]
    
    Returns:
    dict: Dictionary of created figures
    """
    # Normalize data if requested
    if normalize:
        plot_data = (data_df - data_df.min().min()) / (data_df.max().max() - data_df.min().min())
    else:
        plot_data = data_df
    
    # Create the coordinates for the 2D plots
    X, Y = np.meshgrid(plot_data.index, plot_data.columns)

    def create_heatmap():
        fig = plt.figure(figsize=aspect_ratio)
        plt.pcolormesh(X, Y, plot_data.T, cmap=colormap)
        plt.colorbar(label='Intensity')
        plt.xlabel(r'k$_{x}$ ($\AA^{-1}$)')
        plt.ylabel(r'k$_{y}$ ($\AA^{-1}$)')
        plt.title(f'Spectrum: {spectrum_name} (Heatmap)')
        return fig
    
    def create_contour():
        fig = plt.figure(figsize=aspect_ratio)
        plt.contourf(X, Y, plot_data.T, levels=50, cmap=colormap)
        plt.colorbar(label='Intensity')
        plt.xlabel(r'k$_{x}$ ($\AA^{-1}$)')
        plt.ylabel(r'k$_{y}$ ($\AA^{-1}$)')
        plt.title(f'Spectrum: {spectrum_name} (Contour)')
        return fig
    
    def create_3d_surface():
        fig = plt.figure(figsize=aspect_ratio)
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, plot_data.T,
                              cmap=colormap,
                              linewidth=0,
                              antialiased=True)
        fig.colorbar(surf, label='Intensity')
        ax.set_xlabel(r'k$_{x}$ ($\AA^{-1}$)')
        ax.set_ylabel(r'k$_{y}$ ($\AA^{-1}$)')
        ax.set_zlabel('Intensity')
        plt.title(f'Spectrum: {spectrum_name} (3D Surface)')
        return fig

    # Create links dictionary to plot functions
    plot_functions = {
        'heatmap': create_heatmap,
        'contour': create_contour,
        '3d_surface': create_3d_surface
    }

    # Prepare iterable with required plots
    if plot_type == 'all':
        plot_types = list(plot_functions.keys())
    else:
        plot_types = [plot_type]
    
    # Loop through all plots
    figures = {}
    for pt in plot_types:
        fig = plot_functions[pt]()
        figures[pt] = fig
        
        # Save if output directory is provided
        if output_dir:
            output_path = os.path.join(output_dir, f'{spectrum_name}_spectrum_{pt}.png')
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved as: {output_path}")
        
        # Show the figure in non-notebook environment
        plt.show()
        plt.close(fig)
    
    return figures


def process_spectral_data_kk(data_dir=None, output_dir=None, store_data=False,
                           on_scale_size=3, format_override=(0,0),
                           plot_type='heatmap', colormap='viridis', normalize=True):
    """
    Process all spectral data files in the specified directory.
    
    Parameters:
    data_dir : str or Path, optional
        Directory containing the data files. Defaults to current directory.
    output_dir : str or Path, optional
        Directory for saving plots. Defaults to None (plots only displayed inline)
    store_data : bool, optional
        Whether to store processed data
    on_scale_size : int, optional
        Base size for automatic aspect ratio calculation
    format_override : tuple, optional
        Override the automatically calculated aspect ratio with fixed values
    plot_type : str, optional
        Type of plot to create ('heatmap', 'contour', '3d_surface', or 'all')
    colormap : str, optional
        Matplotlib colormap to use
    normalize : bool, optional
        Whether to normalize intensity values
    
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
    
    print(f"Processing {len(txt_files)} files...")
    
    # Process each file
    processed_data = {}
    for file_path in tqdm(txt_files, desc="Processing spectra"):
        try:
            # Read and process the data
            spectrum_name, data_df = read_spectral_data_kk(file_path)
            
            # Store processed data
            if store_data:
                processed_data[file_path.name] = {
                    'name': spectrum_name,
                    'data': data_df
                }

            # Determine the format of the final pictures
            if format_override != (0,0):
                aspect_ratio = format_override
            else:
                aspect_ratio = (
                    on_scale_size*(1 + np.floor(abs(
                        (data_df.index.values[-1]-data_df.index.values[0]) / (data_df.columns.values[-1]-data_df.columns.values[0])
                    ))), on_scale_size)
                        
            # Create and display plots
            plot_spectrum_kk(
                spectrum_name,
                data_df,
                output_dir=output_dir,
                aspect_ratio=aspect_ratio,
                plot_type=plot_type,
                colormap=colormap,
                normalize=normalize
            )
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")
    
    return processed_data


# === Main script ===

if __name__ == "__main__":
    # Example usage for k-E data
    print("\nProcessing k-E data:")
    results_kE = process_spectral_data_kE(
        data_dir='k-E_data_to_process',
        output_dir='plots',
        # store_data=True,
        # size=10,
        # aspect_ratio=2,
        plot_type='contour',
        colormap='plasma',
        normalize=True,
        max_energy=0.0
    )
    
    # Example usage for k-k data
    print("\nProcessing k-k data:")
    results_kk = process_spectral_data_kk(
        data_dir='k-k_data_to_process',
        output_dir='plots',
        # store_data=True,
        on_scale_size=4,
        # format_override=(20,6),
        plot_type='contour',
        colormap='plasma',
        normalize=True
    )
    
    print("\nProcessing complete!")