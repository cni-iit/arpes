import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from typing import Dict, Any, Optional, Union, Tuple, List



#=====================#
# FILTERING FUNCTIONS #
#=====================#


# Smooth a single kE dataset via Savitzky-Golay filter
#  FROM  { md:{}, d:DF(k,E) }
#    TO  { md:{}, d:DF(k,E) }

def apply_savgol_smoothing(
        kE_dataset: Dict[str, Any], 
        window_length: int = 5, 
        polyorder: int = 2,
        axis: str = 'k',
        show_comparison: bool = True
    ) -> Dict[str, Any]:
    """
    Applies Savitzky-Golay smoothing to a single kE dataset.
    
    Parameters:
    -----------
    input_dict : dict 
        Dictionary with keys 'metadata' and 'data'.
    window_length : int
        The length of the filter window (must be odd and >= polyorder + 2).
    polyorder : int 
        The order of the polynomial used to fit the samples.
    axis : str
        Specifies along which axis to apply the smoothing
    show_comparison : bool
        Whether to display a comparison of original and smoothed data
    
    Returns:
    --------
    dict
        Dictionary with smoothed data and updated metadata.
    """
    # Extract metadata and data
    metadata = kE_dataset.get('metadata', {})
    data = kE_dataset.get('data')
    
    # Apply Savitzky-Golay filter along the chosen axis
    if axis == 'k':
        smoothed_data = pd.DataFrame(
            savgol_filter(data.values, window_length=window_length, polyorder=polyorder, axis=1),
            index=data.index,
            columns=data.columns
        )
    elif axis == 'E':
        smoothed_data = pd.DataFrame(
            savgol_filter(data.values, window_length=window_length, polyorder=polyorder, axis=0),
            index=data.index,
            columns=data.columns
        )
    else:
        raise ValueError("Expected 'k' of 'E' input as axis")
    
    # Construct new metadata
    new_metadata = {
        'name': metadata.get('name'),
        'units': metadata.get('units'),
        'k_range': metadata.get('k_range'),
        'E_range': metadata.get('E_range'),
        'k_step': metadata.get('k_step'),
        'E_step': metadata.get('E_step'),
        'smoothing method': 'Savitzky-Golay',
        'smoothing axis': axis,
        'savgol parameters': {
            'window_length': window_length,
            'polyorder': polyorder
        },
        'original metadata': metadata,
    }
    
    if show_comparison:
        # Plot comparison of original and smoothed data
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        im0 = axes[0].imshow(data.values, aspect='auto', origin='lower',
            extent=[data.columns.min(), data.columns.max(), data.index.min(), data.index.max()])
        axes[0].set_title('Original Data')
        axes[0].set_xlabel('Wavenumber')
        axes[0].set_ylabel('Energy')
        plt.colorbar(im0, ax=axes[0], label='Intensity')
        
        im1 = axes[1].imshow(smoothed_data.values, aspect='auto', origin='lower',
        extent=[data.columns.min(), data.columns.max(), data.index.min(), data.index.max()])
        axes[1].set_title('Smoothed Data (Savitzky-Golay)')
        axes[1].set_xlabel('Wavenumber')
        plt.colorbar(im1, ax=axes[1], label='Intensity')
        
        plt.tight_layout()
        plt.show()
    
    return {'metadata': new_metadata, 'data': smoothed_data}


# Smooth a single kE dataset via 2D Gaussian filter
#  FROM  { md:{}, d:DF(k,E) }
#    TO  { md:{}, d:DF(k,E) }

def apply_gaussian_filtering(
        kE_dataset: Dict[str, Any],
        sigma: float = 1.0,
        show_comparison: bool = True
    ) -> Dict[str, Any]:
    """
    Applies 2D Gaussian filtering to a single kE dataset.
    
    Parameters:
    -----------
    kE_dataset : dict
        Dictionary with keys 'metadata' and 'data'.
    sigma : float
        Standard deviation for Gaussian kernel.
    show_comparison : bool
        Whether to display a comparison of original and filtered data
    
    Returns:
    --------
    dict
        Dictionary with filtered data and updated metadata.
    """
    # Extract metadata and data
    metadata = kE_dataset.get('metadata', {})
    data = kE_dataset.get('data')
    
    # Apply 2D Gaussian filter
    filtered_array = gaussian_filter(data.values, sigma=sigma)
    filtered_data = pd.DataFrame(filtered_array, index=data.index, columns=data.columns)
    
    if show_comparison:
        # Plot comparison of original and smoothed data
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        im0 = axes[0].imshow(data.values, aspect='auto', origin='lower',
                            extent=[data.columns.min(), data.columns.max(), data.index.min(), data.index.max()])
        axes[0].set_title('Original Data')
        axes[0].set_xlabel('Wavenumber')
        axes[0].set_ylabel('Energy')
        plt.colorbar(im0, ax=axes[0])
        
        im1 = axes[1].imshow(filtered_data.values, aspect='auto', origin='lower',
                            extent=[data.columns.min(), data.columns.max(), data.index.min(), data.index.max()])
        axes[1].set_title('Gaussian Filtered Data')
        axes[1].set_xlabel('Wavenumber')
        plt.colorbar(im1, ax=axes[1])
        
        plt.tight_layout()
        plt.show()
    
    # Construct new metadata
    new_metadata = {
        'name': metadata.get('name'),
        'units': metadata.get('units'),
        'k_range': metadata.get('k_range'),
        'E_range': metadata.get('E_range'),
        'k_step': metadata.get('k_step'),
        'E_step': metadata.get('E_step'),
        'filtering method': '2D Gaussian',
        'gaussian parameters': {
            'sigma': sigma
        },
        'original metadata': metadata,
    }
    
    return {'metadata': new_metadata, 'data': filtered_data}


# Smooth a single kE dataset via 2D low-pass Fourier filter
#  FROM  { md:{}, d:DF(k,E) }
#    TO  { md:{}, d:DF(k,E) }

def apply_fourier_filtering(
        kE_dataset: Dict[str, Any],
        cutoff_frequency : float = 0.1,
        show_comparison: bool = True,
        show_frequencies: bool = True
    ) -> Dict[str, Any]:
    """
    Applies Fourier low-pass filtering to a single kE dataset.
    
    Parameters:
    -----------
    kE_dataset : dict
        Dictionary with keys 'metadata' and 'data'.
    cutoff_frequency : float
        Normalized cutoff frequency (0 to 1) for low-pass filtering.
    show_comparison : bool
        Whether to display a comparison of original and filtered data
    show_frequencies : bool
        Whether to display a comparison of original and filtered frequencies according to Fourier transform
    
    Returns:
    --------
    dict
        Dictionary with filtered data and updated metadata.
    """
    # Extract metadata and data
    metadata = kE_dataset.get('metadata', {})
    data = kE_dataset.get('data')
    
    # Perform 2D Fourier transform
    fft_data = np.fft.fft2(data.values)
    fft_shifted = np.fft.fftshift(fft_data)
    
    # Create a low-pass filter mask
    rows, cols = data.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), dtype=np.float32)
    radius = int(min(rows, cols) * cutoff_frequency)
    for i in range(rows):
        for j in range(cols):
            if (i - crow)**2 + (j - ccol)**2 <= radius**2:
                mask[i, j] = 1
    
    # Apply mask
    filtered_fft = fft_shifted * mask
    filtered_data_array = np.fft.ifft2(np.fft.ifftshift(filtered_fft)).real
    filtered_data = pd.DataFrame(filtered_data_array, index=data.index, columns=data.columns)
    
    if show_comparison:
        # Plot comparison of original and filtered data
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        im0 = axes[0].imshow(
            data.values, aspect='auto', origin='lower',
            extent=[data.columns.min(), data.columns.max(), data.index.min(), data.index.max()]
        )
        axes[0].set_title('Original Data')
        axes[0].set_xlabel('Wavenumber')
        axes[0].set_ylabel('Energy')
        plt.colorbar(im0, ax=axes[0])
        
        im1 = axes[1].imshow(filtered_data.values, aspect='auto', origin='lower',
                            extent=[data.columns.min(), data.columns.max(), data.index.min(), data.index.max()])
        axes[1].set_title('Fourier Filtered Data')
        axes[1].set_xlabel('Wavenumber')
        plt.colorbar(im1, ax=axes[1])
        
        plt.tight_layout()
        plt.show()
    
    if show_frequencies:
        # Plot magnitude spectrum before and after filtering
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        magnitude_original = np.log(np.abs(fft_shifted) + 1)
        magnitude_filtered = np.log(np.abs(filtered_fft) + 1)
        
        axes[0].imshow(magnitude_original, cmap='viridis')
        axes[0].set_title('Original Magnitude Spectrum')
        
        axes[1].imshow(magnitude_filtered, cmap='viridis')
        axes[1].set_title('Filtered Magnitude Spectrum')
        
        plt.tight_layout()
        plt.show()
    
    # Construct new metadata
    new_metadata = {
        'name': metadata.get('name'),
        'units': metadata.get('units'),
        'k_range': metadata.get('k_range'),
        'E_range': metadata.get('E_range'),
        'k_step': metadata.get('k_step'),
        'E_step': metadata.get('E_step'),
        'filtering method': 'Fourier Low-Pass',
        'fourier parameters': {
            'cutoff_frequency': cutoff_frequency
        },
        'original metadata': metadata
    }
    
    return {'metadata': new_metadata, 'data': filtered_data}


# Smooth a single kE dataset via PCA
#  FROM  { md:{}, d:DF(k,E) }
#    TO  { md:{}, d:DF(k,E) }

def apply_pca_filtering(
        kE_dataset: Dict[str, Any],
        n_components=2,
        show_comparison: bool = True,
        show_scatter_scree: bool = True,
        show_loadings: bool = True
    ) -> Dict[str, Any]:
    """
    Applies PCA-based filtering to a single kE dataset.
    
    Parameters:
    -----------
    kE_dataset : dict
        Dictionary with keys 'metadata' and 'data'.
    n_components : int
        Number of principal components to retain for reconstruction.
    show_comparison : bool
        Whether to display a comparison of original and filtered data
    show_scatter_scree : bool
        Whether to display scatterplot and screeplot
    show_loadings : bool
        Whether to display loading plots
    
    Returns:
    --------
    dict
        Dictionary with PCA-filtered data and updated metadata.
    """
    # Extract metadata and data
    metadata = kE_dataset.get('metadata', {})
    data = kE_dataset.get('data')
    
    # Flatten the 2D data into 2D matrix (samples x features)
    data_matrix = data.values.T
    original_shape = data_matrix.shape
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(data_matrix)
    reconstructed = pca.inverse_transform(transformed).T
    filtered_data = pd.DataFrame(reconstructed, index=data.index, columns=data.columns)
    
    if show_comparison:
        # Plot comparison
        fig1, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        im0 = axes[0].imshow(
            data.values, aspect='auto', origin='lower',
            extent=[data.columns.min(), data.columns.max(), data.index.min(), data.index.max()]
        )
        axes[0].set_title('Original Data')
        axes[0].set_xlabel('Wavenumber')
        axes[0].set_ylabel('Energy')
        plt.colorbar(im0, ax=axes[0])
        
        im1 = axes[1].imshow(
            reconstructed, aspect='auto', origin='lower',
            extent=[data.columns.min(), data.columns.max(), data.index.min(), data.index.max()]
        )
        axes[1].set_title('PCA Filtered Data')
        axes[1].set_xlabel('Wavenumber')
        plt.colorbar(im1, ax=axes[1])
        plt.tight_layout()
        plt.show()
    
    if show_scatter_scree:
        # Plot PC scatter and scree plot
        cmap = plt.get_cmap('coolwarm')
        colors = cmap(np.linspace(0, 1, original_shape[0]))
        fig2 = plt.figure(figsize=(10, 5))
        if n_components == 2:
            ax1 = fig2.add_subplot(1, 2, 1)
            ax1.scatter(transformed[:, 0], transformed[:, 1], c=colors, alpha=0.6)
            ax1.set_xlabel('PC1')
            ax1.set_ylabel('PC2')
            ax1.set_title('PC1 vs PC2')
        elif n_components >= 3:
            from mpl_toolkits.mplot3d import Axes3D
            ax1 = fig2.add_subplot(1, 2, 1, projection='3d')
            ax1.scatter(transformed[:, 0], transformed[:, 1], transformed[:, 2], c=colors, alpha=0.6)
            ax1.set_xlabel('PC1')
            ax1.set_ylabel('PC2')
            ax1.set_zlabel('PC3')
            ax1.set_title('PC1 vs PC2 vs PC3')
        
        ax2 = fig2.add_subplot(1, 2, 2)
        ax2.set_title('Scree Plot')
        ax2.set_xlabel('Principal Component')
        ax2.bar(np.arange(1, n_components + 1), pca.explained_variance_ratio_)
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        ax2.set_ylabel('Explained Variance Ratio', color='tab:blue')
        
        ax2s = ax2.twinx()
        ax2s.plot(np.arange(1, n_components + 1), np.cumsum(pca.explained_variance_ratio_), c='tab:orange', marker='o')
        ax2s.set_ylim(0.0, 1.0)
        ax2s.tick_params(axis='y', labelcolor='tab:orange')
        ax2s.set_ylabel('Cumulative Explained Variance', color='tab:orange')
        plt.tight_layout()
        plt.show()
    
    if show_loadings:
        # Plot loading plots
        fig3, axes = plt.subplots(nrows=(n_components + 2) // 3, ncols=3, figsize=(15, 3 * ((n_components + 2) // 3)))
        axes = axes.flatten()
        for i in range(n_components):
            axes[i].plot(data.index, pca.components_[i])
            axes[i].set_title(f'Loading Plot PC{i+1}')
            axes[i].set_xlabel('Energy')
            axes[i].set_ylabel('Loading')
        for j in range(n_components, len(axes)):
            fig3.delaxes(axes[j])
        plt.tight_layout()
        plt.show()
    
    # Construct new metadata
    new_metadata = {
        'name': metadata.get('name'),
        'units': metadata.get('units'),
        'k_range': metadata.get('k_range'),
        'E_range': metadata.get('E_range'),
        'k_step': metadata.get('k_step'),
        'E_step': metadata.get('E_step'),
        'filtering method': 'PCA',
        'pca parameters': {
            'n_components': n_components,
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist()
        },
        'original metadata': metadata
    }
    
    return {'metadata': new_metadata, 'data': filtered_data}

