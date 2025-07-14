import numpy as np
import pandas as pd
from scipy import optimize, signal
import warnings
from typing import Dict, Tuple, List, Optional
import matplotlib.pyplot as plt


def fit_energy_distribution_curve(
        edc_data: Dict, 
        range: Optional[Tuple[float, float]] = None,
        temperature: float = 295.0,
        background_type: str = 'shirley',
        max_peaks: int = 5,
        min_peak_height: float = 0.1,
        peak_detection_prominence: float = 0.1,
        convergence_threshold: float = 1e-6,
        max_iterations: int = 1000,
        instrumental_sigma: Optional[float] = None
    ) -> Dict:
    """
    Fit an Energy Distribution Curve with physically accurate Lorentzian-Fermi-Gaussian model.
    
    Each peak is modeled as:
    1. Lorentzian profile (intrinsic line shape)
    2. Multiplied by Fermi-Dirac distribution (electronic occupation)
    3. Convolved with Gaussian (instrumental resolution)
    
    Parameters:
    -----------
    edc_data : dict
        Input dictionary with 'metadata' and 'data' keys
        - 'metadata': dict with experimental parameters
        - 'data': pandas Series with energy as index (referenced to Fermi level)
            and intensity as values
    range : (float, float)
        Energy range in which to perform the fit
    temperature : float
        Temperature in Kelvin for Fermi-Dirac distribution (default: 295K)
    background_type : str
        Type of background ('linear', 'poly3', 'exponential', 'shirley')
    max_peaks : int
        Maximum number of peaks to fit
    min_peak_height : float
        Minimum relative height for peak detection
    peak_detection_prominence : float
        Prominence threshold for peak detection
    convergence_threshold : float
        Convergence threshold for fitting
    max_iterations : int
        Maximum iterations for optimization
    instrumental_sigma : float, optional
        Gaussian width for instrumental broadening. If None, estimated from data
        
    Returns:
    --------
    dict
        Dictionary with fitted results including metadata and DataFrame with
        energy axis, background, individual peaks, and envelope
    """
    
    # Extract data
    energy = edc_data['data'].index.values
    intensity = edc_data['data'].values
    
    # Constrain the fit by filtering for values in the given range
    if range is not None:
        min_val, max_val = range
        mask = (energy >= min_val) & (energy <= max_val)
        energy = energy[mask]
        intensity = intensity[mask]
    
    # Physical constants
    kb = 8.617333e-5  # Boltzmann constant in eV/K
    
    # Estimate instrumental resolution if not provided
    if instrumental_sigma is None:
        # Estimate from energy resolution (typical values for photoemission)
        energy_range = energy.max() - energy.min()
        instrumental_sigma = energy_range / 1000  # Conservative estimate
    
    # Step 1: Background subtraction
    background = _fit_background(energy, intensity, background_type)
    intensity_corrected = intensity - background
    
    # Step 2: Peak detection
    peak_positions = _detect_peaks(energy, intensity_corrected, min_peak_height, peak_detection_prominence)
    
    # Step 3: Iterative peak fitting
    fitted_params, instrumental_sigma_fitted, final_residual = _iterative_peak_fitting(
        energy, intensity_corrected, peak_positions, temperature, kb,
        instrumental_sigma, max_peaks, convergence_threshold, max_iterations
    )
    
    # Step 4: Create output DataFrame
    output_df = pd.DataFrame(index=energy)
    output_df['background'] = background
    
    # Add individual peaks
    for i, peak_params in enumerate(fitted_params):
        peak_curve = _lorentzian_fermi_gaussian_model(energy, *peak_params, temperature, kb, instrumental_sigma_fitted)
        output_df[f'peak_{i+1}'] = peak_curve
    
    # Add envelope (sum of all peaks)
    envelope = np.sum([
        _lorentzian_fermi_gaussian_model(energy, *params, temperature, kb, instrumental_sigma_fitted) for params in fitted_params
        ], axis=0)
    output_df['envelope'] = envelope
    
    # Step 5: Create comprehensive metadata
    output_metadata = _create_output_metadata(
        edc_data['metadata'],
        fitted_params,
        temperature,
        background_type,
        final_residual,
        len(fitted_params),
        instrumental_sigma_fitted
    )
    
    return {'metadata': output_metadata, 'data': output_df}


def _fermi_dirac(energy: np.ndarray, temperature: float, kb: float) -> np.ndarray:
    """Fermi-Dirac distribution function."""
    # Avoid overflow in exponential
    x = energy / (kb * temperature)
    return 1.0 / (1.0 + np.exp(np.clip(x, -500, 500)))


def _lorentzian_profile(energy: np.ndarray, center: float, amplitude: float, gamma: float) -> np.ndarray:
    """Lorentzian profile (natural line shape)."""
    return amplitude * gamma**2 / ((energy - center)**2 + gamma**2)


def _gaussian_kernel(energy: np.ndarray, sigma: float) -> np.ndarray:
    """Normalized Gaussian kernel for convolution."""
    return np.exp(-0.5 * (energy / sigma)**2) / (sigma * np.sqrt(2 * np.pi))


def _lorentzian_fermi_gaussian_model(energy: np.ndarray, center: float, amplitude: float,
                                   gamma: float, temperature: float, kb: float,
                                   instrumental_sigma: float) -> np.ndarray:
    """
    Physical model: Lorentzian * Fermi-Dirac, then convolved with Gaussian.
    
    This represents the complete physics:
    1. Intrinsic Lorentzian line shape
    2. Fermi-Dirac cutoff (electronic occupation)
    3. Instrumental Gaussian broadening
    """
    # Step 1: Create Lorentzian profile
    lorentzian = _lorentzian_profile(energy, center, amplitude, gamma)
    
    # Step 2: Multiply by Fermi-Dirac distribution
    fermi = _fermi_dirac(energy, temperature, kb)
    lorentzian_fermi = lorentzian * fermi
    
    # Step 3: Convolve with Gaussian (instrumental resolution)
    # For convolution, we need to be careful with the energy grid
    de = np.abs(energy[1] - energy[0])  # Energy step
    
    # Create Gaussian kernel with same energy spacing
    # Kernel width should be several times sigma
    kernel_width = int(6 * instrumental_sigma / de)
    if kernel_width % 2 == 0:
        kernel_width += 1  # Make odd for symmetry
    
    kernel_energy = np.linspace(-3*instrumental_sigma, 3*instrumental_sigma, kernel_width)
    kernel = _gaussian_kernel(kernel_energy, instrumental_sigma)
    
    # Perform convolution
    # Use 'same' mode to keep same length as input
    convolved = np.convolve(lorentzian_fermi, kernel, mode='same') * de
    
    return convolved


def _fit_background(energy: np.ndarray, intensity: np.ndarray, bg_type: str) -> np.ndarray:
    """Fit and return background."""
    if bg_type == 'linear':
        # Linear background
        p = np.polyfit(energy, intensity, 1)
        return np.polyval(p, energy)
    
    elif bg_type == 'poly3':
        # Polynomial background (degree 3)
        p = np.polyfit(energy, intensity, 3)
        return np.polyval(p, energy)
    
    elif bg_type == 'exponential':
        # Exponential background
        def exp_bg(x, a, b, c):
            return a * np.exp(b * x) + c
        
        try:
            popt, _ = optimize.curve_fit(exp_bg, energy, intensity, p0=[intensity.min(), -0.1, 0])
            return exp_bg(energy, *popt)
        except:
            # Fall back to linear if exponential fails
            p = np.polyfit(energy, intensity, 1)
            return np.polyval(p, energy)
    
    elif bg_type == 'shirley':
        # Shirley background (iterative)
        return _shirley_background(energy, intensity)
    
    else:
        raise ValueError(f"Unknown background type: {bg_type}")


def _shirley_background(energy: np.ndarray, intensity: np.ndarray, max_iter: int = 50, tolerance: float = 1e-6) -> np.ndarray:
    """Calculate Shirley background using iterative method."""
    # Sort by energy (descending)
    sort_idx = np.flip(np.argsort(energy))
    e_sorted = energy[sort_idx]
    i_sorted = intensity[sort_idx]
    
    # Average the initial and final values
    initial_i = np.mean(i_sorted[:5])
    final_i = np.mean(i_sorted[-5:])
    
    # Initialize background
    bg = np.linspace(initial_i, final_i, len(i_sorted))
    
    # Iterative calculation
    for iteration in range(max_iter):
        bg_old = bg.copy()
        
        # Calculate cumulative integral
        cumsum = np.cumsum(i_sorted - bg)
        total_integral = cumsum[-1]
        
        if total_integral == 0:
            break
            
        # Update background
        for i in range(len(bg)):
            if i == 0:
                bg[i] = initial_i
            else:
                bg[i] =  initial_i + (final_i - initial_i) * cumsum[i] / total_integral
        
        # Check convergence
        if np.max(np.abs(bg - bg_old)) < tolerance:
            break
    
    # Restore original order
    bg_original_order = np.zeros_like(intensity)
    bg_original_order[sort_idx] = bg
    
    return bg_original_order


def _detect_peaks(energy: np.ndarray, intensity: np.ndarray, min_height: float, prominence: float) -> List[float]:
    """Detect peak positions in the spectrum."""
    # Normalize intensity for peak detection
    intensity_norm = (intensity - intensity.min()) / (intensity.max() - intensity.min())
    
    # Apply a Savitzky-Golay filter for improving peak detection in noisy data
    intensity_norm = signal.savgol_filter(intensity_norm, 15, 3)
    
    # Find peaks
    peaks, properties = signal.find_peaks(intensity_norm, height=min_height, prominence=prominence)
    
    # Convert indices to energy values
    peak_energies = energy[peaks].tolist()
    
    # Sort peaks by intensity (highest first)
    peak_intensities = intensity_norm[peaks]
    sorted_indices = np.argsort(peak_intensities)[::-1]
    
    return [peak_energies[i] for i in sorted_indices]


def _iterative_peak_fitting(
        energy: np.ndarray, intensity: np.ndarray,
        initial_peaks: List[float], temperature: float, kb: float,
        instrumental_sigma: float, max_peaks: int,
        threshold: float, max_iter: int
    ) -> Tuple[List[List[float]], float, float]:
    """
    Iteratively fit peaks until convergence or maximum number reached.
    Includes instrumental_sigma as a fitted parameter.
    """
    fitted_params = []
    current_residual = intensity.copy()
    fitted_instrumental_sigma = instrumental_sigma
    
    for n_peaks in range(min(len(initial_peaks), max_peaks)):
        # Add next peak
        peak_energy = initial_peaks[n_peaks]
        
        # Initial parameter guess for new peak
        peak_intensity = max(np.interp(peak_energy, energy, current_residual), 0.01)
        initial_guess = [peak_energy, peak_intensity, 0.1]  # center, amplitude, gamma
        
        # Add to current parameter list
        current_params = fitted_params + [initial_guess]
        
        # Fit all peaks simultaneously (including instrumental sigma)
        try:
            fitted_params_flat, fitted_sigma = _fit_multiple_peaks(
                energy, intensity, current_params, temperature, kb, 
                fitted_instrumental_sigma, max_iter
            )
            
            # Reshape parameters (3 per peak: center, amplitude, gamma)
            fitted_params = [fitted_params_flat[i:i+3] for i in range(0, len(fitted_params_flat), 3)]
            fitted_instrumental_sigma = fitted_sigma
            
            # Calculate residual
            model = np.sum([
                _lorentzian_fermi_gaussian_model(energy, *params, temperature, kb, fitted_instrumental_sigma) 
                for params in fitted_params
                ], axis=0)
            new_residual = intensity - model
            residual_improvement = np.sum((current_residual - new_residual)**2)
            
            # Check for convergence
            if residual_improvement < threshold * np.sum(intensity**2):
                break
                
            current_residual = new_residual
            
        except Exception as e:
            warnings.warn(f"Peak fitting failed at peak {n_peaks + 1}: {str(e)}")
            break
    
    final_residual = np.sqrt(np.mean(current_residual**2))
    
    return fitted_params, fitted_instrumental_sigma, final_residual


def _fit_multiple_peaks(
        energy: np.ndarray, intensity: np.ndarray,
        initial_params: List[List[float]], temperature: float, 
        kb: float, instrumental_sigma: float, 
        max_iter: int
    ) -> Tuple[List[float], float]:
    """Fit multiple peaks simultaneously, including instrumental sigma."""
    
    # Flatten parameters (3 per peak)
    params_flat = [param for peak_params in initial_params for param in peak_params]
    
    # Add instrumental sigma to parameters
    all_params = params_flat + [instrumental_sigma]
    
    def objective(params):
        # Extract instrumental sigma
        sigma_instr = params[-1]
        peak_params_flat = params[:-1]
        
        # Reshape peak parameters
        peak_params = [peak_params_flat[i:i+3] for i in range(0, len(peak_params_flat), 3)]
        
        # Calculate model
        model = np.sum([
            _lorentzian_fermi_gaussian_model(energy, *p_params, temperature, kb, sigma_instr) 
            for p_params in peak_params
        ], axis=0)
        
        # Return residual
        return intensity - model
    
    # Set parameter bounds
    bounds = []
    for _ in initial_params:
        bounds.extend([
            (energy.min(), energy.max()),  # center
            (0, intensity.max() * 10),     # amplitude
            (0.001, 0.5),                  # gamma (Lorentzian width)
        ])
    # Add bounds for instrumental sigma
    bounds.append((0.001, 1.0))  # instrumental sigma
    
    # Perform fit
    try:
        result = optimize.least_squares(objective, all_params, bounds=tuple(zip(*bounds)), max_nfev=max_iter, ftol=1e-12, xtol=1e-12)
        fitted_params = result.x[:-1].tolist()
        fitted_sigma = result.x[-1]
        return fitted_params, fitted_sigma
    except Exception as e:
        warnings.warn(f"Optimization failed: {str(e)}")
        return params_flat, instrumental_sigma


def _create_output_metadata(input_metadata: Dict, fitted_params: List[List[float]],
                          temperature: float, background_type: str,
                          final_residual: float, n_peaks: int, 
                          instrumental_sigma: float) -> Dict:
    """Create comprehensive output metadata."""
    
    output_metadata = input_metadata.copy()
    
    # Add fitting parameters
    output_metadata['fitting'] = {
        'temperature_K': temperature,
        'background_type': background_type,
        'n_peaks_fitted': n_peaks,
        'final_residual_rms': final_residual,
        'instrumental_sigma_eV': instrumental_sigma,
        'convergence_achieved': True,
        'model_type': 'Lorentzian_Fermi_Gaussian'
    }
    
    # Add peak parameters
    output_metadata['peaks'] = {}
    for i, params in enumerate(fitted_params):
        center, amplitude, gamma = params
        output_metadata['peaks'][f'peak_{i+1}'] = {
            'center_eV': center,
            'amplitude': amplitude,
            'lorentzian_width_eV': gamma,
            'fwhm_lorentzian_eV': 2 * gamma,  # FWHM of Lorentzian
            'intrinsic_fwhm_eV': 2 * gamma,   # Before instrumental broadening
            'observed_fwhm_eV': 2 * np.sqrt(gamma**2 + (instrumental_sigma * 2.355)**2 / 2.355**2)  # After convolution
        }
    
    return output_metadata


# Example usage and test function
def test_edc_fitting():
    """Test function with synthetic data."""
    # Create synthetic EDC data
    energy = np.linspace(-3, 1, 1000)
    
    # Create synthetic spectrum with two peaks using the new model
    # Peak 1: Lorentzian * Fermi
    peak1_lorentz = 500 * 0.1**2 / ((energy - (-1.5))**2 + 0.1**2)
    peak2_lorentz = 300 * 0.08**2 / ((energy - (-0.5))**2 + 0.08**2)
    
    # Add Fermi cutoff
    fermi = 1.0 / (1.0 + np.exp(energy / (8.617333e-5 * 300)))
    
    # Apply Fermi cutoff to peaks
    peak1_fermi = peak1_lorentz * fermi
    peak2_fermi = peak2_lorentz * fermi
    
    # Convolve with Gaussian (instrumental resolution)
    sigma_instr = 0.05
    de = energy[1] - energy[0]
    kernel_width = int(6 * sigma_instr / de)
    if kernel_width % 2 == 0:
        kernel_width += 1
    
    kernel_energy = np.linspace(-3*sigma_instr, 3*sigma_instr, kernel_width)
    kernel = np.exp(-0.5 * (kernel_energy / sigma_instr)**2) / (sigma_instr * np.sqrt(2 * np.pi))
    
    peak1 = np.convolve(peak1_fermi, kernel, mode='same') * de
    peak2 = np.convolve(peak2_fermi, kernel, mode='same') * de
    
    # Add background and noise
    background = 50 + 10 * energy
    noise = np.random.normal(0, 5, len(energy))
    
    intensity = peak1 + peak2 + background + noise
    
    # Create input dictionary
    edc_input = {
        'metadata': {
            'sample': 'Test Sample',
            'photon_energy_eV': 21.2,
            'temperature_K': 300,
            'analyzer': 'Test Analyzer'
        },
        'data': pd.Series(intensity, index=energy)
    }
    
    # Fit the data
    result = fit_energy_distribution_curve(edc_input, temperature=300, 
                                         instrumental_sigma=0.05)
    
    return result

if __name__ == "__main__":
    # Run test
    test_result = test_edc_fitting()
    print("Test completed successfully!")
    print(f"Number of peaks fitted: {test_result['metadata']['fitting']['n_peaks_fitted']}")
    print(f"Final residual: {test_result['metadata']['fitting']['final_residual_rms']:.6f}")