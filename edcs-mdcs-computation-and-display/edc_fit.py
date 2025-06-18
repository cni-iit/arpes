import numpy as np
import pandas as pd
from scipy import optimize, special, signal
from scipy.special import voigt_profile
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
        max_iterations: int = 1000
    ) -> Dict:
    """
    Fit an Energy Distribution Curve with Voigt-Fermi convolutions.
    
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
    
    # Step 1: Background subtraction
    background = _fit_background(energy, intensity, background_type)
    intensity_corrected = intensity - background
    
    # Step 2: Peak detection
    peak_positions = _detect_peaks(energy, intensity_corrected, min_peak_height, peak_detection_prominence)
    
    # Step 3: Iterative peak fitting
    fitted_params, final_residual = _iterative_peak_fitting(
        energy, intensity_corrected, peak_positions, temperature, kb,
        max_peaks, convergence_threshold, max_iterations
    )
    
    # Step 4: Create output DataFrame
    output_df = pd.DataFrame(index=energy)
    output_df['background'] = background
    
    # Add individual peaks
    for i, peak_params in enumerate(fitted_params):
        peak_curve = _voigt_fermi_product(energy, *peak_params, temperature, kb)
        output_df[f'peak_{i+1}'] = peak_curve
    
    # Add envelope (sum of all peaks)
    envelope = np.sum([_voigt_fermi_product(energy, *params, temperature, kb) for params in fitted_params], axis=0)
    output_df['envelope'] = envelope
    
    # Step 5: Create comprehensive metadata
    output_metadata = _create_output_metadata(
        edc_data['metadata'],
        fitted_params,
        temperature,
        background_type,
        final_residual,
        len(fitted_params)
    )
    
    return {'metadata': output_metadata, 'data': output_df}


def _fermi_dirac(energy: np.ndarray, temperature: float, kb: float) -> np.ndarray:
    """Fermi-Dirac distribution function."""
    # Avoid overflow in exponential
    x = energy / (kb * temperature)
    return 1.0 / (1.0 + np.exp(np.clip(x, -500, 500)))


def _voigt_profile(energy: np.ndarray, center: float, amplitude: float, sigma: float, gamma: float) -> np.ndarray:
    """Voigt profile (convolution of Gaussian and Lorentzian)."""
    # Use scipy's voigt_profile which is more numerically stable
    return amplitude * voigt_profile(energy - center, sigma, gamma) / voigt_profile(0, sigma, gamma)


def _voigt_fermi_product(energy: np.ndarray, center: float, amplitude: float, sigma: float, gamma: float, temperature: float, kb: float) -> np.ndarray:
    """
    Product of Voigt profile and Fermi-Dirac distribution.
    This represents the broadened spectral line as observed in photoemission.
    """
    voigt = _voigt_profile(energy, center, amplitude, sigma, gamma)
    fermi = _fermi_dirac(energy, temperature, kb)
    
    return voigt * fermi


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
                bg[i] = initial_i + (final_i - initial_i) * cumsum[i] / total_integral
        
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
    intensity_norm = signal.savgol_filter(intensity_norm, 10, 1)
    
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
        max_peaks: int, threshold: float, max_iter: int
    ) -> Tuple[List[List[float]], float]:
    """
    Iteratively fit peaks until convergence or maximum number reached.
    """
    fitted_params = []
    current_residual = intensity.copy()
    
    for n_peaks in range(min(len(initial_peaks), max_peaks)):
        # Add next peak
        peak_energy = initial_peaks[n_peaks]
        
        # Initial parameter guess for new peak
        peak_intensity = max(np.interp(peak_energy, energy, current_residual) , 0.01)
        initial_guess = [peak_energy, peak_intensity, 0.1, 0.1]  # center, amp, sigma, gamma
        
        # Add to current parameter list
        current_params = fitted_params + [initial_guess]
        
        # Fit all peaks simultaneously
        try:
            fitted_params_flat = _fit_multiple_peaks(energy, intensity, current_params, temperature, kb, max_iter)
            
            # Reshape parameters
            fitted_params = [fitted_params_flat[i:i+4] for i in range(0, len(fitted_params_flat), 4)]
            
            # Calculate residual
            model = np.sum([_voigt_fermi_product(energy, *params, temperature, kb) for params in fitted_params], axis=0)
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
    
    return fitted_params, final_residual


def _fit_multiple_peaks(
        energy: np.ndarray, intensity: np.ndarray,
        initial_params: List[List[float]], temperature: float, 
        kb: float, max_iter: int
    ) -> List[float]:
    """Fit multiple peaks simultaneously."""
    
    # Flatten parameters
    params_flat = [param for peak_params in initial_params for param in peak_params]
    
    def objective(params):
        # Reshape parameters
        peak_params = [params[i:i+4] for i in range(0, len(params), 4)]
        
        # Calculate model
        model = np.sum([_voigt_fermi_product(energy, *p_params, temperature, kb) for p_params in peak_params], axis=0)
        
        # Return residual
        return intensity - model
    
    # Set parameter bounds
    bounds = []
    for _ in initial_params:
        bounds.extend([
            (energy.min(), energy.max()),  # center
            (0, intensity.max() * 10),     # amplitude
            (0.01, 0.3),                   # sigma
            (0.001, 0.3)                    # gamma
        ])
    
    # Perform fit
    try:
        result = optimize.least_squares(objective, params_flat, bounds=tuple(zip(*bounds)), max_nfev=max_iter, ftol=1e-12, xtol=1e-12)
        return result.x.tolist()
    except Exception as e:
        warnings.warn(f"Optimization failed: {str(e)}")
        return params_flat

def _create_output_metadata(input_metadata: Dict, fitted_params: List[List[float]],
                          temperature: float, background_type: str,
                          final_residual: float, n_peaks: int) -> Dict:
    """Create comprehensive output metadata."""
    
    output_metadata = input_metadata.copy()
    
    # Add fitting parameters
    output_metadata['fitting'] = {
        'temperature_K': temperature,
        'background_type': background_type,
        'n_peaks_fitted': n_peaks,
        'final_residual_rms': final_residual,
        'convergence_achieved': True
    }
    
    # Add peak parameters
    output_metadata['peaks'] = {}
    for i, params in enumerate(fitted_params):
        center, amplitude, sigma, gamma = params
        output_metadata['peaks'][f'peak_{i+1}'] = {
            'center_eV': center,
            'amplitude': amplitude,
            'gaussian_width_eV': sigma,
            'lorentzian_width_eV': gamma,
            'fwhm_eV': 2 * np.sqrt(2 * np.log(2)) * sigma + gamma  # Approximate FWHM
        }
    
    return output_metadata


# Example usage and test function
def test_edc_fitting():
    """Test function with synthetic data."""
    
    kb = 8.617333e-5  # Boltzmann constant in eV/K
    
    # Create synthetic EDC data
    energy = np.linspace(-3, 1, 1000)
    
    # Create synthetic spectrum with two peaks
    peak1 = 500 * voigt_profile(energy - (-1.5), 0.15, 0.1)
    peak2 = 300 * voigt_profile(energy - (-0.1), 0.12, 0.08)
    
    # Add Fermi cutoff
    fermi = 1.0 / (1.0 + np.exp(energy / (kb * 300)))
    
    # Add background and noise
    background = - 20 * (energy-1)
    background[750:] = np.zeros(250)
    noise = np.random.normal(0, 10, len(energy))
    
    intensity = (peak1 + peak2) * fermi + background + noise
    
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
    result = fit_energy_distribution_curve(
        edc_input, 
        temperature=300,
        background_type = 'shirley',
        max_peaks = 10,
        min_peak_height = 0.1,
        peak_detection_prominence = 0.1,
        convergence_threshold = 1e-6,
        max_iterations = 1000
    )
    
    return (result, edc_input)

if __name__ == "__main__":
    # Run test
    test_result, input = test_edc_fitting()
    print("Test completed successfully!")
    print(f"Number of peaks fitted: {test_result['metadata']['fitting']['n_peaks_fitted']}")
    print(f"Final residual: {test_result['metadata']['fitting']['final_residual_rms']:.6f}")
    
    plt.figure()
    plt.plot(input['data'].index, input['data'], marker='o', ms='1', ls='')
    for curve in test_result['data'].columns:
        if curve !='background':
            plt.plot(test_result['data'].index, test_result['data']['background'] + test_result['data'][curve])
        else:
            plt.plot(test_result['data'].index, test_result['data'][curve])
    plt.show()