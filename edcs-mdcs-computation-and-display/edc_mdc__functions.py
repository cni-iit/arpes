import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import rcParams
from matplotlib.collections import LineCollection
from pathlib import Path
from joblib import Memory
from typing import Dict, Any, Optional, Union, Tuple, List
import re
import warnings

memory = Memory(location='cachedir', verbose=0)


#===================#
# READING FUNCTIONS #
#===================#

# Read a single Energy vs k, Igor-exported, tab-separated .txt file
#  .txt -> md-d(E,k) dict
def read_Ek_igor_txt(
        file_path: Union[str, Path]
    ) -> Dict[str, Any]:
    """
    Read a single Energy vs kx, Igor-exported, tab-separated .txt file.
    
    Parameters:
    -----------
    file_path : str or Path
        Path to the Igor-exported .txt file
    
    Returns:
    --------
    dict
        Dictionary containing 'metadata' and 'data' keys
    
    Raises:
    -------
    FileNotFoundError
        If the file doesn't exist
    ValueError
        If the file format is invalid or data cannot be parsed
    """
    # Begin by converting a possible string to a Path object
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    if not file_path.suffix.lower() == '.txt':
        warnings.warn(f"Expected .txt file, got {file_path.suffix}")
    
    try:
        # Import the file, transpose to have wavenumbers as columns, energies as rows
        df = pd.read_csv(file_path, sep='\t', dtype=np.float64, header=1, index_col=0).T
        
        # Validate that we have numeric data
        if df.empty:
            raise ValueError("File contains no data")
            
        if df.isna().all().all():
            raise ValueError("File contains only NaN values")
    
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        raise ValueError(f"Failed to parse {file_path.name}: {e}")
    except Exception as e:
        raise ValueError(f"Failed to read {file_path.name}: {e}")
    
    # Ensure rows and columns are labeled by floats
    try:
        df.index = df.index.astype(np.float64)
        df.columns = df.columns.astype(np.float64)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Could not convert indices to float in {file_path.name}: {e}")
    
    # Appropriately label the DataFrame axes
    df.index.name = 'Energy (eV)'
    df.columns.name = 'Wavenumber (Å⁻¹)'
    
    # Create comprehensive metadata dictionary
    df_meta = {
        'name': file_path.stem,
        'file_path': str(file_path),
        'units': 'CPS',
        'data_shape': df.shape,
        'k_range': (df.columns.min(), df.columns.max()),
        'E_range': (df.index.min(), df.index.max()),
        'k_step': np.diff(df.columns).mean() if len(df.columns) > 1 else 0,
        'E_step': np.diff(df.index).mean() if len(df.index) > 1 else 0,
        'total_counts': df.sum().sum(),
        'max_intensity': df.max().max(),
        'description': f"Electronic dispersion from file {file_path.stem}"
    }
    
    return {'metadata': df_meta, 'data': df}

# Support function for natural sorting of files in dictionaries
def natural_sort_key(
        path: Path
    ) -> List[Union[int, str]]:
    """
    Generate a natural sorting key for filenames containing numbers.
    
    This handles cases like: file1.txt, file2.txt, file10.txt, file20.txt
    (instead of alphabetical: file1.txt, file10.txt, file2.txt, file20.txt)
    """
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', path.stem)]

# Read all Energy vs kx, Igor-exported, tab-separated .txt files in a folder
#  batch of .txt -> dict of files, each with md-d(E,k) dict
@memory.cache
def read_all_Ek_igor_txt(
        folder_path: Optional[Union[str, Path]] = None, 
        pattern: str = "*.txt",
        verbose: bool = True
    ) -> Dict[str, Dict[str, Any]]:
    """
    Read all Energy vs kx, Igor-exported, tab-separated .txt files in a folder.
    
    Parameters:
    -----------
    folder_path : str, Path, or None
        Path to folder containing .txt files. If None, uses current working directory.
    pattern : str
        Glob pattern for file matching (default: "*.txt")
    verbose : bool
        Whether to print progress messages
        
    Returns:
    --------
    dict
        Dictionary with filenames as keys and dataset dictionaries as values
        
    Raises:
    -------
    FileNotFoundError
        If the folder doesn't exist
    ValueError
        If no matching files are found
    """
    # Begin by converting folder_path to a Path object
    folder_path = Path(folder_path) if folder_path else Path.cwd()
    
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
        
    if not folder_path.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")
    
    # Find all matching files in the specified folder
    file_paths_list = list(folder_path.glob("*.txt"))
    
    if not file_paths_list:
        raise ValueError(f"No files matching '{pattern}' found in {folder_path}")
    
    # Sort the file paths in natural numerical order
    file_paths_list = sorted(file_paths_list, key=natural_sort_key)

    if verbose:
        print(f"Found {len(file_paths_list)} files matching '{pattern}' in {folder_path}")
    
    # Prepare a dictionary for storing the E-k datasets
    Ek_dict = {}
    failed_files = []
    
    # Read the files one by one and store them in the dictionary
    for file_path in file_paths_list:
        try:
            if verbose:
                print(f"Reading {file_path.name}...")
            Ek_dict[file_path.stem] = read_Ek_igor_txt(file_path)
        except Exception as e:
            failed_files.append((file_path.name, str(e)))
            if verbose:
                print(f"Warning: Failed to read {file_path.name}: {e}")
    
    if failed_files and verbose:
        print(f"\nSummary: Successfully read {len(Ek_dict)} files, failed on {len(failed_files)} files")
        for filename, error in failed_files:
            print(f"  - {filename}: {error}")
    
    if not Ek_dict:
        raise ValueError("No files could be successfully read")
    
    return Ek_dict


#======================#
# VALIDATION FUNCTIONS #
#======================#

# Support function for validating single datasets
def validate_Ek_dataset(
        dataset: Dict[str, Any]
    ) -> bool:
    """
    Validate that an Ek dataset has the expected structure and content.
    
    Parameters:
    -----------
    dataset : dict
        Dataset dictionary to validate
        
    Returns:
    --------
    bool
        True if valid, raises ValueError if invalid
    """
    required_keys = ['metadata', 'data']
    if not all(key in dataset for key in required_keys):
        raise ValueError(f"Dataset missing required keys: {required_keys}")
    
    if not isinstance(dataset['data'], pd.DataFrame):
        raise ValueError("Dataset 'data' must be a pandas DataFrame")
    
    if dataset['data'].empty:
        raise ValueError("Dataset contains empty DataFrame")
    
    metadata = dataset['metadata']
    required_meta_keys = ['name', 'units', 'k_range', 'E_range']
    missing_keys = [key for key in required_meta_keys if key not in metadata]
    if missing_keys:
        raise ValueError(f"Metadata missing required keys: {missing_keys}")
    
    return True

# Internal version
def _validate_Ek_dataset(
        dataset: Dict[str, Any]
    ) -> None:
    """Validate Ek dataset structure (internal function)."""
    required_keys = ['metadata', 'data']
    if not all(key in dataset for key in required_keys):
        raise KeyError(f"Dataset missing required keys: {required_keys}")
    
    if not isinstance(dataset['data'], pd.DataFrame):
        raise ValueError("Dataset 'data' must be a pandas DataFrame")
    
    if dataset['data'].empty:
        raise ValueError("Dataset contains empty DataFrame")

# Support function for summarizing whole dictionaries
def get_dict_summary(
        Ek_dict: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
    """
    Create a summary DataFrame of all datasets in the collection.
    
    Parameters:
    -----------
    Ek_dict : dict
        Dictionary of Ek datasets
        
    Returns:
    --------
    pd.DataFrame
        Summary table with key properties of each dataset
    """
    summary_data = []
    
    for name, dataset in Ek_dict.items():
        meta = dataset['metadata']
        data = dataset['data']
        
        summary_data.append({
            'name': name,
            'shape': meta.get('data_shape', data.shape),
            'k_range': meta['k_range'],
            'E_range': meta['E_range'],
            'k_step': meta.get('k_step', 'N/A'),
            'E_step': meta.get('E_step', 'N/A'),
            'max_intensity': meta.get('max_intensity', data.max().max()),
            'total_counts': meta.get('total_counts', data.sum().sum())
        })
    
    return pd.DataFrame(summary_data).set_index('name')


#======================#
# PROCESSING FUNCTIONS #
#======================#

#=== EDCs ===#

# Extract a single EDC from an E-k dataset, possibly interpolating the intensities
#  md-d(E,k) dict -> md-d(E) dict
def extract_edc(
        Ek_dataset: Dict[str, Any], 
        target_k: float, 
        interpolation_method: str = 'linear',
        bounds_error: bool = False,
        fill_value: Optional[float] = 0.0,
        validate_input: bool = True
    ) -> Dict[str, Any]:
    """
    Extract a single Energy Distribution Curve (EDC) from an E-k dataset.
    
    Parameters:
    -----------
    Ek_dataset : dict
        Dictionary containing 'metadata' and 'data' keys from ARPES measurement
    target_k : float
        Target wavenumber in Å⁻¹ for EDC extraction
    interpolation_method : str
        Interpolation method ('linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic')
    bounds_error : bool
        If True, raise error when target_k is outside data range
    fill_value : float or None
        Value to use for points outside data range when bounds_error=False
    validate_input : bool
        Whether to validate input dataset structure
    
    Returns:
    --------
    dict
        Dictionary containing EDC metadata and data
    
    Raises:
    -------
    ValueError
        If dataset is invalid or target_k is out of bounds (when bounds_error=True)
    KeyError
        If required keys are missing from dataset
    """
    
    if validate_input:
        _validate_Ek_dataset(Ek_dataset)
    
    data = Ek_dataset['data']
    metadata = Ek_dataset['metadata']
    
    # Load wavenumbers and energies (already validated to be numeric)
    wavenumbers = data.columns.astype(np.float64)
    energies = data.index.astype(np.float64)
    
    # Check if target_k is within data range
    k_min, k_max = wavenumbers.min(), wavenumbers.max()
    if target_k < k_min or target_k > k_max:
        warning_msg = f"target_k ({target_k:.4f}) is outside data range [{k_min:.4f}, {k_max:.4f}]"
        if bounds_error:
            raise ValueError(warning_msg)
        else:
            warnings.warn(warning_msg + f". Using fill_value={fill_value}")
    
    # Choose interpolation method based on data density and target location
    if interpolation_method == 'linear':
        # Optimized linear interpolation using numpy.interp (fastest)
        if fill_value is None:
            # Use nearest neighbor extrapolation
            edc_values = np.array([
                np.interp(target_k, wavenumbers, row.values) 
                for row in data.itertuples(index=False, name=None)
            ])
        else:
            # Use specified fill value for extrapolation
            edc_values = np.array([
                np.interp(target_k, wavenumbers, row.values, left=fill_value, right=fill_value)
                for row in data.itertuples(index=False, name=None)
            ])
    else:
        # Use scipy.interpolate for advanced interpolation methods
        from scipy.interpolate import interp1d
        edc_values = np.zeros(len(energies))
        
        for i, (_, row) in enumerate(data.iterrows()):
            try:
                f = interp1d(wavenumbers, row.values, kind=interpolation_method, bounds_error=bounds_error, fill_value=fill_value)
                edc_values[i] = f(target_k)
            except ValueError as e:
                if bounds_error:
                    raise ValueError(f"Interpolation failed at energy {energies[i]:.4f} eV: {e}")
                edc_values[i] = fill_value if fill_value is not None else 0.0
    
    # Create pandas Series for the EDC
    edc = pd.Series(edc_values, index=energies, name=f'EDC_k={target_k:.4f}')
    edc.index.name = 'Energy (eV)'
    
    # Create a dictionary with metadata
    edc_meta = {
        'name': f"{metadata['name']}_edc_k{target_k:.4f}",
        'original_dataset': metadata['name'],
        'units': metadata.get('units', 'CPS'),
        'k_target': target_k,
        'k_range_original': metadata.get('k_range', (k_min, k_max)),
        'E_range': metadata.get('E_range', (energies.min(), energies.max())),
        'interpolation_method': interpolation_method,
        'bounds_error': bounds_error,
        'fill_value': fill_value,
        'is_interpolated': not np.any(np.isclose(wavenumbers, target_k, rtol=1e-10)),
        'max_intensity': edc_values.max(),
        'total_counts': edc_values.sum(),
        'description': f"EDC from {metadata['name']} at k = {target_k:.4f} Å⁻¹"
    }
    
    return {'metadata': edc_meta, 'data': edc}

# Extract multiple EDCs from an E-k dataset, possibly interpolating the intensities
#  md-d(E,k) dict -> dict of k, each with md-d(E) dict
@memory.cache
def extract_multiple_edc(
        Ek_dataset: Dict[str, Any], 
        target_k_list: List[float],
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
    """
    Extract multiple EDCs from a single E-k dataset efficiently.
    
    Parameters:
    -----------
    Ek_dataset : dict
        Dictionary containing 'metadata' and 'data' keys from ARPES measurement
    target_k_list : list of float
        List of target wavenumbers in Å⁻¹ for EDC extraction
    **kwargs
        Additional arguments passed to extract_edc()
        
    Returns:
    --------
    dict
        Dictionary with k-values as keys and EDC datasets as values
    """
    
    edc_dict = {}
    
    # Validate input once for efficiency
    kwargs['validate_input'] = kwargs.get('validate_input', True)
    
    for i, target_k in enumerate(target_k_list):
        # Only validate first iteration
        if i > 0:
            kwargs['validate_input'] = False
            
        key = f"k_{target_k:.4f}"
        edc_dict[key] = extract_edc(Ek_dataset, target_k, **kwargs)
    
    return edc_dict

# Extract all EDCs from a file collection of E-k datasets, possibly interpolating the intensities
#  dict of files, each with md-d(E,k) dict -> 
#   A.  dict of files, each with md-d(E) dict                       (if single k was asked)
#   B1. dict of files, each with dict of k, each with md-d(E) dict  (if multiple k were asked)
#   B2. dict of files, each with md-d(E,kn) dict                    (if multiple k were asked and combine k was required)
@memory.cache
def extract_all_edc(
        Ek_dict: Dict[str, Dict[str, Any]],
        target_k: Union[float, List[float]],
        combine_results: bool = False,
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
    """
    Extract EDCs from multiple E-k datasets.
    
    Parameters:
    -----------
    Ek_dict : dict
        Dictionary of E-k datasets (from read_all_Ek_igor_txt)
    target_k : float or list of float
        Target wavenumber(s) in Å⁻¹ for EDC extraction
    combine_results : bool
        If True and target_k is a list, combine all EDCs into a single DataFrame
    **kwargs
        Additional arguments passed to extract_edc()
        
    Returns:
    --------
    dict
        Dictionary with dataset names as keys and EDC data as values
    """
    
    if not Ek_dict:
        raise ValueError("Input Ek_dict is empty")
    
    # Handle single vs multiple k-values
    target_k_list = target_k if isinstance(target_k, list) else [target_k]
    
    edc_results = {}
    failed_extractions = []
    
    # Extract EDCs from each dataset
    for dataset_name, Ek_dataset in Ek_dict.items():
        try:
            if len(target_k_list) == 1:
                # Single EDC extraction
                edc_results[dataset_name] = extract_edc(Ek_dataset, target_k_list[0], **kwargs)
            else:
                # Multiple EDC extraction
                edc_results[dataset_name] = extract_multiple_edc(Ek_dataset, target_k_list, **kwargs)
                
        except Exception as e:
            failed_extractions.append((dataset_name, str(e)))
            warnings.warn(f"Failed to extract EDC from {dataset_name}: {e}")
    
    if failed_extractions:
        print(f"Warning: Failed to extract EDCs from {len(failed_extractions)} datasets:")
        for name, error in failed_extractions:
            print(f"  - {name}: {error}")
    
    if not edc_results:
        raise ValueError("No EDCs could be extracted from any dataset")
    
    # Optionally combine results for multiple k-values
    if combine_results and len(target_k_list) > 1:
        edc_results = _combine_multiple_edcs(edc_results, target_k_list)
    
    return edc_results

# Prepare a single dataframe out of a dictionary of individual EDC (keys being either k values or files)
#  A. dict of k, each with md-d(E) dict     -> d(E,kn)
#  B. dict of files, each with md-d(E) dict -> d(E,files)
def compare_edcs(
        edc_dict: Dict[str, Dict[str, Any]], 
        normalize: bool = False,
        energy_range: Optional[Tuple[float, float]] = None
    ) -> pd.DataFrame:
    """
    Compare multiple EDCs by combining them into a single DataFrame.
    
    Parameters:
    -----------
    edc_dict : dict
        Dictionary of EDC datasets
    normalize : bool
        Whether to normalize each EDC to its maximum value
    energy_range : tuple of float, optional
        (E_min, E_max) to restrict the comparison range
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with energies as index and EDCs as columns
    """
    
    if not edc_dict:
        raise ValueError("Input edc_dict is empty")
    
    edc_data = {}
    
    for name, edc_dataset in edc_dict.items():
        edc_series = edc_dataset['data'].copy()
        
        # Apply energy range filter if specified
        if energy_range is not None:
            E_min, E_max = energy_range
            edc_series = edc_series[(edc_series.index >= E_min) & (edc_series.index <= E_max)]
        
        # Normalize if requested
        if normalize:
            max_val = edc_series.max()
            if max_val > 0:
                edc_series = edc_series / max_val
        
        edc_data[name] = edc_series
    
    # Combine into DataFrame with common energy grid
    combined_df = pd.DataFrame(edc_data)
    combined_df.index.name = 'Energy (eV)'
    
    return combined_df

# Internal version (for extract_all_edc B2 variant)
def _combine_multiple_edcs(
        edc_results: Dict[str, Dict[str, Dict[str, Any]]], 
        target_k_list: List[float]
    ) -> Dict[str, pd.DataFrame]:
    """Combine multiple EDCs into DataFrames (internal function)."""
    combined_results = {}
    
    for dataset_name, edc_dict in edc_results.items():
        edc_dataframes = {}
        
        for k_val in target_k_list:
            k_key = f"k_{k_val:.4f}"
            if k_key in edc_dict:
                edc_dataframes[f"k={k_val:.4f}"] = edc_dict[k_key]['data']
        
        if edc_dataframes:
            combined_df = pd.DataFrame(edc_dataframes)
            combined_df.index.name = 'Energy (eV)'
            
            # Create combined metadata
            combined_meta = {
                'name': f"{dataset_name}_multiple_edcs",
                'original_dataset': dataset_name,
                'k_values': target_k_list,
                'description': f"Multiple EDCs from {dataset_name}"
            }
            
            combined_results[dataset_name] = {
                'metadata': combined_meta,
                'data': combined_df
            }
    
    return combined_results

#=== BINNED EDCs ===#

# Extract a binned EDC from a dataframe
def extract_binned_edc(Ek_dataset, target_k, bin_width):
    
    # Load wavenumbers and energies
    wavenumbers = Ek_dataset['data'].columns.astype(np.float64)
    energies = Ek_dataset['data'].index.astype(np.float64)
    
    # Define the bin range
    lower_k_bound = target_k - bin_width/2
    upper_k_bound = target_k + bin_width/2
    
    # Select columns within the bin range
    selected_columns = [col for col in wavenumbers if lower_k_bound <= col <= upper_k_bound]
    
    # Compute the mean intensity across the selected wavenumbers
    binned_edc = Ek_dataset['data'][selected_columns].mean(axis=1)
    binned_edc.index = energies
    
    # Appropriately label the DataFrame energy axis
    binned_edc.index.name = r'Energy (eV)'
    
    # Create a dictionary with metadata
    binned_edc_meta = {
        'name': f"{Ek_dataset['metadata']['name']}_binned_edc",
        'units': 'Average CPS',
        'k': target_k,
        'k binning': bin_width,
        'E_range': Ek_dataset['metadata']['E_range'],
        'description': f"Binned EDC from file {Ek_dataset['metadata']['name']} around {target_k} ± {bin_width/2} " + r'$\AA^{-1}$'
    }
    
    # Build a dictionary containing data and metadata
    binned_edc_dataset = {'metadata': binned_edc_meta, 'data': binned_edc}
    
    return binned_edc_dataset

# Extract all binned EDCs from a dictionary of E-k datasets
@memory.cache
def extract_all_binned_edc(Ek_dict, target_k, bin_width):
    
    # Prepare a dictionary for storing the EDCs
    binned_edc_dict = Ek_dict.copy()
    
    # Compute the EDCs one by one and store them in the dictionary
    for _ in Ek_dict.keys():
        binned_edc_dict[_] = extract_binned_edc(Ek_dict[_], target_k, bin_width)
    
    return binned_edc_dict

#=== MDCs ===#

# Extract a single MDC from a dataframe, possibly interpolating the intensities
def extract_mdc(Ek_dataset, target_E):
    
    # Load wavenumbers and energies
    wavenumbers = Ek_dataset['data'].columns.astype(np.float64)
    energies = Ek_dataset['data'].index.astype(np.float64)
    
    # Interpolate intensity at the target energy for each wavenumber
    mdc = Ek_dataset['data'].apply(lambda col: np.interp(target_E, energies, col.values), axis=0)
    mdc.index = wavenumbers
    
    # Appropriately label the Series wavenumber axis
    mdc.index.name = r'Wavenumber ($\AA^{-1}$)'
    
    # Create a dictionary with metadata
    mdc_meta = {
        'name': f"{Ek_dataset['metadata']['name']}_mdc",
        'units': 'CPS',
        'E': target_E,
        'k_range': Ek_dataset['metadata']['k_range'],
        'description': f"MDC from file {Ek_dataset['metadata']['name']} at {target_E} eV"
    }
    
    # Build a dictionary containing data and metadata
    mdc_dataset = {'metadata': mdc_meta, 'data': mdc}
    
    return mdc_dataset

# Extract all MDCs from a dictionary of E-k datasets, possibly interpolating the intensities
@memory.cache
def extract_all_mdc(Ek_dict, target_E):
    
    # Prepare a dictionary for storing the MDCs
    mdc_dict = Ek_dict.copy()
    
    # Compute the MDCs one by one and store them in the dictionary
    for _ in Ek_dict.keys():
        mdc_dict[_] = extract_edc(Ek_dict[_], target_E)
    
    return mdc_dict

#=== BINNED MDCs ===#

# Extract a binned MDC from a dataframe
def extract_binned_mdc(Ek_dataset, target_E, bin_width):
    
    # Load wavenumbers and energies
    wavenumbers = Ek_dataset['data'].columns.astype(np.float64)
    energies = Ek_dataset['data'].index.astype(np.float64)
    
    # Define the bin range
    lower_E_bound = target_E - bin_width/2
    upper_E_bound = target_E + bin_width/2
    
    # Select rows within the bin range
    selected_rows = [row for row in energies if lower_E_bound <= row <= upper_E_bound]
    
    # Compute the mean intensity across the selected energies
    binned_mdc = Ek_dataset['data'].loc[selected_rows].mean(axis=0)
    binned_mdc.index = wavenumbers
    
    # Appropriately label the Series wavenumbers axis
    binned_mdc.index.name = r'Wavenumber ($\AA^{-1}$)'
    
    # Create a dictionary with metadata
    binned_mdc_meta = {
        'name': f"{Ek_dataset['metadata']['name']}_binned_edc",
        'units': 'Average CPS',
        'E': target_E,
        'E binning': bin_width,
        'k_range': Ek_dataset['metadata']['k_range'],
        'description': f"Binned MDC from file {Ek_dataset['metadata']['name']} around {target_E} ± {bin_width/2} eV"
    }
    
    # Build a dictionary containing data and metadata
    binned_mdc_dataset = {'metadata': binned_mdc_meta, 'data': binned_mdc}
    
    return binned_mdc_dataset

# Extract all binned MDCs from a dictionary of E-k datasets
@memory.cache
def extract_all_binned_mdc(Ek_dict, target_E, bin_width):
    
    # Prepare a dictionary for storing the MDCs
    binned_mdc_dict = Ek_dict.copy()
    
    # Compute the MDCs one by one and store them in the dictionary
    for _ in Ek_dict.keys():
        binned_mdc_dict[_] = extract_binned_mdc(Ek_dict[_], target_E, bin_width)
    
    return binned_mdc_dict


#====================#
# PLOTTING FUNCTIONS #
#====================#

#=== Dispersions ===#

# Plot a single E-k dispersion, returning the figure
def plot_dispersion(
        Ek_dataset,
        min_k = None,
        max_k = None,
        min_E = None,
        max_E = None,
        normalize = True,
        colormap = 'viridis'
    ):
    
    # Extract only the relevant interval if required
    if min_k and max_k:
        plot_k = [col for col in Ek_dataset['data'].columns if min_k <= col <= max_k]
    else:
        plot_k = Ek_dataset['data'].columns
    
    if min_E and max_E:
        plot_E = [row for row in Ek_dataset['data'].index if min_E <= row <= max_E]
    else:
        plot_E = Ek_dataset['data'].index
    
    if (min_k and max_k) or (min_E and max_E):
        plot_data = Ek_dataset['data'][plot_k].loc[plot_E]
    else:
        plot_data = Ek_dataset['data']
    
    # Normalize data if required
    if normalize:
        plot_data = (plot_data - plot_data.min().min()) / (plot_data.max().max() - plot_data.min().min())
        colorbar_label = 'Intensity (arb. u.)'
    else:
        colorbar_label = Ek_dataset['metadata']['units']
    
    # Create the coordinates for the 2D plots
    X, Y = np.meshgrid(plot_k, plot_E)
    
    # Create the figure
    fig, ax = plt.subplots()
    
    # Plot the data
    pcm = ax.pcolormesh(X, Y, plot_data, colormap=colormap, shading='gouraud')
    
    # Complement the plot with colorbar, axes labels and title
    plt.colorbar(pcm, label=colorbar_label)
    ax.set_xlabel(plot_data.columns.name)
    ax.set_ylabel(plot_data.index.name)
    ax.set_title(Ek_dataset['metadata']['description'])
    
    return fig

# Plot and save a batch of E-k dispersion, from a dictionary of datasets
def plot_and_save_all_dispersions(
        Ek_dict,
        min_k = None,
        max_k = None,
        min_E = None,
        max_E = None,
        normalize = True,
        colormap = 'viridis',
        save_path = None,
        format = 'png'
    ):
    
    # Set up output directory
    savePath = Path(save_path) if save_path else Path.cwd()
    savePath = savePath / 'dispersions plots'
    savePath.mkdir(exist_ok=True)
    
    # Loop over all E-k datasets in the dictionary
    for _ in Ek_dict.keys():
        
        # Create the figure and plot according to requirements
        fig = plot_dispersion(
            Ek_dict[_],
            min_k=min_k,
            max_k=max_k,
            min_E=min_E,
            max_E=max_E,
            normalize=normalize,
            colormap=colormap
        )
        
        # Save and close the picture
        filename = Ek_dict[_]['metadata']['name'] + '.' + format
        fig.savefig(
            savePath / filename,
            bbox_inches='tight'
        )
        plt.close(fig)

    return

#=== EDCs / MDCs ===#

# Plot a single EDC or MDC, returning the figure
def plot_dc(
        dc_dataset: Dict[str, Any],
        range: Optional[Tuple[float, float]] = None,
        normalize: bool = True,
        style: str = 'line',
        color: str = 'blue',
        linewidth: float = 1.5,
        alpha: float = 1.0,
        figsize: Tuple[float, float] = (8, 6),
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        grid: bool = True,
        return_axes: bool = False
    ) -> Union[plt.Figure, Tuple[plt.Figure, plt.Axes]]:
    """
    Plot a single EDC or MDC with enhanced customization options.
    
    Parameters:
    -----------
    dc_dataset : dict
        Dictionary containing 'metadata' and 'data' keys
    range : tuple of float, optional
        (min, max) energy/momentum range to plot
    normalize : bool
        Whether to normalize data to [0, 1] range
    style : str
        Plot style: 'line', 'scatter', 'step', 'bar'
    color : str
        Color for the plot
    linewidth : float
        Line width for line plots
    alpha : float
        Transparency (0-1)
    figsize : tuple
        Figure size (width, height)
    title, xlabel, ylabel : str, optional
        Custom labels (if None, uses defaults from data)
    grid : bool
        Whether to show grid
    return_axes : bool
        If True, return (fig, ax) tuple instead of just fig
    
    Returns:
    --------
    matplotlib.figure.Figure or tuple
        Figure object, or (Figure, Axes) if return_axes=True
    """
    
    # Validate input
    if not isinstance(dc_dataset, dict) or 'data' not in dc_dataset:
        raise ValueError("dc_dataset must be a dictionary with 'data' key")
    
    data = dc_dataset['data']
    metadata = dc_dataset.get('metadata', {})
    
    # Extract only the relevant interval if required
    if range is not None:
        min_val, max_val = range
        mask = (data.index >= min_val) & (data.index <= max_val)
        plot_data = data.loc[mask].copy()
        if plot_data.empty:
            warnings.warn(f"No data points in range [{min_val}, {max_val}]")
            plot_data = data.copy()
    else:
        plot_data = data.copy()
    
    # Normalize data if required
    if normalize:
        data_min, data_max = plot_data.min(), plot_data.max()
        if data_max > data_min:  # Avoid division by zero
            plot_data = (plot_data - data_min) / (data_max - data_min)
            default_ylabel = 'Intensity (arb. u.)'
        else:
            warnings.warn("Data has no variation, normalization skipped")
            default_ylabel = metadata.get('units', 'Intensity')
    else:
        default_ylabel = metadata.get('units', 'Intensity')
    
    # Create figure and prepare data
    fig, ax = plt.subplots(figsize=figsize)
    x_data = plot_data.index
    y_data = plot_data.values
    
    # Plot based on style
    if style == 'line':
        ax.plot(x_data, y_data, color=color, linewidth=linewidth, alpha=alpha)
    elif style == 'scatter':
        ax.scatter(x_data, y_data, color=color, alpha=alpha, s=linewidth*10)
    elif style == 'step':
        ax.step(x_data, y_data, color=color, linewidth=linewidth, alpha=alpha, where='mid')
    elif style == 'bar':
        ax.bar(x_data, y_data, color=color, alpha=alpha, width=np.diff(x_data).mean() if len(x_data) > 1 else 1)
    else:
        raise ValueError(f"Unknown style: {style}. Use 'line', 'scatter', 'step', or 'bar'")
    
    # Set labels and title
    ax.set_xlabel(xlabel or plot_data.index.name or 'Energy (eV)')
    ax.set_ylabel(ylabel or default_ylabel)
    ax.set_title(title or metadata.get('description', f"Data from {metadata.get('name', 'unknown')}"))
    
    # Optional grid
    if grid:
        ax.grid(True, alpha=0.3)
    
    # Tight layout
    fig.tight_layout()
    
    return (fig, ax) if return_axes else fig

# Plot multiple EDCs or MDCs, returning the figure
def plot_dc_comparison(
        dc_dict: Dict[str, Dict[str, Any]],
        range: Optional[Tuple[float, float]] = None,
        normalize: bool = True,
        normalize_individually: bool = False,
        colors: Optional[List[str]] = None,
        linestyles: Optional[List[str]] = None,
        linewidth: float = 1.5,
        alpha: float = 0.8,
        figsize: Tuple[float, float] = (10, 6),
        legend: bool = True,
        grid: bool = True,
        title: Optional[str] = None
    ) -> plt.Figure:
    """
    Compare multiple DC curves in a single plot.
    
    Parameters:
    -----------
    dc_dict : dict
        Dictionary of DC datasets
    range : tuple, optional
        Range to plot  
    normalize : bool
        Whether to normalize data globally
    normalize_individually : bool
        Whether to normalize each curve individually (overrides normalize)
    colors : list, optional
        List of colors for each curve
    linestyles : list, optional
        List of line styles for each curve
    linewidth : float
        Line width
    alpha : float
        Line transparency
    figsize : tuple
        Figure size
    legend : bool
        Whether to show legend
    grid : bool
        Whether to show grid
    title : str, optional
        Plot title
    
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object
    """
    
    if not dc_dict:
        raise ValueError("dc_dict is empty")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data
    plot_data = {}
    for key, dc_dataset in dc_dict.items():
        data = dc_dataset['data'].copy()
        if range is not None:
            min_E, max_E = range
            mask = (data.index >= min_E) & (data.index <= max_E)
            data = data.loc[mask]
        plot_data[key] = data
    
    # Handle normalization
    if normalize_individually:
        for key in plot_data:
            data = plot_data[key]
            data_min, data_max = data.min(), data.max()
            if data_max > data_min:
                plot_data[key] = (data - data_min) / (data_max - data_min)
        y_label = 'Intensity (arb. u.)'
    elif normalize:
        all_values = np.concatenate([plot_data[key].values for key in plot_data])
        global_min, global_max = all_values.min(), all_values.max()
        if global_max > global_min:
            for key in plot_data:
                plot_data[key] = (plot_data[key] - global_min) / (global_max - global_min)
        y_label = 'Intensity (arb. u.)'
    else:
        first_key = list(dc_dict.keys())[0]
        y_label = dc_dict[first_key]['metadata'].get('units', 'Intensity')
    
    # Set up colors and line styles
    n_curves = len(plot_data)
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, min(n_curves, 10)))
        if n_curves > 10:
            colors = plt.cm.viridis(np.linspace(0, 1, n_curves))
    
    if linestyles is None:
        linestyles = ['-'] * n_curves
    
    # Plot each curve
    for i, (key, data) in enumerate(plot_data.items()):
        color = colors[i % len(colors)] if isinstance(colors, list) else colors
        linestyle = linestyles[i % len(linestyles)] if isinstance(linestyles, list) else linestyles[0]
        
        ax.plot(data.index, data.values, c=color, ls=linestyle, lw=linewidth, alpha=alpha, label=key)
    
    # Formatting
    ax.set_xlabel(list(plot_data.values())[0].index.name or 'Energy (eV)')
    ax.set_ylabel(y_label)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Comparison of {n_curves} curves')
    
    if legend and n_curves <= 20:  # Avoid cluttered legends
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    if grid:
        ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    
    return fig

# Plot all EDCs from a dictionary of Series
def plot_and_save_all_edc(
        dc_dict: Dict[str, Dict[str, Any]],
        energy_range: Optional[Tuple[float, float]] = None,
        normalize: bool = True,
        save_path: Optional[Union[str, Path]] = None,
        file_format: str = 'png',
        dpi: int = 300,
        figsize: Tuple[float, float] = (8, 6),
        style: str = 'line',
        color: str = 'blue',
        create_subfolder: bool = True,
        verbose: bool = True,
        **plot_kwargs
    ) -> List[str]:
    """
    Plot and save all EDC curves from a dictionary.
    
    Parameters:
    -----------
    dc_dict : dict
        Dictionary of DC datasets
    energy_range : tuple, optional
        Energy range to plot
    normalize : bool
        Whether to normalize data
    save_path : str or Path, optional
        Directory to save plots (default: current directory)
    file_format : str
        File format ('png', 'pdf', 'svg', 'eps')
    dpi : int
        Resolution for raster formats
    figsize : tuple
        Figure size
    style : str
        Plot style
    color : str
        Plot color
    create_subfolder : bool
        Whether to create a subfolder for plots
    verbose : bool
        Whether to print progress
    **plot_kwargs
        Additional arguments passed to plot_dc()
    
    Returns:
    --------
    list
        List of saved file paths
    """
    
    if not dc_dict:
        raise ValueError("dc_dict is empty")
    
    # Set up output directory
    save_path = Path(save_path) if save_path else Path.cwd()
    if create_subfolder:
        save_path = save_path / 'dc_plots'
        save_path.mkdir(exist_ok=True)
    
    saved_files = []
    failed_plots = []
    
    # Loop over all edc datasets in the dictionary
    for name, dc_dataset in dc_dict.items():
        try:
            if verbose:
                print(f"Plotting {name}...")
            
            # Create plot
            fig = plot_dc(
                dc_dataset,
                range=energy_range,
                normalize=normalize,
                style=style,
                color=color,
                figsize=figsize,
                **plot_kwargs
            )
            
            # Generate filename
            safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"{safe_name}.{file_format}"
            filepath = save_path / filename
            
            # Save figure
            fig.savefig(filepath, bbox_inches='tight', dpi=dpi, format=file_format)
            plt.close(fig)
            
            saved_files.append(str(filepath))
        
        except Exception as e:
            failed_plots.append((name, str(e)))
            if verbose:
                print(f"Failed to plot {name}: {e}")
    
    if verbose:
        print(f"\nSaved {len(saved_files)} plots to {save_path}")
        if failed_plots:
            print(f"Failed to create {len(failed_plots)} plots:")
            for name, error in failed_plots:
                print(f"  - {name}: {error}")
    
    return saved_files

# Plot all MDCs from a dictionary of Series
def plot_and_save_all_mdc(
        mdc_dict,
        min_k = None,
        max_k = None,
        normalize = True,
        save_path = None,
        format = 'png'
    ):
    
    # Set up output directory
    savePath = Path(save_path) if save_path else Path.cwd()
    savePath = savePath / 'mdc plots'
    savePath.mkdir(exist_ok=True)
    
    # Loop over all edc datasets in the dictionary
    for _ in mdc_dict.keys():
        
        # Create the figure and plot according to requirements
        fig = plot_dc(
            mdc_dict[_],
            min=min_k,
            max=max_k,
            normalize=normalize,
        )
        
        # Save and close the picture
        filename = mdc_dict[_]['metadata']['name'] + '.' + format
        fig.savefig(
            savePath / filename,
            bbox_inches='tight'
        )
        plt.close(fig)
    
    return

# Plot EDCs cascade from a dictionary of Series
def plot_edc_cascade(
        edc_dict: Dict[str, Dict[str, Any]],
        energy_range: Optional[Tuple[float, float]] = None,
        normalize: bool = True,
        y_offset_factor: float = 0.5,
        figsize: Tuple[float, float] = (10, 8),
        colormap: Optional[str] = None,
        line_color: str = 'black',
        linewidth: float = 1.5,
        alpha: float = 0.8,
        sort_key: Optional[callable] = None,
        reverse_order: bool = False,
        show_colorbar: bool = True,
        title: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        filename: Optional[str] = None,
        file_format: str = 'png',
        dpi: int = 300,
        return_figure: bool = False) -> Optional[plt.Figure]:
    """
    Create a cascade plot of multiple EDC curves with enhanced customization.
    
    Parameters:
    -----------
    edc_dict : dict
        Dictionary of EDC datasets
    energy_range : tuple, optional
        Energy range to plot
    normalize : bool
        Whether to normalize data globally
    y_offset_factor : float
        Vertical offset between curves as fraction of data range
    figsize : tuple
        Figure size
    colormap : str, optional
        Colormap name for gradient coloring (e.g., 'viridis', 'plasma')
    line_color : str
        Single color for all lines (used if colormap is None)
    linewidth : float
        Line width
    alpha : float
        Line transparency
    sort_key : callable, optional
        Function to sort the dictionary keys
    reverse_order : bool
        Whether to reverse the plotting order
    show_colorbar : bool
        Whether to show colorbar (only relevant if colormap is used)
    title : str, optional
        Custom plot title
    save_path : str or Path, optional
        Directory to save plot
    filename : str, optional
        Custom filename (without extension)
    file_format : str
        File format for saving
    dpi : int
        Resolution for raster formats
    return_figure : bool
        Whether to return the figure object
        
    Returns:
    --------
    matplotlib.figure.Figure or None
        Figure object if return_figure=True
    """
    
    if not edc_dict:
        raise ValueError("edc_dict is empty")
    
    # Sort the dictionary if requested
    sorted_keys = list(edc_dict.keys())
    if sort_key:
        sorted_keys = sorted(sorted_keys, key=sort_key)
    if reverse_order:
        sorted_keys = sorted_keys[::-1]
    
    # Filter energy range if specified
    plot_data = {}
    for key in sorted_keys:
        data = edc_dict[key]['data'].copy()
        if energy_range is not None:
            min_E, max_E = energy_range
            mask = (data.index >= min_E) & (data.index <= max_E)
            data = data.loc[mask]
        plot_data[key] = data
    
    # Get energy axis (assuming all datasets have the same energy grid)
    first_key = sorted_keys[0]
    energy_axis = plot_data[first_key].index
    
    # Calculate global normalization if requested
    if normalize:
        all_values = np.concatenate([plot_data[key].values for key in sorted_keys])
        global_min, global_max = all_values.min(), all_values.max()
        
        if global_max > global_min:
            for key in sorted_keys:
                plot_data[key] = (plot_data[key] - global_min) / (global_max - global_min)
            y_label = 'Intensity (arb. u.)'
            data_range = 1.0
        else:
            y_label = edc_dict[first_key]['metadata'].get('units', 'Intensity')
            data_range = global_max
    else:
        y_label = edc_dict[first_key]['metadata'].get('units', 'Intensity')
        data_range = max([plot_data[key].max() for key in sorted_keys])
    
    # Calculate y-offset
    y_offset = y_offset_factor * data_range
    n_curves = len(sorted_keys)
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot curves
    if colormap:
        # Use colormap for gradient coloring
        cmap = plt.get_cmap(colormap)
        colors = cmap(np.linspace(0, 1, n_curves))
        
        for i, key in enumerate(sorted_keys):
            y_data = plot_data[key].values + i * y_offset
            
            # Create line collection for gradient coloring
            points = np.array([energy_axis, y_data]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            lc = LineCollection(segments, colors=[colors[i]], linewidths=linewidth, alpha=alpha)
            ax.add_collection(lc)
        
        # Add the colorbar
        if show_colorbar:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(global_min, global_max))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label(y_label)
    
    else:
        # Use single color for all lines
        for i, key in enumerate(sorted_keys):
            y_data = plot_data[key].values + i * y_offset
            ax.plot(energy_axis, y_data, color=line_color, linewidth=linewidth, alpha=alpha)
    
    # Set axes limits
    ax.set_xlim(energy_axis.min(), energy_axis.max())
    y_min = -data_range * 0.1
    y_max = data_range * (1 + y_offset_factor * n_curves + 0.1)
    ax.set_ylim(y_min, y_max)
    
    # Set labels
    ax.set_xlabel(energy_axis.name or 'Energy (eV)')
    ax.set_ylabel(y_label)
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        # Try to create intelligent title from metadata
        first_meta = edc_dict[first_key]['metadata']
        if 'k' in first_meta:
            default_title = f"Cascade plot at k = {first_meta['k']:.4f} Å⁻¹"
        else:
            default_title = f"Cascade plot ({n_curves} curves)"
        ax.set_title(default_title)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Tight layout
    fig.tight_layout()
    
    # Save if requested
    if save_path:
        save_path = Path(save_path)
        if not save_path.exists():
            save_path.mkdir(parents=True, exist_ok=True)
        
        if filename:
            full_filename = f"{filename}.{file_format}"
        else:
            first_name = sorted_keys[0]
            last_name = sorted_keys[-1]
            full_filename = f"cascade_{first_name}_to_{last_name}.{file_format}"
        
        filepath = save_path / full_filename
        fig.savefig(filepath, bbox_inches='tight', dpi=dpi, format=file_format)
        print(f"Cascade plot saved to: {filepath}")
        
        if not return_figure:
            plt.close(fig)
            return None
    
    return fig if return_figure else None

# Plot MDCs cascade from a dictionary of Series
def plot_mdc_cascade(
        mdc_dict,
        min_k = None,
        max_k = None,
        normalize = True,
        y_offset_factor = 0.5,
        figsize = (8, 8),
        colors = '0',
        save_path = None,
        format = 'png'
    ):
    
    # Filter wavevector range, and data accordingly
    if min_k and max_k:
        k_mask = (mdc_dict[next(iter(mdc_dict))]['data'].index >= min_k) & (mdc_dict[next(iter(mdc_dict))]['data'].index <= max_k)
        plot_k = mdc_dict[next(iter(mdc_dict))]['data'].index[k_mask]
        plot_data = mdc_dict.copy()
        for _ in plot_data.keys():
            plot_data[_]['data'] = mdc_dict[_]['data'].loc[k_mask]
    else:
        plot_k = mdc_dict[next(iter(mdc_dict))]['data'].index
        plot_data = mdc_dict
    
    # Find global maximum and minimum of intensity across the dictionary
    maxima, minima = [], []
    for _ in plot_data.keys():
        maxima.append(plot_data[_]['data'].max())
        minima.append(plot_data[_]['data'].min())
    global_max, global_min = max(maxima), min(minima)
    
    # Normalize data (to the global maximum) if requested
    if normalize:
        for _ in plot_data.keys():
            plot_data[_]['data'] = (plot_data[_]['data']-global_min) / (global_max-global_min)
            y_label = 'Intensity (arb. u.)'
        global_max, global_min = 1, 0
    else:
        y_label = mdc_dict[next(iter(mdc_dict))]['metadata']['units']
    
    # Calculate y_offset
    y_offset = y_offset_factor * global_max
    
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each line with y-graded colors, if requested
    if colors != '0':
        for i, _ in enumerate(plot_data.keys()):
            offset_mdc = plot_data[_]['data'] + i * y_offset
            points = np.array([plot_k, offset_mdc]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=colors, norm=plt.Normalize(global_min, global_max))
            lc.set_array(plot_data[_]['data'])
            lc.set_linewidth(1.5)
            ax.add_collection(lc)
        
        # Add the colorbar
        if colors != '0':
            sm = plt.cm.ScalarMappable(cmap=colors, norm=plt.Normalize(global_min, global_max))
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label(y_label)
    
    # Else, plot just with black lines
    else:
        for i, _ in enumerate(plot_data.keys()):
            offset_mdc = plot_data[_]['data'] + i * y_offset
            plt.plot(plot_k, offset_mdc, color='0', linewidth=1.5)
    
    # Set axes limits
    ax.set_xlim(plot_k.min(), plot_k.max())
    ax.set_ylim(global_min-global_max/2, global_max*(1+y_offset_factor*len(mdc_dict.keys())))
    
    # Set axes labels
    plt.xlabel(mdc_dict[next(iter(mdc_dict))]['data'].index.name)
    plt.ylabel(y_label)
    
    # Set plot title
    if 'E binning' in mdc_dict[next(iter(mdc_dict))]['metadata'].keys():
        title = f"MDCs around {mdc_dict[next(iter(mdc_dict))]['metadata']['E']} ± {mdc_dict[next(iter(mdc_dict))]['metadata']['E binning']/2} eV"
    else:
        title = f"MDCs at {mdc_dict[next(iter(mdc_dict))]['metadata']['E']} eV"
    plt.title(title)
    
    # Set up output directory
    savePath = Path(save_path) if save_path else Path.cwd()
    savePath = savePath / 'mdc cascade plots'
    savePath.mkdir(exist_ok=True)
    
    # Save and close the picture
    filename = 'from ' + next(iter(mdc_dict)) + ' to ' + next(reversed(mdc_dict)) + '.' + format
    fig.savefig(
        savePath / filename,
        bbox_inches='tight'
    )
    plt.close(fig)
    
    return
