import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List
import re
import warnings



#==================#
# IMPORT FUNCTIONS #
#==================#


# Read a single Intensity vs k-E, Igor-exported, tab-separated .txt file
#  FROM  .txt 
#    TO  { md:{}, d:DF(k,E) }

def read_kE_igor_txt(
        file_path: Union[str, Path]
    ) -> Dict[str, Any]:
    """
    Read a single Intensity vs k-E, Igor-exported, tab-separated .txt file.
    
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



#======================#
# VALIDATION FUNCTIONS #
#======================#


# Validate a single kE datasets
def validate_kE_dataset(
        dataset: Dict[str, Any]
    ) -> bool:
    """
    Validate that an kE dataset has the expected structure and content.
    
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



#==================#
# EXPORT FUNCTIONS #
#==================#




#=========================================#
#=====                               =====#
#=       BATCH PROCESSING VERSIONS       =#
#=====                               =====#
#=========================================#


#==================#
# IMPORT FUNCTIONS #
#==================#


# Internal function for natural sorting of files in dictionaries
def natural_sort_key(path: Path) -> List[Union[int, str]]:
    """
    Generate a natural sorting key for filenames containing numbers.
    
    This handles cases like: file1.txt, file2.txt, file10.txt, file20.txt
    (instead of alphabetical: file1.txt, file10.txt, file2.txt, file20.txt)
    """
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', path.stem)]


# Read all Intensity vs k-E, Igor-exported, tab-separated .txt files in a folder
#  FROM  folder of .txt 
#    TO  { file.stem: { md:{}, d:DF(k,E) } }

def read_all_Ek_igor_txt(
        folder_path: Optional[Union[str, Path]] = None, 
        pattern: str = "*.txt",
        natural_sort: bool = True,
        verbose: bool = True
    ) -> Dict[str, Dict[str, Any]]:
    """
    Read all Intensity vs k-E, Igor-exported, tab-separated .txt files in a folder.
    
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
    
    # Sort the file paths in natural numerical order, if required
    if natural_sort:
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
            Ek_dict[file_path.stem] = read_kE_igor_txt(file_path)
        except Exception as e:
            failed_files.append((file_path.name, str(e)))
            if verbose:
                print(f"Warning: Failed to read {file_path.name}: {e}")
    
    if failed_files and verbose:
        print(f"\nSummary: Successfully read {len(Ek_dict)} files, failed on {len(failed_files)} files")
        for filename, error in failed_files:
            print(f"  - {filename}: {error}")
    elif verbose and not failed_files:
        print("Successfully read all files")
    
    if not Ek_dict:
        raise ValueError("No files could be successfully read")
    
    return Ek_dict



#======================#
# VALIDATION FUNCTIONS #
#======================#


# Summarize a batch of kE datasets
#  FROM  { file.stem: { md:{}, d:DF(k,E) } }
#    TO  DF(name,md)

def summarize_kE_dict(
        kE_dict: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
    """
    Create a summary DataFrame of all datasets in the collection.
    
    Parameters:
    -----------
    kE_dict : dict
        Dictionary of kE datasets
        
    Returns:
    --------
    pd.DataFrame
        Summary table with key properties of each dataset
    """
    summary_data = []
    
    for name, dataset in kE_dict.items():
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
