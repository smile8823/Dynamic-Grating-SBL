import numpy as np
import csv
import os
from typing import Tuple


def read_csv_data(csv_filename: str = "10s_1.csv") -> Tuple[np.ndarray, np.ndarray]:
    """
    Read Fiber Bragg Grating (FBG) data in CSV format
    
    CSV Format Specification:
    - First column: Timestamp
    - First row (excluding timestamp column): X-axis coordinates (Wavelength values 1527.0000 to 1568.0000)
    - From second row onwards (excluding timestamp column): Data frames (one frame per row)
    
    Args:
        csv_filename: CSV filename, defaults to "10s_1.csv"
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (wavelengths, data_matrix)
            - wavelengths: 1D array containing wavelength coordinates
            - data_matrix: 2D array with shape (num_wavelengths, num_frames), containing all frame data
    """
    # Construct data file path (supports relative and absolute paths)
    if os.path.isabs(csv_filename):
        # If absolute path, use directly
        csv_path = csv_filename
    else:
        # If relative path, try multiple possible paths
        possible_paths = [
            os.path.join("data", csv_filename),  # From project root's data subdirectory
            os.path.join("..", "data", csv_filename),  # From parent directory's data subdirectory (relative to src)
            csv_filename  # Use provided filename directly
        ]
        
        csv_path = None
        for path in possible_paths:
            if os.path.exists(path):
                csv_path = path
                break
        
        if csv_path is None:
            # If not found, use the first one for error message
            csv_path = possible_paths[0]
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    
    print(f"Reading data file: {csv_path}")
    
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        
        # Read the first row to get x-axis coordinates (wavelengths)
        first_row = next(reader)
        # Skip the first column (timestamp), starting from the second column are wavelength values
        wavelengths_str = first_row[1:]
        wavelengths = np.array([float(w) for w in wavelengths_str])
        num_wavelengths = len(wavelengths)
        
        # Read all data rows
        data_rows = []
        for row in reader:
            # Skip the first column (timestamp), read data part
            data_row = [float(value) for value in row[1:]]
            data_rows.append(data_row)
        
        # Convert to numpy array and transpose, shape becomes (num_wavelengths, num_frames)
        data_matrix = np.array(data_rows).T
        
        # Verify data shape
        if data_matrix.shape[0] != num_wavelengths:
            raise ValueError(f"Data shape mismatch: Number of wavelengths is {num_wavelengths}, but data columns is {data_matrix.shape[0]}")
        
        num_frames = data_matrix.shape[1]
        
        print(f"Data reading completed:")
        print(f"  - Wavelength range: {wavelengths[0]:.4f} - {wavelengths[-1]:.4f} nm")
        print(f"  - Wavelength points: {num_wavelengths}")
        print(f"  - Number of frames: {num_frames}")
        print(f"  - Data matrix shape: {data_matrix.shape}")
        
        return wavelengths, data_matrix


def get_fbg_training_data(num_frames: int = None, start_frame: int = 0, filename: str = None) -> np.ndarray:
    """
    Get FBG data for training
    
    Args:
        num_frames: Number of training frames, use all frames if None
        start_frame: Start frame index
        filename: Data filename (required)
        
    Returns:
        np.ndarray: Training data matrix with shape (num_wavelengths, num_frames)
    """
    if filename is None:
        raise ValueError("Must provide filename for get_fbg_training_data")
        
    wavelengths, data_matrix = read_csv_data(filename)
    
    if num_frames is None:
        return data_matrix
    else:
        end_frame = start_frame + num_frames
        if end_frame > data_matrix.shape[1]:
            end_frame = data_matrix.shape[1]
            print(f"Warning: Requested frames exceed range, using maximum available frames {end_frame - start_frame}")
        
        return data_matrix[:, start_frame:end_frame]


def get_fbg_sampled_data(num_samples: int = 10, start_frame: int = 0, filename: str = None) -> np.ndarray:
    """
    Get grouped sampled FBG data
    
    Args:
        num_samples: Number of samples (number of groups)
        start_frame: Start frame index
        filename: Data filename (required)
        
    Returns:
        np.ndarray: Sampled data matrix with shape (num_wavelengths, num_samples)
    """
    if filename is None:
        raise ValueError("Must provide filename for get_fbg_sampled_data")
        
    wavelengths, data_matrix = read_csv_data(filename)
    
    total_frames = data_matrix.shape[1]
    available_frames = total_frames - start_frame
    
    if available_frames <= 0:
        raise ValueError(f"Start frame {start_frame} exceeds data range {total_frames}")
    
    if num_samples > available_frames:
        print(f"Warning: Requested samples {num_samples} exceed available frames {available_frames}, using maximum available frames")
        num_samples = available_frames
    
    # Calculate group size
    group_size = available_frames // num_samples
    
    if group_size == 0:
        # If group size is 0, directly return the first num_samples frames
        return data_matrix[:, start_frame:start_frame + num_samples]
    
    # Calculate center frame index for each group
    sampled_indices = []
    for i in range(num_samples):
        group_start = start_frame + i * group_size
        group_center = group_start + group_size // 2
        sampled_indices.append(group_center)
    
    print(f"Grouped sampling information:")
    print(f"  - Total frames: {total_frames}")
    print(f"  - Available frames: {available_frames}")
    print(f"  - Number of groups: {num_samples}")
    print(f"  - Group size: {group_size}")
    print(f"  - Sampled frame indices: {sampled_indices}")
    
    # Extract sampled frames
    sampled_data = data_matrix[:, sampled_indices]
    
    return sampled_data


def normalize_data(data: np.ndarray, method: str = 'minmax') -> Tuple[np.ndarray, dict]:
    """
    Data normalization
    
    Args:
        data: Input data matrix
        method: Normalization method ('minmax' or 'zscore')
        
    Returns:
        Tuple[np.ndarray, dict]: (Normalized data, Normalization parameters)
    """
    if method == 'minmax':
        # Min-Max normalization to [0, 1]
        min_val = np.min(data)
        max_val = np.max(data)
        normalized = (data - min_val) / (max_val - min_val + 1e-10)
        params = {'method': 'minmax', 'min': min_val, 'max': max_val}
    elif method == 'zscore':
        # Z-score standardization
        mean_val = np.mean(data)
        std_val = np.std(data)
        normalized = (data - mean_val) / (std_val + 1e-10)
        params = {'method': 'zscore', 'mean': mean_val, 'std': std_val}
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    
    return normalized, params


def denormalize_data(normalized_data: np.ndarray, params: dict) -> np.ndarray:
    """
    Denormalize data
    
    Args:
        normalized_data: Normalized data
        params: Normalization parameters
        
    Returns:
        np.ndarray: Denormalized data
    """
    if params['method'] == 'minmax':
        return normalized_data * (params['max'] - params['min']) + params['min']
    elif params['method'] == 'zscore':
        return normalized_data * params['std'] + params['mean']
    else:
        raise ValueError(f"Unsupported normalization method: {params['method']}")


if __name__ == "__main__":
    # Test data reading functionality
    wavelengths, data = read_csv_data()
    print(f"Wavelength range: {wavelengths[0]:.4f} - {wavelengths[-1]:.4f} nm")
    print(f"Data shape: {data.shape}")
    
    # Test getting training data
    training_data = get_fbg_training_data(num_frames=100, filename="10s_1.csv")
    print(f"Training data shape: {training_data.shape}")
    
    # Test data normalization
    normalized_data, norm_params = normalize_data(training_data)
    print(f"Normalized data range: [{np.min(normalized_data):.6f}, {np.max(normalized_data):.6f}]")