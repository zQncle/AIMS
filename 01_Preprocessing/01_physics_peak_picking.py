"""
Step 1: Physics-Informed Peak Picking & Temporal Filtering
----------------------------------------------------------
This script processes raw .mzML files to extract high-fidelity metabolic peaks.
It applies:
1. Dynamic peak picking based on instrument resolution (R=70,000).
2. FWHM (Full Width at Half Maximum) constraints to filter noise.
3. Temporal filtering to exclude unstable scans (e.g., retention time < 0.5 min).

Output:
    Processed data saved as .pkl files containing filtered peaks (m/z, area, RT).
"""

import os
import glob
import numpy as np
import pickle
from pyteomics import mzml
from scipy.signal import find_peaks, peak_widths
from tqdm import tqdm

# ================= Configuration =================
# Path to raw mzML files
DATA_DIR = "/mnt/vos-9tc8cgur/PythonProgram/DeepRMSD-Vina_Optimization-master/lung_cancer/mzML_0112/selected"
# Output directory for processed pickle files
OUTPUT_DIR = "./processed_peaks_step1"

# Peak Picking Parameters
PEAK_PROMINENCE = 48
PEAK_HEIGHT = 100

# Instrument Resolution Constraints (Orbitrap Q-Exactive)
INSTRUMENT_RESOLUTION = 70000 
FWHM_TOLERANCE_MIN = 0.8  # Lower bound multiplier for theoretical FWHM
FWHM_TOLERANCE_MAX = 1.2  # Upper bound multiplier for theoretical FWHM

# Temporal Filtering (Unit: Minute)
MIN_RETENTION_TIME = 0.5  # Exclude scans before 0.5 min (solvent front/unstable)
# =============================================

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def parse_label(filename):
    """
    Derive label from filename.
    Returns:
        0 for Healthy control
        1 for Lung Cancer patient
        -1 for unknown/exclude
    """
    name_lower = filename.lower()
    if 'health' in name_lower:
        return 0 
    elif 'lung' in name_lower:
        return 1 
    else:
        return -1 

def get_retention_time_in_min(spectrum):
    """
    Safely extract retention time and convert to minutes.
    Handles different units ('second' or 'minute') from pyteomics metadata.
    """
    scan_list = spectrum.get('scanList')
    rt = None
    
    # Try getting RT from scanList params
    if scan_list and 'scan' in scan_list:
        scan_params = scan_list['scan'][0]
        rt = scan_params.get('scan start time')
    
    # Fallback to general retention time attribute
    if rt is None:
        rt = spectrum.get('retention time')
    
    # Handle UnitFloat objects from pyteomics
    if hasattr(rt, 'magnitude'): 
        # Check if unit is seconds and convert to minutes
        if rt.unit_info and 'second' in rt.unit_info.lower():
            return rt.magnitude / 60.0
        return rt.magnitude # Default assume minutes if unit is minute or unknown

    # If simple float, return as is (assuming minutes)
    return float(rt) if rt is not None else None

def process_spectrum(mzs, intensities):
    """
    Core function for dynamic peak picking with resolution constraints.
    """
    # 1. Basic Peak Finding
    peaks_indices, properties = find_peaks(intensities, height=PEAK_HEIGHT, prominence=PEAK_PROMINENCE)
    
    if len(peaks_indices) == 0:
        return np.array([]), np.array([])

    # 2. Calculate FWHM (Full Width at Half Maximum)
    widths, width_heights, left_ips, right_ips = peak_widths(intensities, peaks_indices, rel_height=0.5)

    valid_mzs = []
    valid_areas = []

    left_bases = properties["left_bases"].astype(int)
    right_bases = properties["right_bases"].astype(int)
    mz_len = len(mzs)

    for i in range(len(peaks_indices)):
        peak_idx = peaks_indices[i]
        current_mz = mzs[peak_idx]

        # Boundary check
        if not (0 <= left_ips[i] < mz_len and 0 <= right_ips[i] < mz_len):
            continue

        # Calculate Measured FWHM (in Da)
        # Interpolate exact m/z at half-maximum points
        mz_left_fwhm = np.interp(left_ips[i], np.arange(mz_len), mzs)
        mz_right_fwhm = np.interp(right_ips[i], np.arange(mz_len), mzs)
        measured_fwhm_da = mz_right_fwhm - mz_left_fwhm

        # 3. Physics-Informed Resolution Filter
        if measured_fwhm_da > 0:
            # Theoretical FWHM = m/z / Resolution
            theoretical_fwhm = current_mz / INSTRUMENT_RESOLUTION
            min_allowed = theoretical_fwhm * FWHM_TOLERANCE_MIN
            max_allowed = theoretical_fwhm * FWHM_TOLERANCE_MAX

            # Check if measured peak width matches theoretical physics
            if not (min_allowed <= measured_fwhm_da <= max_allowed):
                continue
        else:
            continue

        # 4. Integration (Area Calculation)
        l_idx = max(0, left_bases[i])
        r_idx = min(mz_len, right_bases[i] + 1)
        
        peak_mz_slice = mzs[l_idx:r_idx]
        peak_int_slice = intensities[l_idx:r_idx]
        
        if len(peak_mz_slice) > 1:
            area = np.trapz(peak_int_slice, peak_mz_slice)
        else:
            area = intensities[peak_idx]

        valid_mzs.append(current_mz)
        valid_areas.append(area)

    return np.array(valid_mzs), np.array(valid_areas)

def main():
    mzml_files = glob.glob(os.path.join(DATA_DIR, "*.mzML"))
    print(f"Found {len(mzml_files)} .mzML files in {DATA_DIR}")
    
    for file_path in mzml_files:
        filename = os.path.basename(file_path)
        label = parse_label(filename)
        
        if label == -1:
            continue
        
        # Check for existing output to skip processing
        save_name = filename.replace('.mzML', '.pkl')
        save_path = os.path.join(OUTPUT_DIR, save_name)
        if os.path.exists(save_path):
            print(f"Skipping existing file: {filename}")
            continue

        print(f"Processing: {filename} | Label: {'Lung Cancer' if label==1 else 'Healthy'}")
        
        file_data = [] # Container for valid scans in one file

        try:
            with mzml.read(file_path) as reader:
                # Use tqdm for progress bar
                for i, spectrum in enumerate(tqdm(reader, desc=f"Scanning {filename[:15]}...", leave=False)):
                    
                    # 1. Get and Check RT
                    rt_min = get_retention_time_in_min(spectrum)
                    
                    if rt_min is None:
                        continue
                    # Temporal Filtering: Remove unstable early scans
                    if rt_min < MIN_RETENTION_TIME:
                        continue 

                    # 2. Get Data Arrays
                    mzs = spectrum['m/z array']
                    ints = spectrum['intensity array']
                    
                    # 3. Process Spectrum (Peak Picking & Filter)
                    peak_mzs, peak_areas = process_spectrum(mzs, ints)
                    
                    if len(peak_mzs) > 0:
                        # Store as float32 to save memory
                        file_data.append({
                            'scan_idx': i,
                            'rt': float(rt_min), 
                            'mzs': peak_mzs.astype(np.float32),
                            'areas': peak_areas.astype(np.float32)
                        })
                        
            # Save results immediately after processing one file
            if file_data:
                final_obj = {
                    'filename': filename,
                    'label': label,
                    'peaks': file_data
                }
                with open(save_path, 'wb') as f:
                    pickle.dump(final_obj, f)
            
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nAll processing complete! Results saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
