import os
import glob
import numpy as np
import rasterio

# --- Helper Function (only calculate_ndmi is needed for single-band NDMI output) ---

def calculate_ndmi(b8a, b11):
    """
    Calculates the Normalized Difference Moisture Index (NDMI) using vectorized operations.
    NDMI = (B8A - B11) / (B8A + B11)
    
    Args:
        b8a (np.ndarray): The B8A band as a NumPy array.
        b11 (np.ndarray): The B11 band as a NumPy array.
        
    Returns:
        np.ndarray: The calculated NDMI array. Values range from -1 to 1, float32 dtype.
    """
    b8a_float = b8a.astype(np.float32)
    b11_float = b11.astype(np.float32)
    
    denominator = b8a_float + b11_float
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # Use np.where to handle division by zero or invalid operations.
        # Where denominator is zero, NDMI is set to 0.0, or np.nan if you prefer to retain nodata explicitly.
        ndmi = np.where(denominator != 0, (b8a_float - b11_float) / denominator, 0.0)
    
    # Clip NDMI values to the theoretical range [-1.0, 1.0]
    ndmi = np.clip(ndmi, -1.0, 1.0)
    
    return ndmi

# --- Main Processing Function (modified for single-band NDMI output) ---

def process_ndmi_image(input_path, output_path, band_map):
    """
    Reads a multi-band GeoTIFF, calculates NDMI, and saves the result as a new
    single-band GeoTIFF with float32 data type.
    
    Args:
        input_path (str): Path to the input multi-band GeoTIFF file.
        output_path (str): Path to save the output single-band NDMI GeoTIFF file.
        band_map (dict): Dictionary mapping model input names to GeoTIFF band indices.
    """
    print(f"Processing {input_path} -> {output_path}")
    try:
        with rasterio.open(input_path) as src:
            profile = src.profile.copy()
            
            # Read the required bands
            b8a_band = src.read(band_map['B8A'])
            b11_band = src.read(band_map['B11'])
            
            # --- Perform NDMI Calculation ---
            ndmi_array = calculate_ndmi(b8a_band, b11_band)
            
            print(f"  NDMI Calculated. Min: {np.min(ndmi_array):.4f}, Max: {np.max(ndmi_array):.4f}")

        # --- Update Profile for Single-Band Float32 Output ---
        profile.update(
            count=1,            # Output will have only one band (NDMI)
            dtype=np.float32,   # Data type will be float32 for NDMI values
            nodata=0.0,         # You might want to define a nodata value for NDMI,
                                # 0.0 is common if calculated as such for no-data areas.
                                # Or choose np.nan if you used that in calculate_ndmi.
            compress='lzw'      # Optional: compression for smaller files
        )

        # Write the NDMI array to a new GeoTIFF
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(ndmi_array, 1) # Write the NDMI array as the first (and only) band
            
        print(f"Successfully wrote single-band NDMI to {output_path}")

    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for more context

# --- Main Execution ---
if __name__ == "__main__":
    # !!! IMPORTANT !!!
    # CHANGE THIS to the folder path containing your multi-band .tif files
    input_folder = r"D:\ABRUZZO\SENT2\2024"   # CHANGE THIS to your folder path with .tif files
    
    # !!! IMPORTANT !!!
    # This dictionary maps the required inputs to the band number in your
    # multi-band GeoTIFF file. YOU MUST VERIFY AND ADJUST THESE INDICES.
    # The indices are 1-based (i.e., Band 1 is index 1).
    # For Sentinel-2 L2A, typical bands for NDMI:
    # B8A (Near-Infrared, typically band 8)
    # B11 (Short-wave Infrared 1, typically band 11)
    # Note: If your product has different band order, adjust accordingly.
    # For Sentinel-2, B8A is usually 864nm and B11 is 1610nm.
    # The `dataMask` is not needed for the single-band NDMI output.
    BAND_MAP = {
        "B8A": 8,   # VERIFY THIS for your specific Sentinel-2 product
        "B11": 11,  # VERIFY THIS for your specific Sentinel-2 product
    }

    # The color ramp is no longer directly used for the output file,
    # as we're saving the raw NDMI values.
    # MOISTURE_RAMPS = [...] 

    # --- Start Processing ---
    if not os.path.isdir(input_folder):
        print(f"Error: Input folder not found at '{input_folder}'")
        print("Please change the 'input_folder' variable to your data directory.")
    else:
        tif_files = glob.glob(os.path.join(input_folder, "*.tif"))
        
        if not tif_files:
            print(f"No .tif files found in '{input_folder}'.")
        
        for input_path in tif_files:
            base, ext = os.path.splitext(input_path)
            # Create a descriptive output filename for the single-band NDMI
            output_path = f"{base}_NDMI{ext}" 
            
            # Call the new processing function
            process_ndmi_image(input_path, output_path, BAND_MAP)
