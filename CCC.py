import math
import os
import numpy as np
import rasterio
from glob import glob

# Mathematical constant for degree to radian conversion
DEG_TO_RAD = math.pi / 180

def tansig(x):
    """
    Hyperbolic tangent activation function.
    Equivalent to the 'tansig' function in the evalscript.
    """
    return 2 / (1 + math.exp(-2 * x)) - 1

def normalize(unnormalized, min_val, max_val):
    """
    Normalizes a value to the range [-1, 1].
    """
    return 2 * (unnormalized - min_val) / (max_val - min_val) - 1

def denormalize(normalized, min_val, max_val):
    """
    Denormalizes a value from the range [-1, 1] back to its original scale.
    """
    return 0.5 * (normalized + 1) * (max_val - min_val) + min_val

# --- LAI Neural Network Functions ---

def neuron1LAI(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm, viewZen_norm, sunZen_norm, relAzim_norm):
    sum_val = (
        + 4.96238030555279
        - 0.023406878966470 * b03_norm
        + 0.921655164636366 * b04_norm
        + 0.135576544080099 * b05_norm
        - 1.938331472397950 * b06_norm
        - 3.342495816122680 * b07_norm
        + 0.902277648009576 * b8a_norm
        + 0.205363538258614 * b11_norm
        - 0.040607844721716 * b12_norm
        - 0.083196409727092 * viewZen_norm
        + 0.260029270773809 * sunZen_norm
        + 0.284761567218845 * relAzim_norm
    )
    return tansig(sum_val)

def neuron2LAI(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm, viewZen_norm, sunZen_norm, relAzim_norm):
    sum_val = (
        + 1.416008443981500
        - 0.132555480856684 * b03_norm
        - 0.139574837333540 * b04_norm
        - 1.014606016898920 * b05_norm
        - 1.330890038649270 * b06_norm
        + 0.031730624503341 * b07_norm
        - 1.433583541317050 * b8a_norm
        - 0.959637898574699 * b11_norm
        + 1.133115706551000 * b12_norm
        + 0.216603876541632 * viewZen_norm
        + 0.410652303762839 * sunZen_norm
        + 0.064760155543506 * relAzim_norm
    )
    return tansig(sum_val)

def neuron3LAI(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm, viewZen_norm, sunZen_norm, relAzim_norm):
    sum_val = (
        + 1.075897047213310
        + 0.086015977724868 * b03_norm
        + 0.616648776881434 * b04_norm
        + 0.678003876446556 * b05_norm
        + 0.141102398644968 * b06_norm
        - 0.096682206883546 * b07_norm
        - 1.128832638862200 * b8a_norm
        + 0.302189102741375 * b11_norm
        + 0.434494937299725 * b12_norm
        - 0.021903699490589 * viewZen_norm
        - 0.228492476802263 * sunZen_norm
        - 0.039460537589826 * relAzim_norm
    )
    return tansig(sum_val)

def neuron4LAI(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm, viewZen_norm, sunZen_norm, relAzim_norm):
    sum_val = (
        + 1.533988264655420
        - 0.109366593670404 * b03_norm
        - 0.071046262972729 * b04_norm
        + 0.064582411478320 * b05_norm
        + 2.906325236823160 * b06_norm
        - 0.673873108979163 * b07_norm
        - 3.838051868280840 * b8a_norm
        + 1.695979344531530 * b11_norm
        + 0.046950296081713 * b12_norm
        - 0.049709652688365 * viewZen_norm
        + 0.021829545430994 * sunZen_norm
        + 0.057483827104091 * relAzim_norm
    )
    return tansig(sum_val)

def neuron5LAI(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm, viewZen_norm, sunZen_norm, relAzim_norm):
    sum_val = (
        + 3.024115930757230
        - 0.089939416159969 * b03_norm
        + 0.175395483106147 * b04_norm
        - 0.081847329172620 * b05_norm
        + 2.219895367487790 * b06_norm
        + 1.713873975136850 * b07_norm
        + 0.713069186099534 * b8a_norm
        + 0.138970813499201 * b11_norm
        - 0.060771761518025 * b12_norm
        + 0.124263341255473 * viewZen_norm
        + 0.210086140404351 * sunZen_norm
        - 0.183878138700341 * relAzim_norm
    )
    return tansig(sum_val)

def layer2LAI(neuron1, neuron2, neuron3, neuron4, neuron5):
    sum_val = (
        + 1.096963107077220
        - 1.500135489728730 * neuron1
        - 0.096283269121503 * neuron2
        - 0.194935930577094 * neuron3
        - 0.352305895755591 * neuron4
        + 0.075107415847473 * neuron5
    )
    return sum_val

# --- Cab Neural Network Functions ---

def neuron1Cab(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm, viewZen_norm, sunZen_norm, relAzim_norm):
    sum_val = (
        + 4.242299670155190
        + 0.400396555256580 * b03_norm
        + 0.607936279259404 * b04_norm
        + 0.137468650780226 * b05_norm
        - 2.955866573461640 * b06_norm
        - 3.186746687729570 * b07_norm
        + 2.206800751246430 * b8a_norm
        - 0.313784336139636 * b11_norm
        + 0.256063547510639 * b12_norm
        - 0.071613219805105 * viewZen_norm
        + 0.510113504210111 * sunZen_norm
        + 0.142813982138661 * relAzim_norm
    )
    return tansig(sum_val)

def neuron2Cab(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm, viewZen_norm, sunZen_norm, relAzim_norm):
    sum_val = (
        - 0.259569088225796
        - 0.250781102414872 * b03_norm
        + 0.439086302920381 * b04_norm
        - 1.160590937522300 * b05_norm
        - 1.861935250269610 * b06_norm
        + 0.981359868451638 * b07_norm
        + 1.634230834254840 * b8a_norm
        - 0.872527934645577 * b11_norm
        + 0.448240475035072 * b12_norm
        + 0.037078083501217 * viewZen_norm
        + 0.030044189670404 * sunZen_norm
        + 0.005956686619403 * relAzim_norm
    )
    return tansig(sum_val)

def neuron3Cab(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm, viewZen_norm, sunZen_norm, relAzim_norm):
    sum_val = (
        + 3.130392627338360
        + 0.552080132568747 * b03_norm
        - 0.502919673166901 * b04_norm
        + 6.105041924966230 * b05_norm
        - 1.294386119140800 * b06_norm
        - 1.059956388352800 * b07_norm
        - 1.394092902418820 * b8a_norm
        + 0.324752732710706 * b11_norm
        - 1.758871822827680 * b12_norm
        - 0.036663679860328 * viewZen_norm
        - 0.183105291400739 * sunZen_norm
        - 0.038145312117381 * relAzim_norm
    )
    return tansig(sum_val)

def neuron4Cab(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm, viewZen_norm, sunZen_norm, relAzim_norm):
    sum_val = (
        + 0.774423577181620
        + 0.211591184882422 * b03_norm
        - 0.248788896074327 * b04_norm
        + 0.887151598039092 * b05_norm
        + 1.143675895571410 * b06_norm
        - 0.753968830338323 * b07_norm
        - 1.185456953076760 * b8a_norm
        + 0.541897860471577 * b11_norm
        - 0.252685834607768 * b12_norm
        - 0.023414901078143 * viewZen_norm
        - 0.046022503549557 * sunZen_norm
        - 0.006570284080657 * relAzim_norm
    )
    return tansig(sum_val)

def neuron5Cab(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm, viewZen_norm, sunZen_norm, relAzim_norm):
    sum_val = (
        + 2.584276648534610
        + 0.254790234231378 * b03_norm
        - 0.724968611431065 * b04_norm
        + 0.731872806026834 * b05_norm
        + 2.303453821021270 * b06_norm
        - 0.849907966921912 * b07_norm
        - 6.425315500537270 * b8a_norm
        + 2.238844558459030 * b11_norm
        - 0.199937574297990 * b12_norm
        + 0.097303331714567 * viewZen_norm
        + 0.334528254938326 * sunZen_norm
        + 0.113075306591838 * relAzim_norm
    )
    return tansig(sum_val)

def layer2Cab(neuron1, neuron2, neuron3, neuron4, neuron5):
    sum_val = (
        + 0.463426463933822
        - 0.352760040599190 * neuron1
        - 0.603407399151276 * neuron2
        + 0.135099379384275 * neuron3
        - 1.735673123851930 * neuron4
        - 0.147546813318256 * neuron5
    )
    return sum_val

def apply_model(bands):
    """
    Applies the neural network model to calculate LAI, Cab, and then CCC for an entire image.

    Args:
        bands (list of np.array): A list containing numpy arrays for each band:
                                  [B03, B04, B05, B06, B07, B8A, B11, B12,
                                   viewZenithMean, viewAzimuthMean, sunZenithAngles, sunAzimuthAngles]

    Returns:
        np.array: A 2D numpy array representing the calculated CCC values.
    """
    b03, b04, b05, b06, b07, b8a, b11, b12, viewZen, viewAzim, sunZen, sunAzim = bands

    # Ensure all input arrays have the same shape
    shape = b03.shape

    # Initialize output arrays
    lai_arr = np.zeros(shape, dtype=np.float32)
    cab_arr = np.zeros(shape, dtype=np.float32)
    ccc_arr = np.zeros(shape, dtype=np.float32)

    # Process pixel by pixel
    for r in range(shape[0]):
        for c in range(shape[1]):
            # Create a sample dictionary for the current pixel
            sample = {
                "B03": b03[r, c],
                "B04": b04[r, c],
                "B05": b05[r, c],
                "B06": b06[r, c],
                "B07": b07[r, c],
                "B8A": b8a[r, c],
                "B11": b11[r, c],
                "B12": b12[r, c],
                "viewZenithMean": viewZen[r, c],
                "viewAzimuthMean": viewAzim[r, c],
                "sunZenithAngles": sunZen[r, c],
                "sunAzimuthAngles": sunAzim[r, c]
            }

            # --- Normalization ---
            b03_norm = normalize(sample["B03"], 0, 0.253061520471542)
            b04_norm = normalize(sample["B04"], 0, 0.290393577911328)
            b05_norm = normalize(sample["B05"], 0, 0.305398915248555)
            b06_norm = normalize(sample["B06"], 0.006637972542253, 0.608900395797889)
            b07_norm = normalize(sample["B07"], 0.013972727018939, 0.753827384322927)
            b8a_norm = normalize(sample["B8A"], 0.026690138082061, 0.782011770669178)
            b11_norm = normalize(sample["B11"], 0.016388074192258, 0.493761397883092)
            b12_norm = normalize(sample["B12"], 0, 0.493025984460231)
            viewZen_norm = normalize(math.cos(sample["viewZenithMean"] * DEG_TO_RAD), 0.918595400582046, 1)
            sunZen_norm  = normalize(math.cos(sample["sunZenithAngles"] * DEG_TO_RAD), 0.342022871159208, 0.936206429175402)
            relAzim_norm = math.cos((sample["sunAzimuthAngles"] - sample["viewAzimuthMean"]) * DEG_TO_RAD)

            # --- LAI Calculation ---
            n1L = neuron1LAI(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm, viewZen_norm, sunZen_norm, relAzim_norm)
            n2L = neuron2LAI(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm, viewZen_norm, sunZen_norm, relAzim_norm)
            n3L = neuron3LAI(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm, viewZen_norm, sunZen_norm, relAzim_norm)
            n4L = neuron4LAI(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm, viewZen_norm, sunZen_norm, relAzim_norm)
            n5L = neuron5LAI(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm, viewZen_norm, sunZen_norm, relAzim_norm)
            
            l2L = layer2LAI(n1L, n2L, n3L, n4L, n5L)
            lai_val = denormalize(l2L, 0.000319182538301, 14.4675094548151)
            lai_arr[r, c] = lai_val
            
            # --- Cab Calculation ---
            n1C = neuron1Cab(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm, viewZen_norm, sunZen_norm, relAzim_norm)
            n2C = neuron2Cab(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm, viewZen_norm, sunZen_norm, relAzim_norm)
            n3C = neuron3Cab(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm, viewZen_norm, sunZen_norm, relAzim_norm)
            n4C = neuron4Cab(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm, viewZen_norm, sunZen_norm, relAzim_norm)
            n5C = neuron5Cab(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm, viewZen_norm, sunZen_norm, relAzim_norm)

            l2C = layer2Cab(n1C, n2C, n3C, n4C, n5C)
            cab_val = denormalize(l2C, 0.007426692959872, 873.908222110306)
            cab_arr[r, c] = cab_val

            # --- Final CCC Calculation ---
            ccc_arr[r, c] = (lai_val * cab_val) / 900
            
    return ccc_arr

def evaluate_pixel(sample):
    """
    Calculates Canopy Chlorophyll Content (CCC) for a single pixel sample.

    Args:
        sample (dict): A dictionary containing the input band values and angles.
                       Keys: "B03", "B04", "B05", "B06", "B07", "B8A", "B11", "B12",
                             "viewZenithMean", "viewAzimuthMean", "sunZenithAngles",
                             "sunAzimuthAngles".

    Returns:
        dict: A dictionary with the calculated CCC value, scaled by 1/900.
              e.g., {'default': [0.75]}
    """
    # --- Normalization ---
    b03_norm = normalize(sample["B03"], 0, 0.253061520471542)
    b04_norm = normalize(sample["B04"], 0, 0.290393577911328)
    b05_norm = normalize(sample["B05"], 0, 0.305398915248555)
    b06_norm = normalize(sample["B06"], 0.006637972542253, 0.608900395797889)
    b07_norm = normalize(sample["B07"], 0.013972727018939, 0.753827384322927)
    b8a_norm = normalize(sample["B8A"], 0.026690138082061, 0.782011770669178)
    b11_norm = normalize(sample["B11"], 0.016388074192258, 0.493761397883092)
    b12_norm = normalize(sample["B12"], 0, 0.493025984460231)
    viewZen_norm = normalize(math.cos(sample["viewZenithMean"] * DEG_TO_RAD), 0.918595400582046, 1)
    sunZen_norm  = normalize(math.cos(sample["sunZenithAngles"] * DEG_TO_RAD), 0.342022871159208, 0.936206429175402)
    relAzim_norm = math.cos((sample["sunAzimuthAngles"] - sample["viewAzimuthMean"]) * DEG_TO_RAD)

    # --- LAI Calculation ---
    n1L = neuron1LAI(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm, viewZen_norm, sunZen_norm, relAzim_norm)
    n2L = neuron2LAI(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm, viewZen_norm, sunZen_norm, relAzim_norm)
    n3L = neuron3LAI(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm, viewZen_norm, sunZen_norm, relAzim_norm)
    n4L = neuron4LAI(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm, viewZen_norm, sunZen_norm, relAzim_norm)
    n5L = neuron5LAI(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm, viewZen_norm, sunZen_norm, relAzim_norm)
    
    l2L = layer2LAI(n1L, n2L, n3L, n4L, n5L)
    lai = denormalize(l2L, 0.000319182538301, 14.4675094548151)
    
    # --- Cab Calculation ---
    n1C = neuron1Cab(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm, viewZen_norm, sunZen_norm, relAzim_norm)
    n2C = neuron2Cab(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm, viewZen_norm, sunZen_norm, relAzim_norm)
    n3C = neuron3Cab(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm, viewZen_norm, sunZen_norm, relAzim_norm)
    n4C = neuron4Cab(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm, viewZen_norm, sunZen_norm, relAzim_norm)
    n5C = neuron5Cab(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm, viewZen_norm, sunZen_norm, relAzim_norm)

    l2C = layer2Cab(n1C, n2C, n3C, n4C, n5C)
    cab = denormalize(l2C, 0.007426692959872, 873.908222110306)

    # --- Final CCC Calculation ---
    ccc = lai * cab
    
    return {'default': [ccc / 900]}

# --- Main Script Execution ---
if __name__ == '__main__':
    # Example input data for a single pixel
    example_sample = {
        "B03": 0.05, "B04": 0.08, "B05": 0.12, "B06": 0.3, "B07": 0.35, "B8A": 0.38,
        "B11": 0.25, "B12": 0.15, "viewZenithMean": 5.0, "viewAzimuthMean": 180.0,
        "sunZenithAngles": 30.0, "sunAzimuthAngles": 150.0
    }

    # Calculate the result for the example pixel
    result = evaluate_pixel(example_sample)

    print("Python script execution example:")
    print(f"Input sample data: {example_sample}")
    print(f"Calculated result: {result}")
    
    print("\n--- Processing TIFF files ---")
    # CHANGE THIS to your folder path with .tif files
    input_folder = r"D:\ABRUZZO\SENT2\2024"  # CHANGE THIS to your folder path with .tif files
    tif_files = glob(os.path.join(input_folder,"*.tif"))

    for input_path in tif_files:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_ccc_output{ext}" # Changed output name for clarity

        print(f"Processing {input_path} -> {output_path}")

        try:
            with rasterio.open(input_path) as src:
                # Bands to read, ensuring correct order for apply_model:
                # B03, B04, B05, B06, B07, B8A, B11, B12, viewZenithMean, viewAzimuthMean, sunZenithAngles, sunAzimuthAngles
                # Make sure these band indices correspond to your TIFF file's band order.
                # Assuming 1-based indexing from the problem description's bands_to_use
                bands_to_read_indices = [3, 4, 5, 6, 7, 9, 11, 12, 16, 15, 14, 13]
                
                # Read all necessary bands into a list of numpy arrays
                bands_data = [src.read(i).astype(np.float32) for i in bands_to_read_indices]
                
                profile = src.profile.copy()
                profile.update(
                    count=1,          # Output will have one band (CCC)
                    dtype='float32',  # Data type for CCC
                    compress='lzw'    # Optional: compression for smaller files
                )

                # Now, call apply_model with the actual band data
                # apply_model is designed to handle the entire image (numpy arrays)
                ccc_output_array = apply_model(bands_data)

            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(ccc_output_array, 1) # Write the calculated CCC array to the first band
            print(f"Successfully wrote {output_path}")

        except Exception as e:
            print(f"Error processing {input_path}: {e}")
