import os
import numpy as np
import rasterio
from glob import glob

def normalize(x, min_val, max_val):
    return 2 * (x - min_val) / (max_val - min_val) - 1

def denormalize(x, min_val, max_val):
    return 0.5 * (x + 1) * (max_val - min_val) + min_val

def tansig(x):
    return 2 / (1 + np.exp(-2 * x)) - 1

def apply_model(bands):
    b03, b04, b05, b06, b07, b8a, b11, b12, viewZen, viewAzim, sunZen, sunAzim = bands

    degToRad = np.pi / 180
    relAzim = np.cos((sunAzim - viewAzim) * degToRad)
    
    inputs = [
        normalize(b03, 0, 0.253061520471542),
        normalize(b04, 0, 0.290393577911328),
        normalize(b05, 0, 0.305398915248555),
        normalize(b06, 0.006637972542253, 0.608900395797889),
        normalize(b07, 0.013972727018939, 0.753827384322927),
        normalize(b8a, 0.026690138082061, 0.782011770669178),
        normalize(b11, 0.016388074192258, 0.493761397883092),
        normalize(b12, 0, 0.493025984460231),
        normalize(np.cos(viewZen * degToRad), 0.918595400582046, 1),
        normalize(np.cos(sunZen * degToRad), 0.342022871159208, 0.936206429175402),
        relAzim
    ]

    def neuron(w, b):
        return tansig(b + np.sum([w[i] * inputs[i] for i in range(len(w))], axis=0))

    neurons = [
        ([-0.0234, 0.9217, 0.1356, -1.9383, -3.3425, 0.9023, 0.2054, -0.0406, -0.0832, 0.2600, 0.2848], 4.9624),
        ([-0.1326, -0.1396, -1.0146, -1.3309, 0.0317, -1.4336, -0.9596, 1.1331, 0.2166, 0.4107, 0.0648], 1.4160),
        ([0.0860, 0.6166, 0.6780, 0.1411, -0.0967, -1.1288, 0.3022, 0.4345, -0.0219, -0.2285, -0.0395], 1.0759),
        ([-0.1094, -0.0710, 0.0646, 2.9063, -0.6739, -3.8381, 1.6960, 0.0470, -0.0497, 0.0218, 0.0575], 1.5340),
        ([-0.0899, 0.1754, -0.0818, 2.2199, 1.7139, 0.7131, 0.1390, -0.0608, 0.1243, 0.2101, -0.1839], 3.0241)
    ]

    neuron_outputs = [neuron(w, b) for w, b in neurons]

    l2 = (
        1.096963107077220
        - 1.500135489728730 * neuron_outputs[0]
        - 0.096283269121503 * neuron_outputs[1]
        - 0.194935930577094 * neuron_outputs[2]
        - 0.352305895755591 * neuron_outputs[3]
        + 0.075107415847473 * neuron_outputs[4]
    )

    lai = denormalize(l2, 0.000319182538301, 14.4675094548151)
    return lai / 3

input_folder = r"D:\ABRUZZO\SENT2\2024"   # CHANGE THIS to your folder path with .tif files
output_folder = os.path.join(input_folder, "LAI")
tif_files = glob(os.path.join(input_folder,"*.tif"))


for input_path in tif_files:
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_lai_output{ext}"

    print(f"Processing {input_path} -> {output_path}")

    with rasterio.open(input_path) as src:
        bands_to_use = [3, 4, 5, 6, 7, 9, 11, 12, 16, 15, 14, 13]
        bands = [src.read(i).astype(np.float32) for i in bands_to_use]
        profile = src.profile.copy()
        profile.update(count=1, dtype='float32')

        lai = apply_model(bands)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(lai.astype(np.float32), 1)
