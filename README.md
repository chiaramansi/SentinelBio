# SentinelBio

`SentinelBio` is an open-source repository providing areproducible workflow to automate the acquisition of Sentinel-2 satellite imagery and compute key forest biophysical parameters (CCC,LAI, NDMI). Designed to support ecological research these scripts have been used for evaluating dynamics in forest ecosystems linked to Land Surface Temperature (LST).
This toolkit bridges the gap between raw satellite data acquisition and advanced ecological modeling.

## Key Features & Workflow

* **AOI Definition:** To easily define and extract the geographic coordinates of your Area of Interest, you can use the [BBox Finder](https://bboxfinder.com/#0.000000,0.000000,0.000000,0.000000) tool to generate the bounding box coordinates needed for the scripts.
* **Optimized Imagery Acquisition:** The core download scripts leverage the Sentinel Hub API to quickly fetch Sentinel-2 imagery. To bypass potential local memory errors, the pipeline automatically splits large AOIs into smaller, manageable tiles (Sentinel2.py script), but if yours is a small AOI you can run 2Sentinel2_NoTiles.py
* **Solar Geometry Integration:** The download script expands upon official documentation by incorporating solar angle geometries, which are strictly required for the accurate downstream calculation of biophysical variables.
* **Biophysical Parameter Estimation:** The repository includes dedicated Python scripts (`.py`) to compute essential ecological indicators, specifically Leaf Area Index (**LAI**), Canopy Chlorophyll Content (**CCC**), and Normalized Difference Moisture Index (**NDMI**). 
* **SNAP-Compliant Conversions:** To ensure scientific consistency with standard ESA software, linear conversion equations are needed to scale the raw $CCC$ and $LAI$ indices outputs into real-world biophysical values identical to those obtained via SNAP. The scaling equations implemented directly within our scripts are:
  $$\text{Real CCC} = 219.349 \times \text{CCC}_{\text{script}} + 43.89$$
  $$\text{Real LAI} = 2.936 \times \text{LAI}_{\text{script}} - 0.027$$


## Data Access & Platforms

Data previewing can be managed through the [Copernicus Data Space Ecosystem (CDSE)](https://dataspace.copernicus.eu/), which offers two primary workflows depending on your processing needs:

### 1. Browser Download (.SAFE Format)
The CDSE browser allows you to inspect and download Sentinel-2 images in the official **.SAFE format**. This format is essentially a `.zip` archive that can be opened directly—without unzipping—using ESA’s Sentinel Toolbox (**SNAP**). However, it's not recommended for downloading a large number of images.

### 2. Cloud Processing via JupyterLab
Within the Copernicus platform, you can access an integrated **JupyterLab** environment, which provides an API for launching scripts and automated processing. 
* **Filtering & Output:** Users can apply filters such as acquisition date and cloud cover to refine their search. The output can be customized as a multiband image composed of only RGB bands, or the full set of Sentinel-2 multispectral bands. 
* **Format Note:** The final product from JupyterLab is delivered as a multiband **.tif** file. While highly convenient for many quick GIS applications, keep in mind that it lacks some of the structural advantages and extensive metadata found in the original `.SAFE` format.

## Requirements & Usage
The provided scripts are fully operational and designed to run within isolated **Conda environments** or directly inside local **Jupyter Notebooks**. 

> <i class="fas fa-exemption"></i> **Important Authentication & Credit Notice:**
> To access the data via API, you must create an **OAuth client** through the Copernicus Dashboard. Remember to save your OAuth Client ID and Secret Password immediately, as the password will not be shown again. 
> 
> Be cautious when running automated workflows: each request consumes processing credits, which are limited on a monthly basis. Keep an eye on your personal dashboard to monitor your remaining monthly credit allowance.

You need a .geojson polygon inside the same directory where your script is host.

If you are not familiar with working in a Conda environment, you can check this [step-by-step guide](Conda.md).

## Acknowledgments
The foundational download framework is built upon the official [SentinelHub Tutorial](https://sentinelhub-py.readthedocs.io/en/latest/sh.html). 
Other specialized custom scripts can be explored via the [SentinelHub Evalscript Repository](https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel/sentinel-2/).
