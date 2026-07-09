# -*- coding: utf-8 -*-
"""
Purpose: Download Sentinel-2 images at 10m resolution in EPSG:32632
Author: chiaramansi
"""

import os
import numpy as np
import geopandas as gpd
import rasterio
import time
from rasterio.transform import from_bounds
from datetime import datetime
from shapely.geometry import box
from sentinelhub import (
    SHConfig, DataCollection, SentinelHubRequest,
    BBox, bbox_to_dimensions, CRS, MimeType,
    SentinelHubCatalog, Geometry
)

def split_geometry_in_two(gdf):
    """Split a single polygon geometry in two vertical tiles."""
    bounds = gdf.total_bounds  # minx, miny, maxx, maxy
    mid_x = (bounds[0] + bounds[2]) / 2

    # Create two bounding boxes
    box1 = box(bounds[0], bounds[1], mid_x, bounds[3])
    box2 = box(mid_x, bounds[1], bounds[2], bounds[3])

    return [box1, box2]

def main():
    # 1. CONFIGURE SENTINEL HUB ACCESS
    config = SHConfig()
    config.sh_client_id = "YOUR CLIENT ID"
    config.sh_client_secret = "YOUR PASSWORD"
    config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    config.sh_base_url = "https://sh.dataspace.copernicus.eu"
    config.save("cdse")

    # 2. PARAMETERS
    resolution_m = 10  # Updated to 10m
    time_interval = ("2022-05-01", "2022-09-30")  # Change the time period
    max_cloud_cover = 34
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)

    # 3. LOAD AND SPLIT GEOJSON
    gdf = gpd.read_file("YOUR AOI in Geojson Format").to_crs(CRS.UTM_33N.pyproj_crs())  # Change crs accordingly
    tile_geoms = split_geometry_in_two(gdf)

    # 4. SETUP CATALOG
    s2_collection = DataCollection.SENTINEL2_L2A.define_from(
        name="s2l2a", service_url=config.sh_base_url
    )
    catalog = SentinelHubCatalog(config=config)

    # 5. LOOP OVER TILES
    for tile_idx, tile_geom in enumerate(tile_geoms):
        mask_geometry = Geometry(tile_geom, crs=CRS.UTM_33N)
        bbox = BBox(tile_geom.bounds, crs=CRS.UTM_33N)
        width, height = bbox_to_dimensions(bbox, resolution=resolution_m)

        if height > 2500:
            scale = 2500 / height
            width = int(width * scale)
            height = 2500

        print(f"\n🧩 Processing Tile {tile_idx+1} - Size: {width}x{height}")

        # SEARCH CATALOG
        search_iterator = catalog.search(
            s2_collection,
            geometry=mask_geometry,
            time=time_interval,
            filter=f"eo:cloud_cover <= {max_cloud_cover}",
            fields={"include": ["id", "properties.datetime", "properties.eo:cloud_cover"]}
        )

        # PROCESS DATES
        for item in search_iterator:
            date_str = item["properties"]["datetime"].split("T")[0]
            cloud_cover = item["properties"]["eo:cloud_cover"]
            print(f"\n📅 {date_str} (cloud cover: {cloud_cover}%)")

            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"S2_{date_str}_TILE{tile_idx+1}_{timestamp}.tif")

            if os.path.exists(output_path):
                print(f"⚠️ File {output_path} exists. Skipping.")
                continue

            # Evalscript
            evalscript = """
            //VERSION=3
            function setup() {
              return {
                input: [{
                  bands: ["B01","B02","B03","B04","B05","B06","B07","B08","B8A","B09","B11","B12",
                          "sunAzimuthAngles", "sunZenithAngles", "viewAzimuthMean", "viewZenithMean"]
                }],
                output: { bands: 16, sampleType: "FLOAT32" }
              };
            }

            function evaluatePixel(sample) {
              return [
                sample.B01, sample.B02, sample.B03, sample.B04,
                sample.B05, sample.B06, sample.B07, sample.B08,
                sample.B8A, sample.B09, sample.B11, sample.B12,
                sample.sunAzimuthAngles, sample.sunZenithAngles,
                sample.viewAzimuthMean, sample.viewZenithMean
              ];
            }
            """

            try:
                request = SentinelHubRequest(
                    evalscript=evalscript,
                    input_data=[SentinelHubRequest.input_data(
                        data_collection=s2_collection,
                        time_interval=(date_str, date_str)
                    )],
                    responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
                    bbox=bbox,
                    geometry=mask_geometry,
                    size=(width, height),
                    config=config,
                )

                image_data = request.get_data()[0]
                print(f"✅ Retrieved image: {image_data.shape}")

                h, w, bands = image_data.shape
                transform = from_bounds(*bbox, width=w, height=h)

                with rasterio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    height=h,
                    width=w,
                    count=bands,
                    dtype=image_data.dtype,
                    crs='EPSG:32633',  # Correct CRS (32632 32N, 32633 33N)
                    transform=transform
                ) as dst:
                    for i in range(bands):
                        dst.write(image_data[:, :, i], i + 1)

                print(f"💾 Saved to {output_path}")
                time.sleep(34)

            except Exception as e:
                print(f"❌ Error processing {date_str}: {e}")

if __name__ == "__main__":
    main()
