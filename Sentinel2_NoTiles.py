# -*- coding: utf-8 -*-
"""
Purpose: Download Sentinel-2 images at 10m resolution in EPSG:32632
Author: chiaramansi
"""

import os
import geopandas as gpd
import rasterio
import time
from rasterio.transform import from_bounds
from collections import defaultdict
from sentinelhub import (
    SHConfig, DataCollection, SentinelHubRequest,
    BBox, bbox_to_dimensions, CRS, MimeType,
    SentinelHubCatalog, Geometry
)

def main():

    # 1️⃣ CONFIGURE SENTINEL HUB ACCESS
    config = SHConfig()
    config.sh_client_id = "YOUR CLIENT ID"
    config.sh_client_secret = "YOUR PASSWORD"
    config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    config.sh_base_url = "https://sh.dataspace.copernicus.eu"

    # 2️⃣ PARAMETERS
    resolution_m = 10
    time_interval = ("2025-05-01", "2025-09-30") # Change the time period
    max_cloud_cover = 34
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)

    # 3️⃣ LOAD AOI (NO SPLITTING)
    gdf = gpd.read_file("YOUR AOI in Geojson Format").to_crs(CRS.UTM_33N.pyproj_crs())   # Change crs accordingly
    full_geom = gdf.unary_union

    geometry = Geometry(full_geom, crs=CRS.UTM_33N)
    bbox = BBox(full_geom.bounds, crs=CRS.UTM_33N)
    width, height = bbox_to_dimensions(bbox, resolution=resolution_m)

    # 4️⃣ SETUP COLLECTION & CATALOG
    s2_collection = DataCollection.SENTINEL2_L2A.define_from(
        name="s2l2a", service_url=config.sh_base_url
    )
    catalog = SentinelHubCatalog(config=config)

    print("🔎 Searching catalog...")

    search_iterator = catalog.search(
        s2_collection,
        geometry=geometry,
        time=time_interval,
        filter=f"eo:cloud_cover <= {max_cloud_cover}",
        fields={"include": ["id", "properties.datetime", "properties.eo:cloud_cover"]}
    )

    items = list(search_iterator)

    if len(items) == 0:
        print("❌ No images found.")
        return

    # 🗂 Group by date and keep lowest cloud cover per day
    best_per_day = {}

    for item in items:
        date_str = item["properties"]["datetime"].split("T")[0]
        cloud = item["properties"]["eo:cloud_cover"]

        if date_str not in best_per_day:
            best_per_day[date_str] = item
        else:
            if cloud < best_per_day[date_str]["properties"]["eo:cloud_cover"]:
                best_per_day[date_str] = item

    print(f"📅 Found {len(best_per_day)} unique days")

    # 📥 DOWNLOAD ONE IMAGE PER DAY
    for date_str, item in sorted(best_per_day.items()):

        cloud_cover = item["properties"]["eo:cloud_cover"]
        print(f"\n📅 {date_str} (cloud cover: {cloud_cover}%)")

        output_path = os.path.join(
            output_dir,
            f"S2_{date_str}.tif"
        )

        if os.path.exists(output_path):
            print("⚠️ Already exists. Skipping.")
            continue

        evalscript = """
        //VERSION=3
        function setup() {
          return {
            input: [{
              bands: ["B01","B02","B03","B04","B05","B06","B07","B08",
                      "B8A","B09","B11","B12",
                      "sunAzimuthAngles","sunZenithAngles",
                      "viewAzimuthMean","viewZenithMean"]
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
                geometry=geometry,
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
                crs='EPSG:32633',
                transform=transform
            ) as dst:
                for i in range(bands):
                    dst.write(image_data[:, :, i], i + 1)

            print(f"💾 Saved to {output_path}")
            time.sleep(1)

        except Exception as e:
            print(f"❌ Error processing {date_str}: {e}")

if __name__ == "__main__":
    main()
