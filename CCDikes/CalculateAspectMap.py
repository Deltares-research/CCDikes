import requests
import rasterio
from io import BytesIO
from osgeo import gdal
import numpy as np
import rioxarray


def calculate_slope(DEM, bounds=None):
    # Calculate slope and export to tif in the espg 28992
    gdal.DEMProcessing('slope.tif', DEM, 'slope')
    rds = rioxarray.open_rasterio('slope.tif')
    slope = rds.rio.write_crs("EPSG:28992")
    slope.rio.to_raster('slope_r.tif')
    return None


def calculate_aspect(DEM):
    gdal.DEMProcessing('aspect.tif', DEM, 'aspect')
    rds = rioxarray.open_rasterio('aspect.tif')
    slope = rds.rio.write_crs("EPSG:28992")
    slope.rio.to_raster('aspect_r.tif')
    return None


# Define the URL for the GetCoverage request
url = "https://service.pdok.nl/rws/ahn/wcs/v1_0"

# Define the parameters for the GetCoverage request
params = {
    "SERVICE": "WCS",
    "request": "GetCoverage",
    "VERSION": "1.0.0",
    "STYLES": "Default",
    "CRS": "EPSG:28992",
    "BBOX": "72404.1738,385300.4088,135175.194,415041.17",
    "FORMAT": "image/tiff",
    "CoverageID": "dtm_05m",
    "COVERAGE": "dtm_05m",
    "RESX": "20",
    "RESY": "20",
}

# Make the GET request
response = requests.get(url, params=params)

if response.status_code == 200:
    # Read the bytes into an in-memory file-like object
    f = BytesIO(response.content)

    # open and save the file
    with open("dtm_05m.tif", "wb") as file:
        file.write(f.read())



    slope = calculate_slope("dtm_05m.tif", [72404.1738,385300.4088,135175.194,415041.17])
    aspect = calculate_aspect("dtm_05m.tif")



