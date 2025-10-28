#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Reader for GEOTiff format
"""

import numpy as np
import struct
from osgeo import gdal, osr

from .FileUtils import ReaderUtilities


class TIFFileReader(ReaderUtilities):
	def __init__(self):
		self.name = "TIFFileReader"
		ReaderUtilities.__init__(self)
		gdal.UseExceptions()

	def get_projection_info(self, file):
		"""Fetch the projection info for this tif"""
		fid = gdal.Open(file)
		meta = fid.GetMetadata()
		projection_info = fid.GetProjection()
		fid = None

		return projection_info

	def get_geo_transform(self, file):
		"""Fetch the geotransform"""
		fid = gdal.Open(file)
		transform = fid.GetGeoTransform()
		fig = None

		return transform

	def read_head(self, file):
		"""As the data comes from different places we just don't need to convert the raster array"""

		fid = gdal.Open(file)

		meta = fid.GetMetadata()
		projection_info = fid.GetProjection()

		"""
		gt0	Top Left X	X-coordinate (longitude or easting) of the upper-left corner of the image.
		gt1	Pixel Width	Size of each pixel in the X direction (positive: east, negative: west).
		gt2	Rotation (X Skew)	Usually 0, but represents rotation if the image is not north-up.
		gt3	Top Left Y	Y-coordinate (latitude or northing) of the upper-left corner.
		gt4	Rotation (Y Skew)	Usually 0, but represents rotation if the image is not north-up.
		gt5	Pixel Height	Size of each pixel in the Y direction (normally negative, as images are stored top-down).
		"""
		tlx, pw, xskew, tly, yskew, ph = fid.GetGeoTransform()

		# Read the first band
		band = fid.GetRasterBand(1)

		# Get raster dimensions
		cols = fid.RasterXSize  # Number of columns
		rows = fid.RasterYSize  # Number of rows

		xllcorner = tlx
		yllcorner = tly + ph * rows

		nodata = band.GetNoDataValue()

		hdat = {
			'filename': file,
			'nrows': rows,
			'ncols': cols,
			'xllcorner': xllcorner,
			'yllcorner': yllcorner,
			'cellsize': pw,
			'nodata': nodata,
		}

		# Close dataset
		fid = None

		return hdat

	def read_tiff_file(self, file):
		"""Read individual file"""

		fid = gdal.Open(file)

		meta = fid.GetMetadata()
		projection_info = fid.GetProjection()

		"""
		gt0	Top Left X	X-coordinate (longitude or easting) of the upper-left corner of the image.
		gt1	Pixel Width	Size of each pixel in the X direction (positive: east, negative: west).
		gt2	Rotation (X Skew)	Usually 0, but represents rotation if the image is not north-up.
		gt3	Top Left Y	Y-coordinate (latitude or northing) of the upper-left corner.
		gt4	Rotation (Y Skew)	Usually 0, but represents rotation if the image is not north-up.
		gt5	Pixel Height	Size of each pixel in the Y direction (normally negative, as images are stored top-down).
		
		geo_x = gt0 + px * gt1 + py * gt2
		geo_y = gt3 + px * gt4 + py * gt5
		"""
		tlx, pw, xskew, tly, yskew, ph = fid.GetGeoTransform()

		# Read the first band
		band = fid.GetRasterBand(1)

		# Get raster dimensions
		cols = fid.RasterXSize  # Number of columns
		rows = fid.RasterYSize  # Number of rows

		# Get data type
		gdal_data_type = band.DataType
		np_data_type = gdal.GetDataTypeName(gdal_data_type)  # e.g., 'Byte', 'UInt16', 'Float32'

		# Read raw raster data as bytes
		raw_data = band.ReadRaster(0, 0, cols, rows, buf_type=gdal_data_type)

		# Determine the correct format string for struct.unpack
		data_type_map = {
			'Byte': 'B',  # Unsigned 8-bit integer
			'UInt16': 'H',  # Unsigned 16-bit integer
			'Int16': 'h',  # Signed 16-bit integer
			'UInt32': 'I',  # Unsigned 32-bit integer
			'Int32': 'i',  # Signed 32-bit integer
			'Float32': 'f',  # 32-bit float
			'Float64': 'd',  # 64-bit float
		}

		# Ensure we have a valid type
		struct_format = data_type_map.get(np_data_type, 'f')  # Default to float32

		# Convert the binary data to a list of numbers
		unpacked_data = struct.unpack(f"{cols * rows}{struct_format}", raw_data)

		# Convert to NumPy array and reshape
		elevation_array = np.array(unpacked_data, dtype=np.float32).reshape((rows, cols))

		# Apply scale and offset if available
		scale = band.GetScale() or 1
		offset = band.GetOffset() or 0
		elevation_array = elevation_array * scale + offset

		# Find the lower left corner points
		xllcorner = tlx
		yllcorner = tly + ph * rows

		# Set nodata if specified
		nodata = band.GetNoDataValue()

		# Output dict
		dat = {
			'filename': file,
			'nrows': rows,
			'ncols': cols,
			'xllcorner': xllcorner,
			'yllcorner': yllcorner,
			'cellsize': pw,
			'nodata': nodata,
			'data': elevation_array,
		}

		# Add some helpful extra entries here:
		dat.setdefault('xurcorner', dat["xllcorner"] + dat["cellsize"] * dat["ncols"])
		dat.setdefault('yurcorner', dat["yllcorner"] + dat["cellsize"] * dat["nrows"])

		# NB: If data none something has gone wrong... Not sure what I want to do here
		if dat["data"] is not None:
			# 			dat.setdefault('mean', np.mean(elevation_array))
			# 			dat.setdefault('standard-deviation', np.std(elevation_array))
			dat.setdefault('min', np.min(elevation_array))
			dat.setdefault('max', np.max(elevation_array))
		else:
			# 			dat.setdefault('mean', None)
			# 			dat.setdefault('standard-deviation', None)
			dat.setdefault('min', None)
			dat.setdefault('max', None)

		# Close dataset
		fid = None

		return dat

	def read(self, fpath):
		"""Reads in a GEOTiff file using GDAL"""
		data_dict = self.read_tiff_file(fpath)
		return data_dict


class TIFFileWriter:
	def __init__(self):
		self.name = "TIFFileWriter"
		gdal.UseExceptions()

	def write(self, file, dat, projection=None):
		"""
		Write a GeoTIFF file from a dict structure like:
		{
			'filename': str,
			'nrows': int,
			'ncols': int,
			'xllcorner': float,
			'yllcorner': float,
			'cellsize': float,
			'nodata': float,
			'data': np.ndarray,
		}
		projection = WKT string else default to WGS84
		"""

		# Sanity checks
		if "data" not in dat:
			raise ValueError("Input dict must contain a 'data' array.")
		if dat["data"].ndim != 2:
			raise ValueError("Data array must be 2D.")

		rows, cols = dat["data"].shape
		cellsize = dat.get("cellsize")
		nodata = dat.get("nodata", -9999.0)
		xllcorner = dat.get("xllcorner")
		yllcorner = dat.get("yllcorner")

		if None in (cellsize, xllcorner, yllcorner):
			raise ValueError("Missing one or more required fields: xllcorner, yllcorner, cellsize")

		# Determine GDAL + struct data type
		data_type_map = {
			np.uint8: (gdal.GDT_Byte, 'B'),
			np.int16: (gdal.GDT_Int16, 'h'),
			np.uint16: (gdal.GDT_UInt16, 'H'),
			np.int32: (gdal.GDT_Int32, 'i'),
			np.uint32: (gdal.GDT_UInt32, 'I'),
			np.float32: (gdal.GDT_Float32, 'f'),
			np.float64: (gdal.GDT_Float64, 'd'),
		}
		gdal_type, struct_code = data_type_map.get(dat["data"].dtype.type, (gdal.GDT_Float32, 'f'))
		data = dat["data"].astype(
			np.float32 if gdal_type == gdal.GDT_Float32 else dat["data"].dtype
		)

		# Replace NaNs with nodata if necessary
		if np.isnan(data).any():
			data = np.nan_to_num(data, nan=nodata)

		# Pack data into binary for WriteRaster
		raw_data = struct.pack(f"{rows * cols}{struct_code}", *data.ravel())

		# Compute GeoTransform
		tlx = xllcorner
		tly = yllcorner + cellsize * rows
		transform = (tlx, cellsize, 0.0, tly, 0.0, -cellsize)

		# Create output dataset
		driver = gdal.GetDriverByName("GTiff")
		out_ds = driver.Create(
			file, cols, rows, 1, gdal_type
		)  # options = ["COMPRESS=LZW, "TILED=YES"]
		if out_ds is None:
			raise RuntimeError(f"Could not create output GeoTIFF: {file}")

		out_ds.SetGeoTransform(transform)

		# Projection (optional)
		if projection:
			out_ds.SetProjection(projection)
		else:
			srs = osr.SpatialReference()
			srs.ImportFromEPSG(4326)  # default to WGS84
			out_ds.SetProjection(srs.ExportToWkt())

		# Write Raster Data to Band: (xoff, yoff, xsize, ysize, buf)
		band = out_ds.GetRasterBand(1)
		band.WriteRaster(0, 0, cols, rows, raw_data, buf_type=gdal_type)
		band.SetNoDataValue(nodata)
		band.FlushCache()

		# Finalize
		out_ds.FlushCache()
		out_ds = None  # close dataset

		return file
