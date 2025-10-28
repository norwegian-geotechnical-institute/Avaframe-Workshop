#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raster Statistics
"""

import numpy as np
from scipy import stats
from itertools import pairwise
from skimage.measure import label, regionprops
from scipy.ndimage import maximum_filter, minimum_filter


class RasterStats:
	def __init__(self, rasterop=None):
		self.rasterop = rasterop  # Supply pointer to RasterOp if needed
		self.mask = None
		self.slope = None
		self.aspect = None

	def __call__(self, stat, dat, *args, mask_dat=None, **kwargs):
		"""
		Compute a given statistic - or set of statistics
		stat can be: 'mean', 'centroid', etc. or 'all' for all available statistics.
		dat: raster dictionary with keys 'data', 'cellsize', 'ncols', 'nrows', etc.
		mask_dat: raster dictionary used to indicate masked area (must be same extent as dat)

		NB: Some data is cached for reuse accross statistics this is cleared before return if
		class called through __call__ method but not through direct methods. Thus, care is needed
		if calling methods directly with different data/masks on same cls instance
		"""
		if stat == "all":
			results = {}
			for m in sorted(self._available(), key=str.lower):  # case-insensitive
				results[m] = getattr(self, m)(dat, *args, mask_dat=mask_dat, **kwargs)
			self._clear_cache()
			return results

		if not hasattr(self, stat):
			raise ValueError(f"Unknown stat '{stat}'")

		out = getattr(self, stat)(dat, *args, mask_dat=mask_dat, **kwargs)
		self._clear_cache()
		return out

	@classmethod
	def _available(cls):
		"""List callable public methods excluding __call__."""
		return [
			m
			for m in dir(cls)
			if not m.startswith("_") and callable(getattr(cls, m)) and m != "__call__"
		]

	def _clear_cache(self):
		"""Clear all cached data"""
		self.mask = None
		self.slope = None
		self.aspect = None

	# Helper Funcs

	def _get_mask_bbox(self, mask):
		"""Return minimal bounding box (row_min, row_max, col_min, col_max) for True values."""
		rows, cols = np.where(mask)
		return rows.min(), rows.max(), cols.min(), cols.max()

	def _compute_mask(self, dat, mask_dat):
		"""
		This ensures that the mask is cached for a unique (dat, mask)
		NB: be careful if cache needs cleared and dat, mask pair recalculated
		"""
		if self.mask is None:
			base_mask = (dat["data"] != dat["nodata"]) & np.isfinite(dat["data"])
			if mask_dat:
				base_mask &= (
					(mask_dat["data"] != mask_dat["nodata"])
					& np.isfinite(mask_dat["data"])
					& (mask_dat["data"] != 0)
				)
			self.mask = base_mask
		return self.mask

	def _compute_slope(self, dat):
		"""Compute and cache the slope angle everywhere - neglecting boundaries"""
		if self.slope is None:
			cs = dat["cellsize"]
			grad_y, grad_x = np.gradient(dat["data"], cs)
			slope_rad = np.sqrt(grad_x**2 + grad_y**2)
			self.slope = np.degrees(np.arctan(slope_rad))
		return self.slope

	def _compute_aspect(self, dat):
		"""Compute and cache the slope aspect everywhere - neglecting boundaries"""
		if self.aspect is None:
			cs = dat["cellsize"]
			grad_y, grad_x = np.gradient(dat["data"], cs)
			aspect_rad = np.arctan2(-grad_y, grad_x)
			aspect_deg = (np.degrees(aspect_rad) + 360) % 360
			flat_mask = np.hypot(grad_x, grad_y) < 1e-3
			aspect_deg[flat_mask] = np.nan
			self.aspect = aspect_deg
		return self.aspect

	# Methods

	def ncols(self, dat, mask_dat=None):
		return dat.get("ncols")

	def nrows(self, dat, mask_dat=None):
		return dat.get("nrows")

	def xllcorner(self, dat, mask_dat=None):
		return dat.get("xllcorner")

	def yllcorner(self, dat, mask_dat=None):
		return dat.get("yllcorner")

	def cellsize(self, dat, mask_dat=None):
		return dat.get("cellsize")

	def nodata(self, dat, mask_dat=None):
		return dat.get("nodata")

	def extent(self, dat, mask_dat=None):
		cs = dat["cellsize"]
		x_min = dat["xllcorner"]
		x_max = dat["xllcorner"] + dat["ncols"] * cs
		y_min = dat["yllcorner"]
		y_max = dat["yllcorner"] + dat["nrows"] * cs
		return (x_min, x_max, y_min, y_max)

	def dataextent(self, dat, mask_dat=None, threshold=1e-12):
		"""Extent of > 0 finite data"""

		mask = np.where(self._compute_mask(dat, mask_dat), 1.0, 0.0)
		row_min, row_max, col_min, col_max = self._get_mask_bbox(mask)

		# NB: To get the realworld bounding box for the area we need to convert to edge aligned coordinates
		xmin, ymin = self.rasterop._loc(dat, row_max, col_min, mode="edgeWS")
		xmax, ymax = self.rasterop._loc(dat, row_min, col_max, mode="edgeEN")

		return (float(xmin), float(xmax), float(ymin), float(ymax))

	def mean(self, dat, mask_dat=None):
		valid_data = dat["data"][self._compute_mask(dat, mask_dat)]
		return float(np.mean(valid_data))

	def median(self, dat, mask_dat=None):
		valid_data = dat["data"][self._compute_mask(dat, mask_dat)]
		return float(np.median(valid_data))

	def mode(self, dat, mask_dat=None):
		valid_data = dat["data"][self._compute_mask(dat, mask_dat)]
		return float(stats.mode(valid_data, keepdims=True).mode[0])

	def standard_deviation(self, dat, mask_dat=None):
		valid_data = dat["data"][self._compute_mask(dat, mask_dat)]
		return float(np.std(valid_data))

	def variance(self, dat, mask_dat=None):
		valid_data = dat["data"][self._compute_mask(dat, mask_dat)]
		return float(np.var(valid_data))

	def min(self, dat, mask_dat=None):
		valid_data = dat["data"][self._compute_mask(dat, mask_dat)]
		return float(np.min(valid_data))

	def max(self, dat, mask_dat=None):
		valid_data = dat["data"][self._compute_mask(dat, mask_dat)]
		return float(np.max(valid_data))

	def sum(self, dat, mask_dat=None):
		valid_data = dat["data"][self._compute_mask(dat, mask_dat)]
		return float(np.nansum(valid_data))

	def relief(self, dat, mask_dat=None):
		return self.max(dat, mask_dat) - self.min(dat, mask_dat)

	def local_relief(self, dat, mask_dat=None, size=3):
		"""
		Max minus min elevation in a moving window (e.g., 3×3 cells).
		Measures local relief; high in rugged terrain, low in flat areas.
		"""
		valid_data = dat["data"][self._compute_mask(dat, mask_dat)]
		return float(
			np.nanmean(
				maximum_filter(valid_data, size=size) - minimum_filter(valid_data, size=size)
			)
		)

	def argmin(self, dat, mask_dat=None):
		data = dat["data"]
		flat_index = np.nanargmin(np.where(self._compute_mask(dat, mask_dat), data, np.nan))
		row, col = np.unravel_index(flat_index, data.shape)
		return (int(row), int(col))

	def argmax(self, dat, mask_dat=None):
		data = dat["data"]
		flat_index = np.nanargmax(np.where(self._compute_mask(dat, mask_dat), data, np.nan))
		row, col = np.unravel_index(flat_index, data.shape)
		return (int(row), int(col))

	def min_loc(self, dat, mask_dat=None):
		cs = dat["cellsize"]
		data = dat["data"]
		flat_index = np.nanargmin(np.where(self._compute_mask(dat, mask_dat), data, np.nan))
		row, col = np.unravel_index(flat_index, data.shape)
		return (
			float(dat["xllcorner"] + col * cs),
			float(dat["yllcorner"] + (dat["nrows"] - row - 1) * cs),
		)

	def max_loc(self, dat, mask_dat=None):
		cs = dat["cellsize"]
		data = dat["data"]
		flat_index = np.nanargmax(np.where(self._compute_mask(dat, mask_dat), data, np.nan))
		row, col = np.unravel_index(flat_index, data.shape)
		return (
			float(dat["xllcorner"] + col * cs),
			float(dat["yllcorner"] + (dat["nrows"] - row - 1) * cs),
		)

	def centroid(self, dat, mask_dat=None):
		cs = dat["cellsize"]
		data = dat["data"]
		mask = self._compute_mask(dat, mask_dat)
		valid_data = data[mask]

		data_range = np.nanmax(valid_data) - np.nanmin(valid_data)
		norm_data = data if data_range == 0 else (data - np.nanmin(valid_data)) / data_range

		intensitymap = np.where(mask, norm_data, 0.0)
		aoi = (intensitymap > 1e-3).astype(np.uint8)
		labeled = label(aoi)
		props = regionprops(labeled, intensity_image=intensitymap)

		if props:
			r_centroid, c_centroid = props[0].centroid
			return (
				float(dat["xllcorner"] + c_centroid * cs),
				float(dat["yllcorner"] + (dat["nrows"] - r_centroid - 1) * cs),
			)
		return (None, None)

	def weighted_centroid(self, dat, mask_dat=None):
		cs = dat["cellsize"]
		data = dat["data"]
		mask = self._compute_mask(dat, mask_dat)
		valid_data = data[mask]

		data_range = np.nanmax(valid_data) - np.nanmin(valid_data)
		norm_data = data if data_range == 0 else (data - np.nanmin(valid_data)) / data_range

		intensitymap = np.where(mask, norm_data, 0.0)
		aoi = (intensitymap > 1e-3).astype(np.uint8)
		labeled = label(aoi)
		props = regionprops(labeled, intensity_image=intensitymap)

		if props:
			rw_centroid, cw_centroid = props[0].weighted_centroid
			return (
				float(dat["xllcorner"] + cw_centroid * cs),
				float(dat["yllcorner"] + (dat["nrows"] - rw_centroid - 1) * cs),
			)
		return (None, None)

	def bounding_box(self, dat, mask_dat=None):
		cs = dat["cellsize"]
		rows, cols = np.where(self._compute_mask(dat, mask_dat))

		min_row, max_row = rows.min(), rows.max()
		min_col, max_col = cols.min(), cols.max()

		bbox_xll = float(dat["xllcorner"] + min_col * cs)
		bbox_yll = float(dat["yllcorner"] + (dat["nrows"] - max_row - 1) * cs)
		bbox_width = float((max_col - min_col + 1) * cs)
		bbox_height = float((max_row - min_row + 1) * cs)

		return (bbox_xll, bbox_yll, bbox_width, bbox_height)

	def area(self, dat, mask_dat=None):
		"""planimetric (oblique) area"""
		cs = dat["cellsize"]
		return float(self._compute_mask(dat, mask_dat).sum() * cs * cs)

	def surface_area(self, dat, mask_dat=None):
		mask = self._compute_mask(dat, mask_dat)
		cs = dat["cellsize"]
		# factor = np.sqrt(1 + (np.tan(np.radians(self._compute_slope(dat, mask_dat))))**2)
		factor = (1 / np.cos(np.radians(self._compute_slope(dat))))[mask]
		return float(np.sum((cs**2) * factor))

	def volume(self, dat, mask_dat=None):
		cs = dat["cellsize"]
		valid_data = dat["data"][self._compute_mask(dat, mask_dat)]
		return float(np.sum(valid_data) * cs * cs)

	def mask_volume(self, dat, mask_dat=None):
		if mask_dat is None:
			return None
		cs = dat["cellsize"]
		return float(mask_dat["data"][self._compute_mask(dat, mask_dat)].sum() * cs * cs)

	def skew(self, dat, mask_dat=None):
		valid_data = dat["data"][self._compute_mask(dat, mask_dat)]
		return float(stats.skew(valid_data))

	def kurtosis(self, dat, mask_dat=None):
		valid_data = dat["data"][self._compute_mask(dat, mask_dat)]
		return float(stats.kurtosis(valid_data))

	def percentiles(self, dat, mask_dat=None, percentiles=[10, 25, 75, 90]):
		valid_data = dat["data"][self._compute_mask(dat, mask_dat)]
		return {f"p{p}": float(np.percentile(valid_data, p)) for p in percentiles}

	def slope_percentage_area(self, dat, mask_dat=None):
		cs = dat["cellsize"]

		# NB: slope may have non-finite values
		slope = self._compute_slope(dat)
		area = self.area(dat, mask_dat=mask_dat)

		slope_valid = slope[self._compute_mask(dat, mask_dat)]

		slope_bins = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
		sbin_labels = [f"{i}-{j}" for i, j in pairwise(slope_bins)]
		sbin_counts, _ = np.histogram(slope_valid, bins=slope_bins)

		sbin_area_percentage = (sbin_counts * cs**2 / area) * 100

		return dict(zip(sbin_labels, sbin_area_percentage.tolist()))

	def mean_slope(self, dat, mask_dat=None):
		# NB: Note just mask may still leave non-finite values in the slope array
		slope_valid = self._compute_slope(dat)[self._compute_mask(dat, mask_dat)]
		return float(np.nanmean(slope_valid))

	def std_slope(self, dat, mask_dat=None):
		# NB: Note just mask may still leave non-finite values in the slope array
		slope_valid = self._compute_slope(dat)[self._compute_mask(dat, mask_dat)]
		return float(np.nanstd(slope_valid))

	def aspect_histogram(self, dat, mask_dat=None):
		# NB: Note just mask may still leave non-finite values (flat areas) in the aspect array
		aspect_valid = self._compute_aspect(dat)[self._compute_mask(dat, mask_dat)]

		aspect_bins = np.array([0, 45, 90, 135, 180, 225, 270, 315, 360])
		bin_labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
		bin_counts, _ = np.histogram(aspect_valid, bins=aspect_bins)

		return dict(zip(bin_labels, bin_counts.tolist()))

	def mean_aspect(self, dat, mask_dat=None):
		# NB: Note just mask may still leave non-finite values (flat areas) in the aspect array
		aspect_valid = self._compute_aspect(dat)[self._compute_mask(dat, mask_dat)]
		return float(np.nanmean(aspect_valid))

	def std_aspect(self, dat, mask_dat=None):
		# NB: Note just mask may still leave non-finite values (flat areas) in the aspect array
		aspect_valid = self._compute_aspect(dat)[self._compute_mask(dat, mask_dat)]
		return float(np.nanstd(aspect_valid))

	def TRI(self, dat, mask_dat=None, size=3):
		"""
		Terrain Ruggedness Index (Riley et al., 1999)
		Mean absolute difference in relief between a cell and its neighbors
		i.e. window of size (default) 3x3
		"""
		mask = self._compute_mask(dat, mask_dat)

		# Get minimal bounding box
		ymin, ymax, xmin, xmax = self._get_mask_bbox(mask)
		arr = dat["data"][ymin:ymax, xmin:xmax]
		submask = mask[ymin:ymax, xmin:xmax]

		pad = size // 2
		padded = np.pad(arr, pad, mode="edge")

		tri = np.zeros_like(arr, dtype=float)
		for dy in range(-pad, pad + 1):
			for dx in range(-pad, pad + 1):
				if dy == 0 and dx == 0:
					continue
				tri += np.abs(
					arr
					- padded[pad + dy : pad + dy + arr.shape[0], pad + dx : pad + dx + arr.shape[1]]
				)

		tri /= size * size - 1
		return float(np.nanmean(tri[submask]))

	def VRM(self, dat, mask_dat=None, size=3):
		"""
		Vector Ruggedness Measure (Hobson, 1972)
		Measures dispersion of surface normals. (0 = smooth, 1 = maximally rough)
		"""
		mask = self._compute_mask(dat, mask_dat)

		# Get minimal bounding box
		ymin, ymax, xmin, xmax = self._get_mask_bbox(mask)
		submask = mask[ymin:ymax, xmin:xmax]

		# Crop slope and aspect
		slope = np.radians(self._compute_slope(dat)[ymin:ymax, xmin:xmax])
		aspect = np.radians(self._compute_aspect(dat)[ymin:ymax, xmin:xmax])

		# Convert slope/aspect to unit normal components
		nx = np.sin(slope) * np.sin(aspect)
		ny = np.sin(slope) * np.cos(aspect)
		nz = np.cos(slope)

		pad = size // 2
		nx_p = np.pad(nx, pad, mode="edge")
		ny_p = np.pad(ny, pad, mode="edge")
		nz_p = np.pad(nz, pad, mode="edge")

		vrm = np.zeros_like(slope, dtype=float)
		for y in range(vrm.shape[0]):
			for x in range(vrm.shape[1]):
				wx = nx_p[y : y + size, x : x + size].mean()
				wy = ny_p[y : y + size, x : x + size].mean()
				wz = nz_p[y : y + size, x : x + size].mean()
				R = np.sqrt(wx**2 + wy**2 + wz**2)
				vrm[y, x] = 1 - R

		return float(np.nanmean(vrm[submask]))

	def wind_exposure(self, dat, mask_dat=None):
		"""
		Compute wind exposure/shelter as the mean dot product between
		surface normals and a given wind direction.

		wind_dir_deg: direction FROM which wind is coming (0 = N, 90 = E, etc.)
		Positive values = facing wind (exposed), negative = sheltered.
		"""
		mask = self._compute_mask(dat, mask_dat)

		# Crop to minimal bounding box
		ymin, ymax, xmin, xmax = self._get_mask_bbox(mask)
		submask = mask[ymin:ymax, xmin:xmax]

		slope = np.radians(self._compute_slope(dat)[ymin:ymax, xmin:xmax])
		aspect = np.radians(self._compute_aspect(dat)[ymin:ymax, xmin:xmax])

		# Surface normal components
		nx = np.sin(slope) * np.sin(aspect)
		ny = np.sin(slope) * np.cos(aspect)
		nz = np.cos(slope)

		wind_dir_deg = np.array([0, 45, 90, 135, 180, 225, 270, 315])
		wind_dir_labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

		wind_exposure = {}
		for wl, wd in zip(wind_dir_labels, wind_dir_deg):

			# Convert wind direction to a unit vector
			# Assume wind is horizontal, coming from wind_dir_deg
			wind_dir_rad = np.radians(wd)
			wx = np.sin(wind_dir_rad)
			wy = np.cos(wind_dir_rad)
			wz = 0.0

			# Dot product: normal ⋅ wind vector
			dot = nx * wx + ny * wy + nz * wz

			wind_exposure[wl] = float(np.nanmean(dot[submask]))

		return wind_exposure

	def valid_fraction(self, dat, mask_dat=None):
		total_cells = dat["data"].size
		valid_cells = self._compute_mask(dat, mask_dat).sum()
		return float(valid_cells / total_cells)
