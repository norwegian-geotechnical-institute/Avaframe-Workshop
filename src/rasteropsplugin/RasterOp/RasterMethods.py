#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raster Methods
"""

import re
import numpy as np
from scipy.ndimage import zoom, correlate1d
from scipy import stats
from skimage.measure import regionprops, label
from rich.pretty import pprint
from pathlib import Path
from osgeo import gdal, osr
import operator
import shutil as sh

gdal.UseExceptions()
from itertools import pairwise
import io


# from Batch.Utilities import WARNING, ERROR, SUCCESS
from ..FileIO.FileReaders import GeoFileReader as Reader
from ..FileIO.FileWriters import ASCIIFileWriter as Writer # TIFFileWriter

from .Recipes import Recipes
from .Normaliser import Normaliser
from .Noise import NoiseAdder
from .Curvature import Curvature
from .Combine import Combine
from .RasterStats import RasterStats
from .Rotations import Rotator
from .Derivatives import FDDerivative


class RasterOperations:
	def __init__(self):
		self.asctimesteppattern = re.compile(r'.*\w\_(\d{4}).asc')
		self.txttimesteppattern = re.compile(r'.*(\d{6}).txt')

	@classmethod
	def _available(cls):
		"""List callable public methods excluding __call__."""
		return [
			m
			for m in dir(cls)
			if callable(getattr(cls, m))
			and m != "__call__"
			and m != "_available"
			and m[-2:] != '__'
		]

	def _metadata(self, dat):
		""" "Extract any metadata from the naming convention"""

		fpath = Path(dat.get("filename", ""))

		if fpath == Path(""):
			RuntimeError("dat does not contain the filepath under key 'fname'.")

		filename = fpath.stem
		ftype = fpath.suffix.lstrip(".")

		timestep = -1

		# For non-timestep pattern filenames give -1 timestep
		if ftype == "asc":
			ascval = self.asctimesteppattern.search(filename)
			if ascval:
				timestep = int(ascval.group(1))
		elif ftype == "txt":
			txtval = self.txttimesteppattern.search(fpath)
			if txtval:
				timestep = int(txtval.group(1))
		else:
			timestep = -1

		# Next we search for a field name
		# If timestep file its the sub-directory name else label with file name (basename)
		if timestep < 0:
			fname = filename
			timestep = 0
		else:
			fname = fpath.parent.stem

		# NB: Group output so its extensible if more elaborate naming convention applies
		out = dict(
			filepath=fpath,
			filename=fname,
			filetype=ftype,
			timestep=timestep,
		)
		return out

	def _loc(self, dat, i, j, mode="centered"):
		"""
		Convert row (i), col (j) indices into (x, y) coordinates.

		i: row index (0 = top row of array)
		j: column index (0 = leftmost col)

		Use mode centered for cell-centered aligned coordinates.
		Use mode edgeWS for the x West aligned and y South aligned edge coordinates.
		Use mode edgeEN for the x East aligned and y North aligned edges coordinates.
		"""
		xll, yll = dat["xllcorner"], dat["yllcorner"]
		cs = dat["cellsize"]
		nrows = dat["nrows"]

		if mode == "centered":
			# col j -> x position
			x = xll + (j + 0.5) * cs
			# row i -> y position (flip since row 0 is top)
			y = yll + (nrows - i - 0.5) * cs
		elif mode == "edgeWS":
			# col j -> x position
			x = xll + j * cs
			# row i -> y position (flip since row 0 is top)
			y = yll + (nrows - 1 - i) * cs
		elif mode == "edgeEN":
			# col j -> x position
			x = xll + (j + 1) * cs
			# row i -> y position (flip since row 0 is top)
			y = yll + (nrows - i) * cs
		else:
			ValueError(f"Mode {mode} unknown. Choose from {('centered', 'edgeWS', 'edgeEN')}")

		return x, y

	def _iloc(self, dat, posx, posy):
		"""
		Convert (x, y) world position into array indices (row, col).
		Returns integers (i, j).
		"""
		xll, yll = dat["xllcorner"], dat["yllcorner"]
		cs = dat["cellsize"]
		nrows, ncols = dat["nrows"], dat["ncols"]

		j = int((posx - xll) / cs)
		i = int((nrows - (posy - yll) / cs) - 1)

		if not all(0 <= i_ < bound for i_, bound in zip([i, j], [nrows, ncols])):
			print(WARNING("Position is outside of array", colour=True))
			return None, None
		return i, j

	def _val(self, dat, posx, posy):
		"""
		Return raster value at world position (x, y).
		Uses nearest-cell lookup.
		Returns nodata if position is outside grid.
		"""
		i, j = self._iloc(dat, posx, posy)

		if i < 0 or i >= dat["nrows"] or j < 0 or j >= dat["ncols"]:
			return dat.get("nodata", np.nan)
		return dat["data"][i, j]

	def _xvals(self, dat, mode="centers"):
		"""Return the xrange vector for the data on centers or edges"""
		xll = dat['xllcorner']
		nx = dat['ncols']
		ds = dat['cellsize']

		# Cell centers
		if mode == "centers":
			return xll + ds * (0.5 + np.arange(nx))
		elif mode == "edges":
			return xll + ds * np.arange(nx + 1)
		else:
			raise ValueError(f"Invalid mode {mode}. Choose from 'centers' or 'edges'.")

	def _yvals(self, dat, mode="centers"):
		"""Return the xrange vector for the data on centers or edges"""
		yll = dat['xllcorner']
		ny = dat['nrows']
		ds = dat['cellsize']

		# Cell centers
		if mode == "centers":
			return yll + ds * (0.5 + np.arange(ny))
		elif mode == "edges":
			return yll + ds * np.arange(ny + 1)
		else:
			raise ValueError(f"Invalid mode {mode}. Choose from 'centers' or 'edges'.")

	def _pvals(self, dat, mode="centers"):
		"""
		Return parametric grids (Xc, Yc for cell centers; Xe, Ye for edges),
		the data array Z, and data extent [xll, xur, yll, yur].
		"""
		xll, yll = dat['xllcorner'], dat['yllcorner']
		ds = dat['cellsize']
		nx, ny = dat['ncols'], dat['nrows']

		# Upper-right corner (edges, not centers)
		xur = xll + nx * ds
		yur = yll + ny * ds

		# Cell centers
		if mode == "centers":
			x_centers = xll + ds * (0.5 + np.arange(nx))
			y_centers = yll + ds * (0.5 + np.arange(ny))
			X, Y = np.meshgrid(x_centers, y_centers)

		# Cell edges
		elif mode == "edges":
			x_edges = xll + ds * np.arange(nx + 1)
			y_edges = yll + ds * np.arange(ny + 1)
			X, Y = np.meshgrid(x_edges, y_edges)

		else:
			raise ValueError(f"Invalid mode {mode}. Choose from 'centers' or 'edges'.")

		Z = dat['data']  # shape (ny, nx), aligned with centers
		extent = [xll, xur, yll, yur]

		return X, Y, Z, extent

	def _abs(self, idat):
		"""Element-wise absolute value of the raster calculated inplace"""
		data = idat["data"]
		data[...] = np.abs(data)
		return idat

	def _swap(self, idat, from_val, to_val):
		"""
		Swap method calculated inplace
		Use keyword from_val="nodata" to swap the nodata val from whatever it is set to
		"""

		if isinstance(from_val, str) and from_val == "nodata":
			from_val = idat["nodata"]

		data = idat["data"]
		data[data == from_val] = to_val
		return idat

	def _threshold(
		self, idat, cond_arr, cond_op, cond_val, bitops=None, threshold_val=1.0, fill_val=0.0
	):
		"""
		Threshold method. Construct and apply a mask from condition operations.
		Available ops: {> >= < <= == != // %}
		Bitwise ops: {& | ^}
		Use keyword self to represent in the input array e.g. cond_arr = self or threshold_val = self etc.
		"""

		OPS = {
			">": operator.gt,
			">=": operator.ge,
			"<": operator.lt,
			"<=": operator.le,
			"==": operator.eq,
			"!=": operator.ne,
			"//": operator.floordiv,
			"%": operator.mod,
		}

		BITOPS = {
			"&": np.logical_and,
			"|": np.logical_or,
			"^": np.logical_xor,  # might be useful if thresholding for area outside an extent etc.
		}

		if not (len(cond_arr) == len(cond_op) == len(cond_val)):
			raise ValueError(
				f"cond_arr ({len(cond_arr)}), cond_ops ({len(cond_op)}), "
				f"and cond_val ({len(cond_val)}) must have the same length."
			)

		if bitops and len(bitops) != len(cond_arr) - 1:
			raise ValueError(
				f"Number of bitops ({len(bitops)}) must be exactly "
				f"len(cond_arr) - 1 ({len(cond_arr) - 1})."
			)

		for op in cond_op:
			if op not in OPS:
				raise ValueError(f"Invalid comparison operator '{op}'. Allowed: {list(OPS.keys())}")

		if bitops:
			for bop in bitops:
				if bop not in BITOPS:
					raise ValueError(
						f"Invalid bitwise operator '{bop}'. Allowed: {list(BITOPS.keys())}"
					)

		# Load the base array
		arrays = []
		for arr_dat in cond_arr:
			if arr_dat == "self":
				arrays.append(idat["data"])
			else:
				arr_data = arr_dat["data"]
				if arr_data.shape != idat["data"].shape:
					raise ValueError(
						f"Condition array shape {arr_data.shape} does not match input shape {idat['data'].shape}"
					)
				arrays.append(arr_data)

		# Build each individual condition mask
		cond_masks = []
		for arr, op, val in zip(arrays, cond_op, cond_val):
			cond_masks.append(OPS[op](arr, val))

		# Combine conditions
		mask = cond_masks[0]
		if bitops:
			for bop, cm in zip(bitops, cond_masks[1:]):
				mask = BITOPS[bop](mask, cm)

		if threshold_val == "self":
			threshold_val = idat["data"]

		if fill_val == "self":
			fill_val = idat["data"]

		# Apply np.where
		result = np.where(mask, threshold_val, fill_val)
		idat["data"] = result
		return idat

	def _snap_to_grid(self, coord, origin, cs):
		"""Snap coordinate to nearest grid point based on origin and cellsize."""
		return origin + round((coord - origin) / cs) * cs

	def _clip(self, idat, xll=None, yll=None, xur=None, yur=None):
		"""Clip raster to bounding box defined by (xll, yll) - (xur, yur)."""
		data = idat["data"]
		cs = idat["cellsize"]
		nodata = idat["nodata"]

		# Original bounds
		x0 = idat["xllcorner"]
		y0 = idat["yllcorner"]
		nrows = idat["nrows"]
		ncols = idat["ncols"]
		xmax = x0 + ncols * cs
		ymax = y0 + nrows * cs

		# Clamp clip window to original extent
		xll = np.clip(xll, x0, xmax) if xll is not None else x0
		yll = np.clip(yll, y0, ymax) if yll is not None else y0
		xur = np.clip(xur, x0, xmax) if xur is not None else xmax
		yur = np.clip(yur, y0, ymax) if yur is not None else ymax

		# Snap to grid
		xll = self._snap_to_grid(xll, x0, cs)
		yll = self._snap_to_grid(yll, y0, cs)
		xur = self._snap_to_grid(xur, x0, cs)
		yur = self._snap_to_grid(yur, y0, cs)

		# Sanity check
		eps = 1e-6
		if not (x0 - eps <= xll < xur <= xmax + eps and y0 - eps <= yll < yur <= ymax + eps):
			raise ValueError("Clip window lies outside the raster domain.")

		# Index window
		col0 = int(round((xll - x0) / cs))
		col1 = int(round((xur - x0) / cs))
		row0 = int(round((ymax - yur) / cs))
		row1 = int(round((ymax - yll) / cs))

		clipped = data[row0:row1, col0:col1]

		out = idat.copy()
		out["ncols"] = clipped.shape[1]
		out["nrows"] = clipped.shape[0]
		out["xllcorner"] = xll
		out["yllcorner"] = yll
		out["data"] = clipped

		return out

	def _extend(self, idat, xll=None, yll=None, xur=None, yur=None, fill_value=0.0):
		"""Extend raster to cover the bounding box (xll, yll) - (xur, yur)."""
		data = idat["data"]
		cs = idat["cellsize"]
		x0 = idat["xllcorner"]
		y0 = idat["yllcorner"]
		nrows = idat["nrows"]
		ncols = idat["ncols"]
		xmax = x0 + ncols * cs
		ymax = y0 + nrows * cs

		# Compute extension bounds
		xll = min(x0, xll) if xll is not None else x0
		yll = min(y0, yll) if yll is not None else y0
		xur = max(xur, xmax) if xur is not None else xmax
		yur = max(yur, ymax) if yur is not None else ymax

		# Snap to grid
		xll = self._snap_to_grid(xll, x0, cs)
		yll = self._snap_to_grid(yll, y0, cs)
		xur = self._snap_to_grid(xur, x0, cs)
		yur = self._snap_to_grid(yur, y0, cs)

		# New dimensions
		new_ncols = int(round((xur - xll) / cs))
		new_nrows = int(round((yur - yll) / cs))

		# Offsets for inserting old data into new array
		col_offset = int(round((x0 - xll) / cs))
		row_offset = int(round((yur - ymax) / cs))

		# Create new extended raster
		extended = np.full((new_nrows, new_ncols), fill_value, dtype=data.dtype)
		extended[row_offset : row_offset + nrows, col_offset : col_offset + ncols] = data

		out = idat.copy()
		out["ncols"] = new_ncols
		out["nrows"] = new_nrows
		out["xllcorner"] = xll
		out["yllcorner"] = yll
		out["data"] = extended

		return out

	def _match(self, idat, tdat, fill_value=0.0):
		"""Match the raster `idat` to the spatial extent and resolution of template `tdat`."""

		headerfields = ["xllcorner", "yllcorner", "cellsize", "ncols", "nrows", "nodata"]

		if all(np.isclose(tdat[k], idat[k]) for k in headerfields):
			return idat  # Already matches

		xll, yll, cs, ncols, nrows, _ = [tdat[k] for k in headerfields]
		xur = xll + ncols * cs
		yur = yll + nrows * cs

		extended = self._extend(idat, xll, yll, xur, yur, fill_value=fill_value)
		matched = self._clip(extended, xll, yll, xur, yur)

		return matched

	def _resample(self, idat, target_cellsize=5, order=1):
		"""Resample method"""

		data = idat["data"]
		cs = idat["cellsize"]
		nodata = idat["nodata"]

		# Calculate zoom factor
		zoom_factor = cs / target_cellsize
		new_data = zoom(data, zoom=zoom_factor, order=order)

		# Round shape to int
		new_data = np.array(new_data, dtype=np.float32)
		new_nrows, new_ncols = new_data.shape

		# Update metadata
		out = dat.copy()
		out["data"] = new_data
		out["ncols"] = new_ncols
		out["nrows"] = new_nrows
		out["cellsize"] = target_cellsize

		return out

	def _scale(self, idat, factor=1):
		"""Factor method"""
		# Fetch input data
		data = idat["data"]
		nodata = idat["nodata"]

		# Only scale valid values
		new_data = np.where(data != nodata, factor * data, nodata)
		new_data[np.isclose(new_data, 0.0)] = 0.0

		out = idat.copy()
		out["data"] = new_data

		return out

	def _normalise(self, idat, method="minmax", **kwargs):
		"""Normalise method"""
		# Fetch input data
		data = idat["data"]
		normalised = Normaliser(nodata=idat["nodata"])(method, data, **kwargs)

		out = idat.copy()
		out["data"] = normalised

		return out

	def _rotate(self, idat, nodata_strat="mask-aware", **kwargs):
		"""Rotate raster about an arbitrary axis in 3D"""
		return Rotator(nodata=idat["nodata"], nodata_strat=nodata_strat).Rotate(idat, **kwargs)

	def _split(self, idat, val):
		"""Split a raster into a band i.e. where element == val"""

		# Fetch input data
		data = idat["data"]
		nodata = idat["nodata"]

		# Compute band
		band = np.where(np.isclose(data, val), data, nodata)

		out = idat.copy()
		out["data"] = band

		return out

	def _union(self, r1, r2, fill_value=0.0, combine="max"):
		"""Union method"""

		# Same resolution failsafe
		cs = r1["cellsize"]
		assert np.isclose(cs, r2["cellsize"]), "Cell sizes must match"

		# Determine union domain
		xll = min(r1["xllcorner"], r2["xllcorner"])
		yll = min(r1["yllcorner"], r2["yllcorner"])
		xur1 = r1["xllcorner"] + r1["ncols"] * cs
		yur1 = r1["yllcorner"] + r1["nrows"] * cs
		xur2 = r2["xllcorner"] + r2["ncols"] * cs
		yur2 = r2["yllcorner"] + r2["nrows"] * cs
		xur = max(xur1, xur2)
		yur = max(yur1, yur2)

		# Extend both
		ext1 = self._extend(r1, xll=xll, yll=yll, xur=xur, yur=yur, fill_value=fill_value)
		ext2 = self._extend(r2, xll=xll, yll=yll, xur=xur, yur=yur, fill_value=fill_value)

		# Combine data
		data1 = ext1["data"]
		data2 = ext2["data"]

		combined = Combine(fill_value, nodata=r1["nodata"])(combine, data1, data2)

		out = ext1.copy()
		out["data"] = combined
		return out

	def _intersection(self, r1, r2, fill_value=0.0, combine="max"):
		"""Intersection method"""

		# Same resolution failsafe
		cs = r1["cellsize"]
		assert np.isclose(cs, r2["cellsize"]), "Cell sizes must match"

		nodata1 = r1["nodata"]
		nodata2 = r2["nodata"]

		# Extend both to the union extent first to align them
		xll = min(r1["xllcorner"], r2["xllcorner"])
		yll = min(r1["yllcorner"], r2["yllcorner"])
		xur = max(r1["xllcorner"] + r1["ncols"] * cs, r2["xllcorner"] + r2["ncols"] * cs)
		yur = max(r1["yllcorner"] + r1["nrows"] * cs, r2["yllcorner"] + r2["nrows"] * cs)

		ext1 = self._extend(r1, xll=xll, yll=yll, xur=xur, yur=yur, fill_value=nodata1)
		ext2 = self._extend(r2, xll=xll, yll=yll, xur=xur, yur=yur, fill_value=nodata2)

		data1 = ext1["data"]
		data2 = ext2["data"]

		# Valid mask for intersection
		valid_mask = (data1 != 0) & (data1 != nodata1) & (data2 != 0) & (data2 != nodata2)

		if not np.any(valid_mask):
			raise ValueError("No overlapping valid data found for intersection.")

		# Bounding box of valid mask
		rows, cols = np.where(valid_mask)
		rmin, rmax = rows.min(), rows.max()
		cmin, cmax = cols.min(), cols.max()

		# Clip to bbox
		cropped1 = data1[rmin : rmax + 1, cmin : cmax + 1]
		cropped2 = data2[rmin : rmax + 1, cmin : cmax + 1]

		# Combine
		combined = Combine(fill_value, nodata=r1["nodata"])(combine, cropped1, cropped2)

		out = ext1.copy()
		out["data"] = combined
		out["nrows"], out["ncols"] = combined.shape
		out["xllcorner"] = xll + cmin * cs
		out["yllcorner"] = yur - (rmax + 1) * cs

		return out

	def _mask(self, dat, mask, fill_value=0.0):
		"""Mask raster 1 with valid values from raster 2"""

		# Ensure raster 2 matches domain of raster 1
		matched = self._match(mask, dat, fill_value=fill_value)

		# Combine data such that
		data = dat["data"]
		mask_data = matched["data"]

		valid_mask = np.isfinite(mask_data) & (mask_data != matched["nodata"])

		masked = np.full_like(data, fill_value)
		masked[mask] = data[mask]

		out = dat.copy()
		out["data"] = masked
		return out

	def _noise(self, idat, mode="normal", seed=None, **kwargs):
		"""Add noise to raster"""
		data = idat["data"]
		ndata = NoiseAdder(nodata=idat["nodata"], seed=seed)(mode, data, **kwargs)

		out = idat.copy()
		out["data"] = ndata
		return out

	def _slope(self, idat):
		"""Compute the slope in degrees per cell"""
		cs = idat["cellsize"]
		grad_y, grad_x = np.gradient(idat["data"], cs)
		slope_rad = np.sqrt(grad_x**2 + grad_y**2)
		slope = np.degrees(np.arctan(slope_rad))

		out = idat.copy()
		out["data"] = slope
		return out

	def _aspect(self, idat):
		"""Compute the aspect in degrees per cell - assuming raster aligned N=0 deg"""
		cs = idat["cellsize"]
		grad_y, grad_x = np.gradient(idat["data"], cs)
		aspect_rad = np.arctan2(-grad_y, grad_x)
		aspect_deg = (np.degrees(aspect_rad) + 360) % 360
		flat_mask = np.hypot(grad_x, grad_y) < 1e-3
		aspect_deg[flat_mask] = np.nan

		out = idat.copy()
		out["data"] = aspect_deg
		return out

	def _curvature(self, idat, method="planform", **kwargs):
		"""Compute curvature rasters"""
		cdata = Curvature()(method, idat, **kwargs)
		out = idat.copy()
		out["data"] = cdata
		return out

	def _derivative(self, dat, dir_str, order=2, mode="extrapolate", **kwargs):
		"""
		Compute directional derivatives of a raster using finite difference schemes.

		dir_str: str containing combination of "x" and "y", e.g. "x", "xy", "yy"
		order: int (1 to 5), accuracy order of finite difference stencil
		mode: str, how to treat boundaries:
			- "nearest": replicate nearest value
			- "reflect": mirror boundary
			- "constant": pad with NaNs
			- "extrapolate": use one-sided stencils at boundaries for full-order accuracy
		"""
		return FDDerivative()(dat, dir_str, order=order, mode=mode)

	def _D8(self, idat, threshold_slope_degs=None, **kwargs):
		"""
		Compute the D8 direction raster for the cardinal/diagonal directions.
		Direction map: 0=NW, 1=N, 2=NE, 3=E, 4=None, 5=SE, 6=S, 7=SW, 8=W
		Picks the steepest *downslope* neighbor. If none downslope -> 4.
		"""

		mask = np.isfinite(idat["data"]) & (idat["data"] != idat["nodata"])
		data = idat["data"]
		cellsize = idat["cellsize"]
		valid_data = np.where(mask, data, -np.inf)  # masked data is -inf so it is never chosen

		# Pad array so there is a one cell margin on outside edge
		padded = np.pad(valid_data, 1, mode="edge")

		window = []
		for row in range(3):
			for col in range(3):
				window.append(
					padded[row : (row + padded.shape[0] - 2), col : (col + padded.shape[1] - 2)]
				)

		# Make a nan array size of idat array
		D8_dir = np.full(idat["data"].shape, np.nan, dtype=float)

		# make a 9 x window array shape 3D-array
		allslopes = np.full([9, window[0].shape[0], window[0].shape[1]], np.nan)

		# For each of the eight directions, compute the tangent of the slope angle
		# from each cell in that direction
		for i in range(9):
			if i == 4:
				continue

			if i % 2 == 0:
				slope = (window[4] - window[i]) / (cellsize * np.sqrt(2))  # Diagonal neighbors
			else:
				slope = (window[4] - window[i]) / cellsize  # Adjacent neighbors

			if threshold_slope_degs and (abs(slope) < np.tan(np.radians(threshold_slope_degs))):
				allslopes[i] = 0
			else:
				allslopes[i] = slope

		D8_dir = np.nanargmax(allslopes, axis=0)

		out = idat.copy()
		out["data"] = D8_dir
		return out

	def _rasterise(
		self,
		shape_file,
		*,
		burn=1.0,
		band=1,
		ncols,
		nrows,
		cellsize,
		xllcorner,
		yllcorner,
		nodata,
		**kwargs,
	):
		"""Internal rasterisation to NumPy array with header info, using in-memory GDAL."""

		def _gdal_dtype_to_numpy(gdal_dtype):
			"""Map GDAL data type enum to NumPy dtype."""
			return {
				gdal.GDT_Byte: np.uint8,
				gdal.GDT_UInt16: np.uint16,
				gdal.GDT_Int16: np.int16,
				gdal.GDT_UInt32: np.uint32,
				gdal.GDT_Int32: np.int32,
				gdal.GDT_Float32: np.float32,
				gdal.GDT_Float64: np.float64,
			}.get(
				gdal_dtype, np.float32
			)  # Default to float32

		shape_file = Path(shape_file)
		fname = shape_file.stem

		# Compute extent
		xend = xllcorner + ncols * cellsize
		yend = yllcorner + nrows * cellsize

		# Use virtual memory
		tmp_vsimem = "/vsimem/tmp_raster.tif"

		# Rasterize to in-memory GeoTIFF
		gdal.Rasterize(
			destNameOrDestDS=tmp_vsimem,
			srcDS=str(shape_file),
			options=gdal.RasterizeOptions(
				format="GTiff",
				outputType=gdal.GDT_Float32,
				noData=nodata,
				initValues=[0.0],
				burnValues=[burn],
				creationOptions=["COMPRESS=LZW"],
				xRes=cellsize,
				yRes=cellsize,
				outputBounds=[xllcorner, yllcorner, xend, yend],
				layers=[fname],
			),
		)

		# Open vsimem dataset and read into memory
		ds = gdal.Open(tmp_vsimem)
		band = ds.GetRasterBand(band)

		cols = ds.RasterXSize
		rows = ds.RasterYSize

		gdal_dtype = band.DataType
		np_dtype = _gdal_dtype_to_numpy(gdal_dtype)

		raw_data = band.ReadRaster(0, 0, cols, rows, buf_type=gdal_dtype)

		arr = np.frombuffer(raw_data, dtype=np_dtype).reshape((rows, cols))

		# Apply scale and offset (if set)
		scale = band.GetScale() or 1.0
		offset = band.GetOffset() or 0.0
		arr = arr.astype(np.float32) * scale + offset

		# Fetch the WKT string
		wkt = ds.GetProjection()  # empty string if none

		ds = None  # cleanup
		gdal.Unlink(tmp_vsimem)
		
		odat = {
			"ncols": ncols,
			"nrows": nrows,
			"cellsize": cellsize,
			"xllcorner": xllcorner,
			"yllcorner": yllcorner,
			"nodata": nodata,
			"data": arr,
		}

		return odat, wkt

	def _project(self, input_file, input_crs=None, target_crs=None, datum="WGS84"):
		"""
		Raster to reproject (ASCII grids may lack CRS info in which case state it in input_crs).
		target_crs: CRS of the output raster (EPSG code, "EPSG:xxxx", "UTM33N", WKT, or PROJ string).
		input_crs: CRS of the input raster if not embedded (same formats as target_crs).
		datum : Geodetic datum, defaults to "WGS84".
		"""

		def utm_to_epsg(zone, hemisphere, datum="WGS84"):
			"""Return EPSG code for a given UTM zone & hemisphere."""
			srs = osr.SpatialReference()
			south = hemisphere.upper() == "S"
			srs.SetWellKnownGeogCS(datum)
			srs.SetUTM(zone, not south)  # True = North, False = South
			epsg_code = srs.GetAttrValue("AUTHORITY", 1)
			if epsg_code:
				return int(epsg_code)
			raise RuntimeError(f"Could not determine EPSG for UTM{zone}{hemisphere} ({datum})")

		def parse_crs(crs, datum="WGS84"):
			"""Helper: build osr.SpatialReference from EPSG/UTM/WKT/Proj4."""
			srs = osr.SpatialReference()
			if isinstance(crs, int):  # EPSG code
				srs.ImportFromEPSG(crs)
			elif isinstance(crs, str):
				if crs.upper().startswith("EPSG:"):
					srs.ImportFromEPSG(int(crs.split(":")[1]))
				elif crs.upper().startswith("UTM"):
					zone = int(crs[3:-1])
				hemisphere = crs[-1].upper()
				epsg = utm_to_epsg(zone, hemisphere, datum=datum)
				srs.ImportFromEPSG(epsg)
			else:
				# Try WKT or PROJ string
				if srs.ImportFromWkt(crs) != 0 and srs.ImportFromProj4(crs) != 0:
					raise ValueError(f"Unrecognized CRS: {crs}")
				else:
					raise TypeError("CRS must be int or str")
			return srs

		# Parse target CRS
		if target_crs is None:
			raise ValueError("No target CRS specified")
		srs_target = parse_crs(target_crs, datum=datum)
		dst_wkt = srs_target.ExportToWkt()

		# Open input raster
		src_ds = gdal.Open(str(input_file))
		if src_ds is None:
			raise RuntimeError(f"Failed to open raster: {input_file}")

		# Ensure source CRS is set - NB: input_crs does not overload we default to internal projection
		src_wkt = src_ds.GetProjection()
		if not src_wkt:  # empty string
			if input_crs is None:
				raise ValueError(
					f"Input raster has no CRS. Provide `input_crs` to continue. File: {input_file}"
				)
			srs_input = parse_crs(input_crs, datum=datum)
			src_wkt = srs_input.ExportToWkt()
			src_ds.SetProjection(src_wkt)  # assign manually

		# Reproject into memory
		mem_path = "/vsimem/reprojected.tif"
		gdal.Warp(mem_path, src_ds, dstSRS=dst_wkt, format="GTiff")
		dst_ds = gdal.Open(mem_path)

		# Export WKT
		wkt = dst_ds.GetProjection()  # empty string if none

		# Export to ASCII in memory
		asc_path = "/vsimem/output.asc"
		gdal.Translate(asc_path, dst_ds, format="AAIGrid")

		# Read ASCII text
		asc_fh = gdal.VSIFOpenL(asc_path, "rb")
		asc_stat = gdal.VSIStatL(asc_path)
		asc_bytes = gdal.VSIFReadL(1, asc_stat.size, asc_fh)
		gdal.VSIFCloseL(asc_fh)
		ascii_text = asc_bytes.decode("utf-8")

		# Parse metadata & data
		lines = ascii_text.strip().splitlines()
		header = {}
		data_start = 0
		for i, line in enumerate(lines):
			parts = line.strip().split()
			if len(parts) == 2 and parts[0].isalpha():
				header[parts[0].lower()] = float(parts[1])
			else:
				data_start = i
				break
		arr = np.loadtxt(io.StringIO("\n".join(lines[data_start:])), dtype=np.float32)

		# Cleanup
		gdal.Unlink(mem_path)
		gdal.Unlink(asc_path)

		return ({**header, "data": arr}, wkt)

	def _stats(self, dat, stat="all", mask_dat=None, **kwargs):
		"""Calculate some useful stats on raster data."""
		return RasterStats(self)(stat, dat, mask_dat=mask_dat, **kwargs)

	def _compstats(self, dat1, dat2, stat="all", mask_dat=None):
		"""Stats comparing one raster to another"""

		comp_stats = {"rmse", "corr", "abs_sum_diff"}

		# Spatially aligned difference
		diff = self._union(dat1, dat2, combine="diff")

		if (stat == "all") or (stat not in comp_stats):
			stats1 = self._stats(dat1, stat=stat, mask_dat=mask_dat)
			stats2 = self._stats(dat2, stat=stat, mask_dat=mask_dat)
			stats3 = self._stats(diff, stat=stat, mask_dat=mask_dat)

		if stat == "all":
			out = {k + "_1": v for k, v in stats1.items()}
			out.update({k + "_2": v for k, v in stats2.items()})
			out.update({k + "_diff": v for k, v in stats3.items()})
			out.update(
				{
					"rmse": np.sqrt(np.nanmean(diff["data"] ** 2)),
					"corr": np.corrcoef(dat1["data"].flatten(), dat2["data"].flatten())[0, 1],
					"abs_sum_diff": np.nansum(np.abs(diff["data"])),
				}
			)

		elif stat in comp_stats:
			if stat == "rmse":
				return np.sqrt(np.nanmean(diff["data"] ** 2))
			elif stat == "corr":
				return np.corrcoef(dat1["data"].flatten(), dat2["data"].flatten())[0, 1]
			elif stat == "abs_sum_diff":
				return np.nansum(np.abs(diff["data"]))
			else:
				raise ValueError(f"Unknown comparison stat {stat}")

		else:
			out = {
				f"{stat}_1": stats1,
				f"{stat}_2": stats2,
				f"{stat}_diff": stats3,
			}

		return out

	def _recipe(self, recipe, datas, **kwargs):
		"""Run a recipe on a raster structure"""
		return Recipes(self)(recipe, *datas, **kwargs)


class RasterMethods(RasterOperations):

	_meta = {
		"abs": {
			"args": {
				"input_file": {"type": "raster", "label": "Input Raster"},
			},
			"description": "Compute the absolute value raster",
		},
		"copy": {
			"args": {
				"input_file": {"type": "raster", "label": "Input Raster"},
			},
			"description": "Make an exact copy of a raster",
		},
		"zeros_like": {
			"args": {
				"input_file": {"type": "raster", "label": "Template Raster"},
			},
			"description": "Create a zero-valued raster with same dimensions as input",
		},
		"ones_like": {
			"args": {
				"input_file": {"type": "raster", "label": "Template Raster"},
			},
			"description": "Create a ones-valued raster with same dimensions as input",
		},
		"full_like": {
			"args": {
				"input_file": {"type": "raster", "label": "Template Raster"},
				"fill_val": {"type": "float", "label": "Fill Value", "default": 1.0},
			},
			"description": "Create a raster filled with a constant value",
		},
		"swap": {
			"args": {
				"input_file": {"type": "raster", "label": "Input Raster"},
				"from_val": {"type": "float", "label": "From Value"},
				"to_val": {"type": "float", "label": "To Value"},
			},
			"description": "Replace occurrences of one value with another",
		},
		"threshold": {
			"args": {
				"input_file": {"type": "raster", "label": "Input Raster"},
				"cond_arr": {"type": "raster-list", "label": "Condition Raster(s)"},
				"cond_op": {"type": "string-list", "label": "Comparison Operators"},
				"cond_val": {"type": "float-list", "label": "Comparison Values"},
				"bitop": {"type": "string-list", "label": "Boolean Operators", "optional": True},
				"threshold": {"type": "float", "label": "Threshold Value", "optional": True},
				"fill": {"type": "float", "label": "Fill Value", "optional": True},
			},
			"description": "Apply conditional raster thresholding using logical operations",
		},
		"clip": {
			"args": {
				"input_file": {"type": "raster", "label": "Input Raster"},
				"xll": {"type": "float", "label": "Lower Left X"},
				"yll": {"type": "float", "label": "Lower Left Y"},
				"xur": {"type": "float", "label": "Upper Right X"},
				"yur": {"type": "float", "label": "Upper Right Y"},
			},
			"description": "Clip raster to new bounds",
		},
		"extend": {
			"args": {
				"input_file": {"type": "raster", "label": "Input Raster"},
				"xll": {"type": "float", "label": "Lower Left X"},
				"yll": {"type": "float", "label": "Lower Left Y"},
				"xur": {"type": "float", "label": "Upper Right X"},
				"yur": {"type": "float", "label": "Upper Right Y"},
				"fill_value": {"type": "float", "label": "Fill Value", "default": 0.0},
			},
			"description": "Extend raster to new bounds",
		},
		"match": {
			"args": {
				"input_file": {"type": "raster", "label": "Input Raster"},
				"template_file": {"type": "raster", "label": "Template Raster"},
				"fill_value": {"type": "float", "label": "Fill Value", "default": 0.0},
			},
			"description": "Match raster extent and resolution to a template",
		},
		"resample": {
			"args": {
				"input_file": {"type": "raster", "label": "Input Raster"},
				"target_cellsize": {"type": "float", "label": "Target Cell Size"},
				"order": {"type": "int", "label": "Interpolation Order", "default": 1},
			},
			"description": "Resample raster to a new resolution",
		},
		"scale": {
			"args": {
				"input_file": {"type": "raster", "label": "Input Raster"},
				"factor": {"type": "float", "label": "Scale Factor", "default": 1.0},
			},
			"description": "Multiply raster values by a constant factor",
		},
		"normalise": {
			"args": {
				"input_file": {"type": "raster", "label": "Input Raster"},
				"method": {
					"type": "enum",
					"label": "Normalisation Method",
					"choices": Normaliser._available(),
					"default": "minmax",
				},
			},
			"description": "Normalise raster values using different methods",
		},
		"rotate": {
			"args": {
				"input_file": {"type": "raster", "label": "Input Raster"},
				"angle": {"type": "float", "label": "Rotation Angle (deg)"},
#				rotationangle, rotationpoint=None, rotationaxis="z", clip=True
			},
			"description": "Rotate a raster around an axis",
		},
		"band": {
			"args": {
				"input_file": {"type": "raster", "label": "Input Raster"},
				"angle": {"type": "int", "label": "Band to select"},
			},
			"description": "Isolate a specific integer band in a raster",
		},
		"union": {
			"args": {
				"input_file_1": {"type": "raster", "label": "Input Raster"},
				"input_file_2": {"type": "raster", "label": "Input Raster"},
				"combine": {
					"type": "enum",
					"label": "Combine Mode",
					"choices": Combine._available(),
					"default": "max",
				},
			},
			"description": "Create a raster combining all input extents (union)",
		},
		"intersection": {
			"args": {
				"input_file_1": {"type": "raster", "label": "Input Raster"},
				"input_file_2": {"type": "raster", "label": "Input Raster"},
				"combine": {
					"type": "enum",
					"label": "Combine Mode",
					"choices": Combine._available(),
					"default": "max",
				},
			},
			"description": "Intersect overlapping regions of rasters",
		},
		"mask": {
			"args": {
				"input_file": {"type": "raster", "label": "Input Raster"},
				"mask_file": {"type": "raster", "label": "Mask Raster"},
				"fill_value": {"type": "float", "label": "Fill value"},
			},
			"description": "Intersect overlapping regions of rasters",
		},
		# For this we need to expand the recipes...
		"recipe": {
			"args": {
				"recipe_name": {
					"type": "enum",
					"label": "Choose recipe",
					"choices": Recipes._available(),
					"default": "treecount2nD",
				},
				"inputfiles": {"type": "raster-list", "label": "Recipe Rasters"},
			},
			"description": "Intersect overlapping regions of rasters",
		},
		"noise": {
			"args": {
				"input_file": {"type": "raster", "label": "Input Raster"},
				"mode": {
					"type": "enum",
					"label": "Noise Type",
					"choices": NoiseAdder._available(),
					"default": "normal",
				},
				"seed": {"type": "int", "label": "Random Seed", "optional": True},
			},
			"description": "Add synthetic noise to raster data",
		},
		"slope_deg": {
			"args": {
				"input_file": {"type": "raster", "name": "input_file", "label": "Input Raster"},
			},
			"description": "Compute slope in degrees from a raster (DEM).",
		},
		"aspect_deg": {
			"args": {
				"input_file": {"type": "raster", "name": "input_file", "label": "Input Raster"},
			},
			"description": "Compute aspect in degrees from a raster (DEM).",
		},
		"curvature": {
			"args": {
				"input_file": {"type": "raster", "name": "input_file", "label": "Input Raster"},
				"method": {
					"type": "enum",
					"label": "Curvature Method",
					"choices": Curvature._available(),
					"default": "planform",
				}
			},
			"description": "Compute curvature raster for a variety of curvature definitions.",
		},
		"D8": {
			"args": {
				"input_file": {"type": "raster", "name": "input_file", "label": "Input Raster"},
			},
			"description": (
				"Compute D8 flow direction raster (NAKSIN convention). "
				"Direction codes: 0=NW, 1=N, 2=NE, 3=E, 4=None, 5=SE, 6=S, 7=SW, 8=W."
			),
		},
		"derivative": {
			"args": {
				"input_file": {"type": "raster", "name": "input_file", "label": "Input Raster"},
				"dir_str": {
					"type": "enum",
					"label": "Direction",
					"choices": ["x", "y", "xy"],
					"default": "x",
				},
			},
			"description": "Compute the spatial derivative in a chosen coordinate direction.",
		},
		"rasterise": {
			"args": {
				"shape_file": {"type": "vector", "label": "Input Shapefile"},
				"template_file": {"type": "raster", "label": "Template Raster", "optional": True},
				"band": {"type": "int", "label": "Band Index", "default": 1},
				"burn": {"type": "float", "label": "Burn Value", "default": 1.0},
			},
			"description": "Rasterise a vector dataset using a template raster",
		},
	}

	def __init__(self):
		super().__init__()
		self.reader = Reader()
		self.writer = Writer()

	@classmethod
	def _available(cls):
		"""List callable public methods excluding __call__."""
		return [
			m
			for m in dir(cls)
			if not m.startswith("_") and callable(getattr(cls, m)) and m != "__call__"
		]

	def abs(self, input_file, output_file):
		"""Compute the element-wise absolute value of the raster array"""
		dat = self.reader.read(input_file)
		projection = self.reader.get_projection_info(input_file)
		odat = self._abs(dat)
		self.writer.write(output_file, odat, projection=projection)

	def copy(self, input_file, output_file):
		"""Make a copy of the raster"""
		dat = self.reader.read(input_file)
		projection = self.reader.get_projection_info(input_file)
		self.writer.write(output_file, dat, projection=projection)

	def zeros_like(self, input_file, output_file):
		"""Make an all zeros raster with matching metadata to input"""
		dat = self.reader.read_head(input_file)
		projection = self.reader.get_projection_info(input_file)
		dat["data"] = np.zeros((dat["nrows"], dat["ncols"]))
		self.writer.write(output_file, dat, projection=projection)

	def ones_like(self, input_file, output_file):
		"""Make an all ones raster with matching metadata to input"""
		dat = self.reader.read_head(input_file)
		projection = self.reader.get_projection_info(input_file)
		dat["data"] = np.ones((dat["nrows"], dat["ncols"]))
		self.writer.write(output_file, dat, projection=projection)

	def full_like(self, input_file, output_file, fill_val):
		"""Make an all fill_val raster with matching metadata to input"""
		dat = self.reader.read_head(input_file)
		projection = self.reader.get_projection_info(input_file)
		dat["data"] = np.full((dat["nrows"], dat["ncols"]), fill_val)
		self.writer.write(output_file, dat, projection=projection)

	def swap(self, input_file, output_file, from_val, to_val):
		"""
		Swap a value in the array e.g. change the height in a release area or change
		NaN value from -9999 to 0 etc.
		"""
		dat = self.reader.read(input_file)
		projection = self.reader.get_projection_info(input_file)
		odat = self._swap(dat, from_val, to_val)
		self.writer.write(output_file, odat, projection=projection)

	def threshold(
		self,
		input_file,
		output_file,
		cond_arr,
		cond_op,
		cond_val,
		bitop=[],
		threshold=None,
		fill=None,
	):
		"""
		Construct np.where condition(s) from arrays, comparison operators, and boolean logic using rasters
		
		# Single condition: Make all values > 0 = 1 
		threshold dem.asc \
		--cond_arr dem.asc \
		--cond_op ">" \
		--cond_val 0
		--threshold 1 \
		-o out.asc

		# Multiple conditions with boolean AND: highlight potential release areas where DEM > 500m and slope_angle > 30
		threshold dem.asc \
		--cond_arr self --cond_op ">" --cond_val 500 \
		--bitop "&" \
		--cond_arr slope.asc --cond_op ">" --cond_val 30 \
		--threshold 1.5
		-o out.asc
		"""
		dat = self.reader.read(input_file)
		projection = self.reader.get_projection_info(input_file)

		# Read all the condition raster arrays
		cond_arrs = [self.reader.read(cf) if cf != "self" else dat for cf in cond_arr]

		# NB: condition compare values could be arrays but how best to detect I guess they will either be Path or string...
		cond_vals = [
			cv if isinstance(cv, (Path, str)) else self.reader.read(str(cv)) for cv in cond_val
		]

		odat = self._threshold(
			dat, cond_arrs, cond_op, cond_vals, bitop=bitop, threshold_val=threshold, fill_val=fill
		)
		self.writer.write(output_file, odat, projection=projection)

	def clip(self, input_file, output_file, xll=None, yll=None, xur=None, yur=None):
		"""Clip raster to new bounds. Fails if the window is outside the raster domain."""
		# Read or use provided data
		dat = self.reader.read(input_file)
		projection = self.reader.get_projection_info(input_file)
		odat = self._clip(dat, xll, yll, xur, yur)
		self.writer.write(output_file, odat, projection=projection)

	def extend(
		self,
		input_file,
		output_file,
		xll=None,
		yll=None,
		xur=None,
		yur=None,
		fill_value=0.0,
		writeout=True,
	):
		"""Extend raster to new bounds, filling new cells with `fill_value`."""

		dat = self.reader.read(input_file)
		projection = self.reader.get_projection_info(input_file)
		odat = self._extend(dat, xll, yll, xur, yur)
		self.writer.write(output_file, odat, projection=projection)

	def match(self, input_file, template_file, output_file, fill_value=0.0):
		"""Match raster extent to a template"""
		tdat = self.reader.read(template_file)
		projection = self.reader.get_projection_info(template_file)
		idat = self.reader.read(input_file)
		odat = self._match(tdat, idat)
		self.writer.write(output_file, odat, projection=projection)

	def resample(self, input_file, output_file, target_cellsize, order=1):
		"""Resample raster to match a new resolution (cellsize), using interpolation."""
		dat = self.reader.read(input_file)
		projection = self.reader.get_projection_info(template_file)
		odat = self._resample(dat, target_cellsize, order=order)
		self.writer.write(output_file, odat, projection=projection)

	def scale(self, input_file, output_file, factor=1):
		"""Scale raster values by factor (ignoring nodata)."""
		dat = self.reader.read(input_file)
		projection = self.reader.get_projection_info(input_file)
		odat = self._scale(dat, factor)
		self.writer.write(output_file, odat, projection=projection)

	def normalise(self, input_file, output_file, method="minmax", **kwargs):
		"""Scale raster values by factor (ignoring nodata)."""
		dat = self.reader.read(input_file)
		projection = self.reader.get_projection_info(input_file)
		odat = self._normalise(dat, method=method, **kwargs)
		self.writer.write(output_file, odat, projection=projection)

	def rotate(self, input_file, output_file, **kwargs):
		"""
		Rotate a raster around an arbitrary axis in 3D handling nodata
		rotationangle, rotationpoint=None, rotationaxis="z", clip=True
		"""
		dat = self.reader.read(input_file)
		projection = self.reader.get_projection_info(input_file)
		odat = self._rotate(dat, **kwargs)
		self.writer.write(output_file, odat, projection=projection)

	def band(self, input_file, output_file, val=None):
		"""Split up a raster into bands"""
		dat = self.reader.read(input_file)
		projection = self.reader.get_projection_info(input_file)
		all_vals = {v for v in dat["data"].flatten()}

		outfile = Path(output_file)
		suffix = outfile.suffix

		if val is None:
			for i, v in enumerate(all_vals):
				odat = self._split(dat, val=v)
				self.writer.write(
					outfile.removesuffix(suffix) + f"_{i}.{suffix}", odat, projection=projection
				)
		elif (isinstance(val, tuple) or isinstance(val, list)) and val[0] <= v < val[1]:
			vals = [v for v in all_vals if val[0] <= v < val[1]]
			for v in vals:
				odat = self._split(dat, val=v)
				self.writer.write(
					outfile.removesuffix(suffix) + f"_{i}.{suffix}", odat, projection=projection
				)
		elif isinstance(val, int) or isinstance(val, float):
			vals = [v for v in all_vals if np.isclose(v)]
			if vals != []:
				odat = self._split(dat, val=vals[0])
			else:
				print(f"No band found with value {val}")
		else:
			raise RuntimeError("Type of val {type(val)} is not handled directly.")

	def union(self, files, output_file, fill_value=0.0, combine='max'):
		"""Create a single raster with union of both input domains."""
		if len(files) < 2:
			raise RuntimeError("Combine method 'union' requires more than one input file!")

		projection = self.reader.get_projection_info(files[0])

		r1 = self.reader.read(files[0])
		r2 = self.reader.read(files[1])
		odat = self._union(r1, r2, fill_value=fill_value, combine=combine)

		if len(files) > 2:
			for f in files[2:]:
				r = self.reader.read(f)
				odat = self._union(odat, r, fill_value=fill_value, combine=combine)

		self.writer.write(output_file, odat, projection=projection)

	def intersection(self, files, output_file, fill_value=0.0, combine='max'):
		"""Crop both rasters to overlapping valid region and combine them."""
		if len(files) < 2:
			raise RuntimeError("Combine method 'intersection' requires more than one input file!")

		projection = self.reader.get_projection_info(files[0])

		r1 = self.reader.read(files[0])
		r2 = self.reader.read(files[1])
		odat = self._intersection(r1, r2, fill_value=fill_value, combine=combine)

		if len(files) > 2:
			for f in files[2:]:
				r = self.reader.read(f)
				odat = self._intersection(odat, r, fill_value=fill_value, combine=combine)

		self.writer.write(output_file, odat, projection=projection)

	def mask(self, input_file, mask_files, output_file, fill_value=0.0, combine='max'):
		"""Mask raster 1 with the valid region of raster 2"""

		if len(files) < 2:
			raise RuntimeError("Combine method 'mask' requires more than one input file!")

		projection = self.reader.get_projection_info(input_file)

		dat = self.reader.read(input_file)
		mask = self.reader.read(mask_files[0])
		odat = self._mask(dat, mask, fill_value=fill_value)

		if len(mask_files) > 1:
			for f in mask_files[1:]:
				m = self.reader.read(f)
				odat = self._mask(odat, m, fill_value=fill_value)

		self.writer.write(output_file, odat, projection=projection)

	def recipe(self, recipe_name, output_file, **kwargs):
		"""Choose and run a recipe"""
		
		recipes = Recipes._available()
		if recipe_name not in recipes:
			raise RuntimeError(f"Recipe name {recipe_name} unknown. Choose from {recipes}.")
		
		# Extract and read only the readable file paths
		datas = []
		newkwargs = {}
		flag = False
		projection = None
		for k,v in kwargs.items():
			try: 
				datas.append(self.reader.read(v))
				flag = True
				if flag: 
					projection = self.reader.get_projection_info(v)
			except RuntimeError: 
				newkwargs.setdefault(k, v)
		
		odat = self._recipe(recipe_name, datas, **newkwargs)
		self.writer.write(output_file, odat, projection=projection)

	def noise(self, input_file, output_file, mode="normal", seed=None, **kwargs):
		"""Add noise to raster using a variety of methods"""
		dat = self.reader.read(input_file)
		projection = self.reader.get_projection_info(input_file)
		odat = self._noise(dat, mode, seed, **kwargs)
		self.writer.write(output_file, odat, projection=projection)

	def slope_deg(self, input_file, output_file, **kwargs):
		"""Compute the slope in degress"""
		dat = self.reader.read(input_file)
		projection = self.reader.get_projection_info(input_file)
		odat = self._slope(dat, **kwargs)
		self.writer.write(output_file, odat, projection=projection)

	def aspect_deg(self, input_file, output_file, **kwargs):
		"""Compute the aspect in degress assuming raster aligned N = 0 deg"""
		dat = self.reader.read(input_file)
		projection = self.reader.get_projection_info(input_file)
		odat = self._aspect(dat, **kwargs)
		self.writer.write(output_file, odat, projection=projection)

	def curvature(self, input_file, output_file, method="planform", **kwargs):
		"""Compute curvature raster for a variety of curvature definitions"""
		dat = self.reader.read(input_file)
		projection = self.reader.get_projection_info(input_file)
		odat = self._curvature(dat, method, **kwargs)
		self.writer.write(output_file, odat, projection=projection)

	def D8(self, input_file, output_file, **kwargs):
		"""
		Compute the D8 direction raster using numbering convention from NAKSIN
		Direction map: 0=NW, 1=N, 2=NE, 3=E, 4=None, 5=SE, 6=S, 7=SW, 8=W
		Picks the steepest *downslope* neighbor. If none downslope -> 4.
		"""
		dat = self.reader.read(input_file)
		projection = self.reader.get_projection_info(input_file)
		odat = self._D8(dat, **kwargs)
		self.writer.write(output_file, odat, projection=projection)

	def stats(
		self, input_file, output_file=None, stat="all", mask=None, compare=None, fill_value=0.0
	):
		"""
		Provide a selection of useful stats about raster(s).
		mode="single"   → stats for one raster
		mode="compare"  → stats for two rasters i.e. intercomparison
		"""

		comparison_stats = {'rmse', 'corr', 'abs_sum_diff'}

		if compare is not None:
			mode = "compare"
		else:
			mode = "single"

		if mode == "single":

			if stat in comparison_stats:
				raise ValueError(f"Stat {stat} is only available in compare mode.")

			dat = self.reader.read(input_file)

			if mask:
				dat["mask"] = mask
				mask = self.reader.read(mask)
			else:
				# NB: Add a placeholder for mask in dat structure when there isn't one
				dat["mask"] = "-"

			odat = self._stats(dat, stat=stat, mask_dat=mask)

		elif mode == "compare":

			# Read both rasters
			dat1 = self.reader.read(input_file)
			dat2 = self.reader.read(compare)

			if mask:
				mask = self.reader.read(mask)

			# Ensure same shape
			if dat1["data"].shape != dat2["data"].shape:
				raise ValueError("Raster shapes do not match for compare mode")

			odat = self._compstats(dat1, dat2, stat=stat, mask_dat=mask)

		else:
			raise ValueError(f"Unknown mode: {mode}")

		# Output
		if output_file:
			outstr = "\n".join([f"{k} {v}" for k, v in odat.items()])
			with open(output_file, "w") as fid:
				fid.write(outstr)
		else:
			if stat == "all":
				pprint({key: odat[key] for key in sorted(odat, key=str.lower)})
			else:
				if isinstance(odat, dict):
					print(f"{stat} = ")
					pprint(odat)
				else:
					print(f"{stat} = {odat}")

	def derivative(self, input_file, output_file, dir_str, **kwargs):
		"""Calculate the derivative in a coordinate direction"""

		dat = self.reader.read(input_file)
		projection = self.reader.get_projection_info(input_file)
		odat = self._derivative(dat, dir_str, **kwargs)
		self.writer.write(output_file, odat, projection=projection)

	def rasterise(self, shape_file, output_file, template_file=None, writeproj=False, **kwargs):
		"""Rasterise a shape file into a raster file"""

		if template_file:
			headerdat = self.reader.read_head(template_file)
		else:
			headerdat = {
				"ncols": kwargs.get(ncols),
				"nrows": kwargs.get(nrows),
				"cellsize": kwargs.get(cellsize),
				"xllcorner": kwargs.get(xllcorner),
				"yllcorner": kwargs.get(yllcorner),
				"nodata": kwargs.get(nodata),
			}

		odat, owkt = self._rasterise(
			shape_file, band=kwargs.get("band", 1), burn=kwargs.get("burn", 1.0), **headerdat
		)

		self.writer.write(output_file, odat, projection=owkt)

		if writeproj:
			oproj_file = Path(output_file).with_suffix(".prj")
			with open(oproj_file, "w") as fid:
				fid.write(owkt)

	def reproject(
		self,
		input_file,
		output_file,
		input_crs=None,
		target_crs=None,
		input_projection_file=None,
		target_projection_file=None,
		datum=None,
		writeproj=None,
		**kwargs,
	):
		"""
		Reproject the raster using an EPSG code, a .prj file, or strings like 'UTM33N' or 'EPSG:33633'.
		projection_file: Path to a .prj file containing the target projection.
		target_crs: EPSG code, UTM zone string (e.g. 'UTM33N'), 'EPSG:4326', or WKT/PROJ string.
		"""
		if input_projection_file:
			srs = osr.SpatialReference()
			if srs.ImportFromESRI([open(input_projection_file).read()]) != 0:
				raise RuntimeError(f"Failed to read projection from {input_projection_file}")
			input_crs = srs.ExportToWkt()

		if target_projection_file:
			srs = osr.SpatialReference()
			if srs.ImportFromESRI([open(target_projection_file).read()]) != 0:
				raise RuntimeError(f"Failed to read projection from {target_projection_file}")
			target_crs = srs.ExportToWkt()

		odat, owkt = self._project(
			input_file, input_crs=input_crs, target_crs=target_crs, datum=datum
		)
		self.writer.write(output_file, odat)

		if writeproj:
			oproj_file = Path(output_file).with_suffix(".prj")
			with open(oproj_file, "w") as fid:
				fid.write(owkt)
