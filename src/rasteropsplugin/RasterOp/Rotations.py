#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image rotation methods

TODO: Ensure the realworld extent calculatons keep track of the realworld coordinates -
I think currently it rotates the scene but doesn't follow the coords - thus coords are wrong
after rotation...
"""

import numpy as np
import cv2


class Rotator:
	def __init__(self, nodata=None, nodata_strat="mask-aware"):
		self.nodata = nodata
		if nodata_strat in {"mask-aware", "nearest", "dilate", "inpaint"}:
			self.nodata_strat = nodata_strat
		else:
			raise ValueError(f"Nodata handling strategy {nodata_strat} unavailable.")

		self.axis_map = {
			"x": np.array([1, 0, 0], dtype=float),
			"y": np.array([0, 1, 0], dtype=float),
			"z": np.array([0, 0, 1], dtype=float),
			"xy": np.array([1, 1, 0], dtype=float),
			"xz": np.array([1, 0, 1], dtype=float),
			"yz": np.array([0, 1, 1], dtype=float),
		}

	def _normalize_axis(self, rotationaxis):
		"""Map string or vector to normalized 3D axis."""
		if isinstance(rotationaxis, str):
			if rotationaxis.lower() not in self.axis_map:
				raise ValueError(
					f"Unknown axis string {rotationaxis}. Choose from {list(self.axis_map.keys())}"
				)
			axis = self.axis_map[rotationaxis.lower()]
		else:
			axis = np.array(rotationaxis, dtype=float)
			if axis.shape != (3,):
				raise ValueError("rotationaxis must be string shorthand or len(3) vector")

		n = np.linalg.norm(axis)
		if not np.isfinite(n) or n == 0:
			raise ValueError("rotationaxis must be non-zero and finite")
		return axis / n

	def _rotation_matrix(self, axis, angle_deg):
		"""Rodrigues' rotation formula."""
		theta = np.radians(angle_deg)
		cos_t, sin_t = np.cos(theta), np.sin(theta)
		axis = axis / np.linalg.norm(axis)
		K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
		R = cos_t * np.eye(3) + sin_t * K + (1 - cos_t) * np.outer(axis, axis)
		return R

	def _world_to_pixel(self, meta, x, y):
		"""Translate (x,y) world coordinate to pixel (row, col)"""

		# Fetch the raster metadata
		xll, yll = meta["xllcorner"], meta["yllcorner"]
		cs = meta["cellsize"]
		nrows, ncols = meta["nrows"], meta["ncols"]

		col = int(np.floor((x - xll) / cs))
		row = int(
			np.floor((yll + nrows * cs - y) / cs)
		)  # world y increases up; image rows increase down
		return row, col

	def _pixel_to_world(self, meta, row, col):
		"""Translate pixel (row, col) to world coordinates (center of cell)"""

		# Fetch the raster metadata
		xll, yll = meta["xllcorner"], meta["yllcorner"]
		cs = meta["cellsize"]
		nrows, ncols = meta["nrows"], meta["ncols"]

		x = xll + (col + 0.5) * cs  # center of cell
		y = yll + (nrows - (row + 0.5)) * cs
		return x, y

	def _image_corners(self, h, w):
		"""Corners in pixel coordinates (inclusive-exclusive grid origin at (0,0))"""
		return np.array([[0, 0], [w, 0], [0, h], [w, h]], dtype=float)

	def _valid_mask(self, image):
		if self.nodata is None:
			return np.ones_like(image, dtype=np.uint8)
		if np.isnan(self.nodata):
			return (~np.isnan(image)).astype(np.uint8)
		return (image != self.nodata).astype(np.uint8)

	def _apply_remap(self, image, map_x, map_y):
		"""
		Apply remap strategy - methods such that nodata never participates in interpolation
		- only valid pixels contribute. Different tradeoffs in accuracy, expense and prettyness.
		{mask-aware, nearest, dilate, inpaint}
		"""
		strat = self.nodata_strat
		if strat == "mask-aware":
			return self._apply_remap_maskaware(image, map_x, map_y)
		elif strat == "nearest":
			return self._apply_remap_nearest(image, map_x, map_y)
		elif strat == "dilate":
			return self._apply_remap_dilate(image, map_x, map_y)
		elif strat == "inpaint":
			return self._apply_remap_inpaint(image, map_x, map_y)
		else:
			raise ValueError(f"Unknown nodata strategy {strat}")

	def _apply_remap_maskaware(self, image, map_x, map_y):
		"""
		Apply remap with nodata handling via mask-aware interpolation.
		Same method as used in raserio / gdal / skimage etc.
		"""
		mask = (image != self.nodata).astype(np.uint8)

		# remap both data and mask
		warped_data = cv2.remap(
			image.astype(np.float32),
			map_x.astype(np.float32),
			map_y.astype(np.float32),
			interpolation=cv2.INTER_LINEAR,
			borderMode=cv2.BORDER_CONSTANT,
			borderValue=self.nodata if self.nodata is not None else 0.0,
		)
		warped_mask = cv2.remap(
			mask.astype(np.float32),
			map_x.astype(np.float32),
			map_y.astype(np.float32),
			interpolation=cv2.INTER_LINEAR,
			borderMode=cv2.BORDER_CONSTANT,
			borderValue=0,
		)

		# normalise where mask > 0
		out = np.full_like(
			warped_data, np.nan if self.nodata is None else float(self.nodata), dtype=np.float32
		)
		valid = warped_mask > 1e-6

		# NB: This cancels out any contribution from nodata pixels, since they were zeros in the mask.
		# Pixels where the mask → 0 are set back to nodata.
		out[valid] = warped_data[valid] / np.maximum(warped_mask[valid], 1e-6)
		if self.nodata is None:
			return out  # float with NaNs
		return out.astype(image.dtype)

	def _apply_remap_nearest(self, image, map_x, map_y):
		"""Apply remap with nodata handling via nearest neighbour interpolation. Safe but jagged"""
		return cv2.remap(
			image,
			map_x,
			map_y,
			interpolation=cv2.INTER_NEAREST,
			borderMode=cv2.BORDER_CONSTANT,
			borderValue=float(self.nodata) if self.nodata is not None else 0.0,
		)

	def _apply_remap_dilate(self, image, map_x, map_y):
		"""
		Apply remap with nodata handling via dilation method. Conservative mask clipping
		if nodata needs to be v. strict
		"""

		# Rotate normally with mask-aware to avoid leakage
		out = self._apply_remap_maskaware(image, map_x, map_y)
		if self.nodata is None:
			return out

		# Dilate nodata zones outward by 1 pixel (could make radius configurable)
		mask = (out != self.nodata).astype(np.uint8)
		dilated = cv2.dilate(mask, np.ones((3, 3), np.uint8))
		out[dilated == 0] = self.nodata
		return out

	def _apply_remap_inpaint(self, image, map_x, map_y):
		"""
		Apply remap with nodata handling via inpaint extrapolation.
		Expensive but 'pretty' - adds made up values...
		"""
		filled = image.copy()
		m = self._valid_mask(image) == 0
		if m.any():
			# OpenCV inpaint works on 8-bit; convert pragmatically
			src = filled.astype(np.float32)
			src_norm = src
			if src.dtype != np.uint8:
				# scale to 0-255 robustly
				finite = np.isfinite(src)
				if finite.any():
					lo, hi = np.percentile(src[finite], [1, 99])
					if hi <= lo:
						hi = lo + 1.0
					src_norm = np.clip((src - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
				else:
					src_norm = np.zeros_like(src, dtype=np.uint8)
			mask8 = m.astype(np.uint8) * 255
			src_filled = cv2.inpaint(src_norm, mask8, 3, cv2.INPAINT_NS)
			# Bring back to float32 scale if needed
			if src.dtype != np.uint8:
				src_filled = src_filled.astype(np.float32) * (hi - lo) / 255.0 + lo
			filled = src_filled.astype(image.dtype)

		# rotate
		rotated = cv2.remap(
			filled.astype(np.float32),
			map_x,
			map_y,
			interpolation=cv2.INTER_LINEAR,
			borderMode=cv2.BORDER_CONSTANT,
			borderValue=float(self.nodata) if self.nodata is not None else 0.0,
		).astype(image.dtype)

		# restore nodata where invalid
		rotated_mask = cv2.remap(
			self._valid_mask(image).astype(np.uint8),
			map_x,
			map_y,
			interpolation=cv2.INTER_NEAREST,
			borderMode=cv2.BORDER_CONSTANT,
			borderValue=0,
		)
		if self.nodata is None:
			rotated = rotated.astype(np.float32)
			rotated[rotated_mask == 0] = np.nan
			return rotated
		rotated[rotated_mask == 0] = self.nodata
		return rotated

	def _rotate_pt(self, point, rotationangle, rotationpoint=(0, 0, 0), rotationaxis="z"):
		"""
		Rotate a 3D point/vector about an arbitrary axis and point. Useful if we want to
		rotate a known axis to be the rotation axis e.g. rotate [0,1,0] by 27 degrees about z then
		rotate the image around that etc.
		"""
		axis = self._normalize_axis(rotationaxis)
		R = self._rotation_matrix(axis, rotationangle)
		p = np.array(point, dtype=float) - np.array(rotationpoint, dtype=float)
		rotated = R @ p + np.array(rotationpoint, dtype=float)
		return rotated

	def Rotate(self, dat, rotationangle, rotationpoint=None, rotationaxis="z", clip=True):
		"""
		Rotate a raster about an arbitrary axis in 3D space and project back.
		- dat: ASCII grid type - e.g. dict {**metadata, data arr}
		- rotationangle: in degrees
		- rotationpoint: (x,y,z) tuple (defaults to image center in xy / z)
		- rotationaxis: str ('x','y','z') or len(3) vector
		- clip: bool (True keeps same size as input, False expands/shrinks to fit rotated image)
		- metadata: headerinfo with {"xllcorner","yllcorner","cellsize","ncols","nrows"}
		"""

		if rotationangle % 360 == 0:
			return dat

		image = dat["data"]
		if image.ndim != 2:
			raise ValueError("Rotator expects a single-band raster array.")

		h, w = image.shape[:2]

		# Resolve rotation point in pixel coordinates (col,row). We track world for metadata if not clip.
		if rotationpoint is None or (isinstance(rotationpoint, str) and rotationpoint == "center"):
			cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
			rot_pt_px = np.array([cx, cy, 0.0])
			rot_pt_world = self._pixel_to_world(dat, cy, cx)
		elif isinstance(rotationpoint, str):
			pos = rotationpoint.lower()
			pt_map = {
				"tl": (0.0, 0.0),
				"tr": (w - 1.0, 0.0),
				"bl": (0.0, h - 1.0),
				"br": (w - 1.0, h - 1.0),
				"center": ((w - 1) / 2.0, (h - 1) / 2.0),
			}
			if pos not in pt_map:
				raise ValueError("rotationpoint string must be one of {center, tl, tr, bl, br}")
			cx, cy = pt_map[pos]
			rot_pt_px = np.array([cx, cy, 0.0])
			rot_pt_world = self._pixel_to_world(dat, cy, cx)
		else:
			# world coordinates provided
			wp = np.array(rotationpoint, dtype=float)
			if wp.size not in (2, 3):
				raise ValueError("rotationpoint must be a string or (x,y) or (x,y,z)")
			if wp.size == 2:
				xw, yw, zw = float(wp[0]), float(wp[1]), 0.0  # z treated as surface
			else:
				xw, yw, zw = float(wp)

			# zw is absolute elevation - interpret z-distance above surface
			row, col = self._world_to_pixel(dat, xw, yw)
			rot_pt_px = np.array([float(col), float(row), 0.0])
			rot_pt_world = (xw, yw)

		axis = self._normalize_axis(rotationaxis)

		# 2D cv2 rotation (axis z)
		if np.allclose(axis, np.array([0.0, 0.0, 1.0]), atol=1e-9):

			R = cv2.getRotationMatrix2D((rot_pt_px[0], rot_pt_px[1]), rotationangle, 1.0)

			if clip:
				# Inverse map over the same canvas size
				grid_x, grid_y = np.meshgrid(
					np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
				)
				ones = np.ones_like(grid_x)
				dst = np.stack([grid_x, grid_y, ones], axis=0).reshape(3, -1)
				R_inv = cv2.invertAffineTransform(R)
				src = (R_inv @ dst).reshape(2, h, w)
				map_x, map_y = src[0], src[1]
				rot_arr = self._apply_remap(image, map_x, map_y)

				out = dat.copy()
				out["data"] = rot_arr
				return out
			else:
				# Compute rotated bounding box of image corners (in pixel space)
				corners = self._image_corners(h, w)  # (4,2)

				# Apply Rotation
				corners_h = np.hstack([corners, np.ones((4, 1), dtype=np.float32)])  # (4,3)
				rot_c = (R @ corners_h.T).T  # (4,2)
				min_x, min_y = np.floor(rot_c.min(axis=0))
				max_x, max_y = np.ceil(rot_c.max(axis=0))
				new_w, new_h = int(max_x - min_x), int(max_y - min_y)
				if new_w <= 0 or new_h <= 0:
					raise RuntimeError("Degenerate output size after rotation.")

				# Shift so output starts at (0,0)
				shift = np.array([[1, 0, -min_x], [0, 1, -min_y]], dtype=np.float32)
				R_shift = shift @ np.vstack([R, [0, 0, 1]]).astype(np.float32)
				R_shift = R_shift[:2]
				# Inverse mapping on expanded canvas
				grid_x, grid_y = np.meshgrid(
					np.arange(new_w, dtype=np.float32), np.arange(new_h, dtype=np.float32)
				)
				ones = np.ones_like(grid_x)
				dst = np.stack([grid_x, grid_y, ones], axis=0).reshape(3, -1)
				R_inv = cv2.invertAffineTransform(R_shift)
				src = (R_inv @ dst).reshape(2, new_h, new_w)
				map_x, map_y = src[0], src[1]
				rot_arr = self._apply_remap(image, map_x, map_y)

				# Update metadata in world coords

				# Transform the four world corners using pure 2D rotation around rot_pt_world
				xll, yll = dat["xllcorner"], dat["yllcorner"]
				cs = dat["cellsize"]
				nrows, ncols = dat["nrows"], dat["ncols"]

				world_corners = np.array(
					[
						[xll, yll + nrows * cs],  # (0,0) -> world top-left
						[xll + ncols * cs, yll + nrows * cs],  # (w,0) -> world top-right
						[xll, yll],  # (0,h) -> world bottom-left
						[xll + ncols * cs, yll],  # (w,h) -> world bottom-right
					],
					dtype=float,
				)

				# 2D rotation about rot_pt_world

				# point, rotationangle, rotationpoint=(0, 0, 0), rotationaxis="z"):
				wc = np.array(
					[self._rotate_pt((*wc_, 0), (*rot_pt_world, 0)) for wc_ in world_corners]
				)

				xmin, ymin = wc[:, 0].min(), wc[:, 1].min()
				xmax, ymax = wc[:, 0].max(), wc[:, 1].max()

				# Keep same cellsize; align to integer cell grid implied by new_w/new_h
				out = dat.copy()
				out.update(
					{
						"xllcorner": float(xmin),
						"yllcorner": float(ymin),
						"ncols": int(new_w),
						"nrows": int(new_h),
						"data": rot_arr,
					}
				)
				return out

		# General 3D rotation case

		# TODO: support arbitrary axes by embedding a 3D-to-2D mapping.
		raise NotImplementedError("Only rotation about z-axis is implemented currently.")

		# I would like to have the ability to rotate arbitrarily and it reproject back into a raster format
		# but this turns out to be a bit more sophisticated that I was initially thinking
