#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raster Derivative Methods
"""

import numpy as np


class FDDerivative:
	"""Finite Difference Derivative"""

	CENTERED_FD_KERNEL = {
		2: np.array([-1, 0, 1]) / 2,
		4: np.array([1, -8, 0, 8, -1]) / 12,
		6: np.array([-1, 9, -45, 0, 45, -9, 1]) / 60,
		8: np.array([1, -32 / 3, 56, -224, 0, 224, -56, 32 / 3, -1]) / 280,
	}

	FORWARD_FD_KERNEL = {
		1: np.array([0, -1, 1]),
		2: np.array([0, 0, -3, 4, -1]) / 2,
		3: np.array([0, 0, 0, -11, 18, -9, 2]) / 6,
		4: np.array([0, 0, 0, 0, -25, 48, -36, 16, -3]) / 12,
		5: np.array([0, 0, 0, 0, 0, -137, 300, -300, 200, -75, 12]) / 60,
	}

	BACKWARD_FD_KERNEL = {
		1: np.array([-1, 1, 0]),
		2: np.array([1, -4, 3, 0, 0]) / 2,
		3: np.array([-2, 9, -18, 11, 0, 0, 0]) / 6,
		4: np.array([3, -16, 36, -48, 25, 0, 0, 0, 0]) / 12,
		5: np.array([-12, 75, -200, 300, -300, 137, 0, 0, 0, 0, 0]) / 60,
	}

	def __call__(self, *args, **kwargs):
		return self.derivative(*args, **kwargs)

	def _finite_diff_kernel(self, order, type="c"):
		"""
		Helper to retrieve appropriate fd kernel
		NB: If `order` is odd, the centered stencil used in the interior will be promoted
		to `order+1` (next even). Forward/backward one-sided stencils at boundaries will
		use the requested `order` where available.
		"""
		if type == "c":
			center_order = order if (order % 2 == 0) else order + 1
			return self.CENTERED_FD_KERNEL[center_order]
		elif type == "f":
			return self.FORWARD_FD_KERNEL[order]
		elif type == "b":
			return self.BACKWARD_FD_KERNEL[order]
		else:
			raise ValueError("Unknown kernel type {type}")

	def _apply_kernel(self, arr, axis, order, mode="extrapolate"):
		"""Helper to apply appropriate kernel to data"""

		if mode == "extrapolate":
			radius = (order if order % 2 == 0 else order + 1) // 2
			out = np.full_like(arr, np.nan)

			# Central difference for the interior
			full = correlate1d(
				arr,
				self._finite_diff_kernel(order, type="c"),
				axis=axis,
				mode="constant",
				cval=np.nan,
			)

			left_k = self._finite_diff_kernel(order, type="f")
			right_k = self._finite_diff_kernel(order, type="b")

			# Use wrap mode to provide valid-but-ignored values at edges.
			# Wrapped entries are multiplied by 0 in padded one-sided stencils so
			# we can approach the boundaries from the inside out, so to speak.
			if axis == 0:
				# y-direction (rows)
				out[radius:-radius, :] = full[radius:-radius, :]
				for i in reversed(range(radius)):
					out[i, :] = correlate1d(arr, left_k, axis=0, mode="wrap", cval=np.nan)[i, :]
					out[-(i + 1), :] = correlate1d(arr, right_k, axis=0, mode="wrap", cval=np.nan)[
						-(i + 1), :
					]

			elif axis == 1:
				# x-direction (columns)
				out[:, radius:-radius] = full[:, radius:-radius]

				for i in reversed(range(radius)):
					out[:, i] = correlate1d(arr, left_k, axis=1, mode="wrap", cval=np.nan)[:, i]
					out[:, -(i + 1)] = correlate1d(arr, right_k, axis=1, mode="wrap", cval=np.nan)[
						:, -(i + 1)
					]

			return out
		else:
			# Standard central kernel with chosen boundary handling
			kernel = self._finite_diff_kernel(order, type="c")
			return correlate1d(arr, kernel, axis=axis, mode=mode)

	def derivative(self, dat, dir_str, order=2, mode="extrapolate"):
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

		data = dat["data"]
		nodata = dat.get("nodata", -9999)
		cs = dat["cellsize"]

		mask = (data != nodata) & np.isfinite(data)
		data = np.where(mask, data, np.nan)

		center_order = order if order % 2 == 0 else order + 1
		radius = center_order // 2
		if data.shape[0] <= 2 * radius or data.shape[1] <= 2 * radius:
			raise ValueError(
				f"Raster too small for requested derivative order {order} "
				f"(requires at least {2*radius+1} cells in each dimension)"
			)

		# Array copy to avoid mutibility issues
		result = data.copy()
		for axis in dir_str:
			ax = 1 if axis == "x" else 0
			result = self._apply_kernel(result, ax, order, mode=mode) / cs

		return {**dat, "data": result}
