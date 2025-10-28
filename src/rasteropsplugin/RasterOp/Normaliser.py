#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raster Normaliser
"""

import numpy as np


class Normaliser:
	def __init__(self, nodata=None):
		self.nodata = nodata

	def __call__(self, method, dat, **kwargs):
		if not hasattr(self, method):
			raise ValueError(f"Unknown normalisation method '{method}'")
		func = getattr(self, method)
		return self._apply_mask(func, dat, **kwargs)

	def _apply_mask(self, func, dat, **kwargs):
		nodata = self.nodata
		if nodata is None:
			return func(dat, **kwargs)
		valid_mask = dat != nodata
		result = np.full(dat.shape, nodata, dtype=dat.dtype)
		result[valid_mask] = func(dat[valid_mask], **kwargs)
		return result

	@classmethod
	def _available(cls):
		return [m for m in dir(cls) if not m.startswith("_") and callable(getattr(cls, m))]

	def minmax(self, dat, range=(0, 1)):
		minval = np.min(dat)
		maxval = np.max(dat)
		if maxval == minval:
			return np.zeros_like(dat)
		a, b = range
		return a + (dat - minval) * (b - a) / (maxval - minval)

	def profile(self, dat, profile="linear", range=(0, 1)):
		"""
		Create a profile-based ramp from input raster-like array.
		profile : If str, {"linear", "quadratic", "cubic", "inv-quadratic", "s-curve"}
				  If list/tuple, treated as polynomial coefficients (c0, c1, ..., cn).
		"""

		minval, maxval = np.min(dat), np.max(dat)
		if maxval == minval:
			return np.zeros_like(dat)

		# Normalize input to [0, 1]
		t = (dat - minval) / (maxval - minval)

		# Handle profile selection
		if isinstance(profile, str):
			profiles = {"linear", "quadratic", "cubic", "inv-quadratic", "s-curve"}
			profile = profile.lower()
			if profile == "linear":
				f = t
			elif profile == "quadratic":
				f = t**2
			elif profile == "cubic":
				f = t**3
			elif profile == "inv-quadratic":
				f = 1 - (1 - t) ** 2
			elif profile == "s-curve":
				# NB: sigmoid curve
				f = 3 * t**2 - 2 * t**3  # smoothstep
			else:
				raise ValueError(
					f"Unknown preset profile '{profile}'. Choose from {profiles} or enter polynomial coeffs."
				)

		else:
			# Power law with exponent
			if isinstance(profile[0], str) and profile[0].lower() == "power":
				try:
					exp = float(profile[1])
				except ValueError:
					raise ValueError(f"Power profile exponent must be numeric, got {profile[1]}")
				f = t**exp

			# Polynomial Coeffs
			else:
				# Assume polynomial coefficients
				coeffs = np.asarray(profile)
				f = sum(c * t**i for i, c in enumerate(coeffs))

				# Normalize f to [0, 1]
				f0 = coeffs[0]  # polynomial value at t=0
				f1 = sum(coeffs)  # polynomial value at t=1
				if f1 == f0:
					f = np.zeros_like(t)
				else:
					f = (f - f0) / (f1 - f0)

		# Scale to target range
		a, b = range
		return a + f * (b - a)

	def zscore(self, dat):
		mean = np.mean(dat)
		std = np.std(dat)
		if std == 0:
			return np.zeros_like(dat)
		return (dat - mean) / std

	def robust(self, dat, prange=(75, 25)):
		median = np.median(dat)
		iqr = np.subtract(*np.percentile(dat, list(prange)))
		if iqr == 0:
			return np.zeros_like(dat)
		return (dat - median) / iqr

	def l1(self, dat):
		norm = np.sum(np.abs(dat))
		if norm == 0:
			return np.zeros_like(dat)
		return dat / norm

	def l2(self, dat):
		norm = np.sqrt(np.sum(dat**2))
		if norm == 0:
			return np.zeros_like(dat)
		return dat / norm

	def linf(self, dat):
		norm = np.max(np.abs(dat))
		if norm == 0:
			return np.zeros_like(dat)
		return dat / norm

	def frobenius(self, dat):
		norm = np.linalg.norm(dat, 'fro')
		if norm == 0:
			return np.zeros_like(dat)
		return dat / norm
