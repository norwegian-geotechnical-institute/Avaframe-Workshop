#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raster Combine Methods
"""

import numpy as np


class Combine:
	def __init__(self, fill_value=0.0, nodata=None):
		self.fill_value = fill_value
		self.nodata = nodata  # Optional

	def __call__(self, method, d1, d2, **kwargs):
		assert d1.shape == d2.shape, "Raster shapes must match!"
		if not hasattr(self, method):
			raise ValueError(f"Unknown combine method '{method}'")
		func = getattr(self, method)
		return self._apply_mask(func, d1, d2, **kwargs)

	def _apply_mask(self, func, d1, d2, **kwargs):
		nodata = self.nodata
		# No if nodata handling — apply function directly
		if nodata is None:
			return func(d1, d2, **kwargs)

		valid_mask = (d1 != nodata) & (d2 != nodata)
		result = np.full(d1.shape, nodata, dtype=d1.dtype)
		result[valid_mask] = func(d1, d2, **kwargs)[valid_mask]
		return result

	@classmethod
	def _available(cls):
		return [m for m in dir(cls) if not m.startswith("_") and callable(getattr(cls, m))]

	def max(self, d1, d2):
		return np.maximum(d1, d2)

	def min(self, d1, d2):
		return np.minimum(d1, d2)

	def sum(self, d1, d2):
		return d1 + d2

	def diff(self, d1, d2):
		return d1 - d2

	def product(self, d1, d2):
		return d1 * d2

	def quotient(self, d1, d2):
		return d1 / d2

	def mean(self, d1, d2):
		return (d1 + d2) / 2

	def wmean(self, d1, d2, w1=0.5, w2=0.5):
		"""Weighted mean. Weights must sum to 1 or will be normalized."""
		w1 = float(w1)
		w2 = float(w2)
		wsum = w1 + w2
		if not np.isclose(wsum, 1.0):
			w1 /= wsum
			w2 /= wsum
		return w1 * d1 + w2 * d2

	def loverwrite(self, d1, d2):
		"""Overwrite values in d1 with non-zero values from d2"""
		return np.where(d2 != 0, d2, d1)

	def roverwrite(self, d1, d2):
		"""Overwrite values in d2 with non-zero values from d1"""
		return np.where(d1 != 0, d1, d2)

	def mask_union(self, d1, d2, fill_value=None):
		"""Union of valid values from both arrays"""
		fill = self.fill_value if fill_value is None else fill_value
		return np.where((d1 != fill) | (d2 != fill), np.maximum(d1, d2), fill)

	def mask_intersection(self, d1, d2, fill_value=None):
		"""Intersection of valid values in both arrays"""
		fill = self.fill_value if fill_value is None else fill_value
		return np.where((d1 != fill) & (d2 != fill), np.maximum(d1, d2), fill)
