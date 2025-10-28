#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raster Curvature Methods
"""

import numpy as np


class Curvature:
	def __init__(self):
		pass

	def __call__(self, method, idat, **kwargs):
		if not hasattr(self, method):
			raise ValueError(f"Unknown curvature method '{method}'. Available: {self._available()}")
		func = getattr(self, method)
		return self._apply_mask(func, idat, **kwargs)

	def _apply_mask(self, func, idat, **kwargs):
		data = idat["data"]
		nodata = idat["nodata"]
		mask = (data != nodata) & np.isfinite(data)
		result = np.full(data.shape, nodata, dtype=data.dtype)
		computed = func(idat, **kwargs)
		result[mask] = computed[mask]
		return result

	@classmethod
	def _available(cls):
		return [m for m in dir(cls) if not m.startswith("_") and callable(getattr(cls, m))]

	@staticmethod
	def _derivatives(z, cellsize):
		"""Return first and second derivatives of DEM using central differences"""
		dzdx = (np.roll(z, -1, axis=1) - np.roll(z, 1, axis=1)) / (2 * cellsize)
		dzdy = (np.roll(z, -1, axis=0) - np.roll(z, 1, axis=0)) / (2 * cellsize)
		d2zdx2 = (np.roll(z, -1, axis=1) - 2 * z + np.roll(z, 1, axis=1)) / (cellsize**2)
		d2zdy2 = (np.roll(z, -1, axis=0) - 2 * z + np.roll(z, 1, axis=0)) / (cellsize**2)
		d2zdxdy = (
			np.roll(np.roll(z, -1, axis=0), -1, axis=1)
			- np.roll(np.roll(z, -1, axis=0), 1, axis=1)
			- np.roll(np.roll(z, 1, axis=0), -1, axis=1)
			+ np.roll(np.roll(z, 1, axis=0), 1, axis=1)
		) / (4 * cellsize**2)
		return dzdx, dzdy, d2zdx2, d2zdy2, d2zdxdy

	def planform(self, idat):
		"""Planform curvature: curvature perpendicular to slope direction"""
		dzdx, dzdy, d2zdx2, d2zdy2, d2zdxdy = self._derivatives(idat["data"], idat["cellsize"])
		pcurv = (dzdx**2 * d2zdy2 - 2 * dzdx * dzdy * d2zdxdy + dzdy**2 * d2zdx2) / (
			(dzdx**2 + dzdy**2) * np.sqrt(1 + dzdx**2 + dzdy**2) + 1e-12
		)
		return pcurv

	def profile(self, idat):
		"""Profile curvature: curvature along slope direction"""
		dzdx, dzdy, d2zdx2, d2zdy2, d2zdxdy = self._derivatives(idat["data"], idat["cellsize"])
		profcurv = (dzdx**2 * d2zdx2 + 2 * dzdx * dzdy * d2zdxdy + dzdy**2 * d2zdy2) / (
			(dzdx**2 + dzdy**2) * (1 + dzdx**2 + dzdy**2) ** 1.5 + 1e-12
		)
		return profcurv

	def mean(self, idat):
		"""Mean curvature: average of principal curvatures"""
		return 0.5 * (self.profile(idat) + self.planform(idat))

	def gaussian(self, idat):
		"""Gaussian curvature: product of principal curvatures"""
		prof = self.profile(idat)
		plan = self.planform(idat)
		return prof * plan

	def tangential(self, idat):
		"""Tangential curvature: perpendicular to slope, normalized by slope magnitude"""
		dzdx, dzdy, d2zdx2, d2zdy2, d2zdxdy = self._derivatives(idat["data"], idat["cellsize"])
		slope_mag = np.sqrt(dzdx**2 + dzdy**2)
		tcurv = (dzdx**2 * d2zdy2 - 2 * dzdx * dzdy * d2zdxdy + dzdy**2 * d2zdx2) / (
			(dzdx**2 + dzdy**2) * slope_mag + 1e-12
		)
		return tcurv

	def longitudinal(self, idat):
		"""Longitudinal curvature: parallel to slope, normalized by slope magnitude"""
		dzdx, dzdy, d2zdx2, d2zdy2, d2zdxdy = self._derivatives(idat["data"], idat["cellsize"])
		slope_mag = np.sqrt(dzdx**2 + dzdy**2)
		lcurv = (dzdx**2 * d2zdx2 + 2 * dzdx * dzdy * d2zdxdy + dzdy**2 * d2zdy2) / (
			(dzdx**2 + dzdy**2) * slope_mag + 1e-12
		)
		return lcurv
