#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raster Recipes
"""

import numpy as np

from .Friction import Friction

class Recipes:

	_meta = {
		"treecount2nD" : {
			"args": {
				"count": {"type": "raster", "label": "Tree count raster"},
				"bhd": {"type": "raster", "label": "Breast Height Diameter (m) raster"},
			},
			"description": "Generate nD raster from treecount and breast height diameter (BHD) rasters",
		},
		"potential_rel_area" : {
			"args": {
				"dem": {"type": "raster", "label": "Tree count raster"},
				"relief_frac": {"type": "float", "label": "relief fraction ignored", "optional": "true"},
				"slope_threshold_low" : {"type": "float", "label": "Slope angle low", "optional": "true"},
				"slope_threshold_high" : {"type": "float", "label": "Slope angle high", "optional": "true"},
			},
			"description": "Generate nD raster from treecount and breast height diameter (BHD) rasters",
		},
		"Froude" : {
			"args": {
				"dem": {"type": "raster", "label": "DEM raster"},
				"hdat": {"type": "raster", "label": "Height raster (m)"},
				"sdat": {"type": "raster", "label": "Speed raster (m/s)"},
			},
			"description": "Calculate Froude number from DEM, H, S rasters",
		},
		"NAKSIN_mu" : {
			"args": {
				"dem": {"type": "raster", "label": "DEM raster"},
				"relh": {"type": "raster", "label": "Release area raster (m)"},
				"T_ret": {"type": "float", "label": "Return Period"},
				"winter_temp_av" : {"type": "float", "label": "Winter temperature average", "optional": "true"},
				"lapse_rate" : {"type": "float", "label": "Temperature lapse rate", "optional": "true"},
			},
			"description": "Calculate NAKSIN's spatial friction mu",
		},
		"NAKSIN_k" : {
			"args": {
				"dem": {"type": "raster", "label": "DEM raster"},
				"relh": {"type": "raster", "label": "Release area raster (m)"},
				"T_ret": {"type": "float", "label": "Return Period"},
				"winter_temp_av" : {"type": "float", "label": "Winter temperature average", "optional": "true"},
				"lapse_rate" : {"type": "float", "label": "Temperature lapse rate", "optional": "true"},
			},
			"description": "Calculate NAKSIN's spatial friction k",
		},
		"snow_accum_layer" : {
			"args": {
				"dem": {"type": "raster", "label": "DEM raster"},
			},
			"description": "Generate nD raster from treecount and breast height diameter (BHD) rasters",
		},
	}


	def __init__(self, rasterop=None):
		self.rasterop = rasterop

	def __call__(self, recipe, *args, **kwargs):
		if not hasattr(self, recipe):
			raise ValueError(f"Unknown recipe '{recipe}'")
		return getattr(self, recipe)(*args, **kwargs)

	@classmethod
	def _available(cls):
		return [m for m in dir(cls) if not m.startswith("_") and callable(getattr(cls, m))]

	def treecount2nD(self, count, bhd):
		"""Generate nD raster from treecount and breast height diameter (BHD) rasters"""
		out = self.rasterop._swap(
			self.rasterop._scale(self.rasterop._union(count, bhd, combine='product'), factor=1e-6),
			-9999.0,
			0,
		)
		
		print(out["data"][out["data"] > 0])
		
		
		return out

	def potential_rel_area(self, dem, relief_frac=1.0, slope_threshold_low=30, slope_threshold_high=55):
		"""
		Multiple thresholds using boolean AND to highlight potential release areas where
		DEM > 500m and slope_angle > 30
		"""

		# Slope in degrees np.degrees(arctan(\sqrt{df/dx^2 + df/dy^2}))
		slope = self.rasterop._slope(dem)

		min_height = self.rasterop._stats(dem, stat="min")
		relief_threshold = min_height + relief_frac * self.rasterop._stats(dem, stat="relief")

		# _threshold (idat, cond_arr, cond_op, cond_val, bitops=None, threshold_val=1.0, fill_val=0.0)
		out = self.rasterop._threshold(
			dem,
			["self", slope, slope],
			[">", ">=", "<="],
			[relief_threshold, slope_threshold_low, slope_threshold_high],
			["&", "&"],
			1.0,
			0.0,
		)
		return out

	def Froude(self, dem, hdat, sdat, g=9.8, hthreshold=1e-12):
		"""
		Compute the Froude number raster
		Requires DEM, H, S rasters
		"""

		# Slope in degrees
		slope = self.rasterop._slope(dem)
		cost = np.cos(np.radians(slope["data"]))

		h = hdat["data"]
		s = sdat["data"]

		# Boolean mask for valid water depths
		mask = h > hthreshold

		# Prepare arrays
		Fr = np.zeros_like(h, dtype=float)

		if np.any(mask):
			speed = np.where(mask, np.abs(s), 0.0)
			wavespeed = np.where(mask, np.sqrt(g * h * cost), 0.0)

			# cells where wavespeed==0 are set to 0
			valid = wavespeed > 0
			Fr[valid] = speed[valid] / wavespeed[valid]

		out = {k: v for k, v in hdat.items() if k != "data"}
		out["data"] = Fr
		return out
	
	def NAKSIN_mu(self, dem, relh, T_ret, winter_temp_av=2, lapse_rate=-0.0065):
		"""
		Create the NASKIN friction coefficient raster for this DEM and ReleaseArea at return period T_ret
		winter_temp_av: winter temperature average at location deg C
		temp_lapse_rate: rate temperature decreases with elevation 
		"""
		
		if winter_temp_av is None: 
			winter_temp_av = 2
		
		if lapse_rate is None:
			lapse_rate = -0.0065
				
		temp_av = (winter_temp_av, lapse_rate)
		friction = Friction(dem, relh)
		mu_data = friction.naksin_mu(T_ret, temp_av=temp_av)
		
		odat = dem.copy()
		odat["data"] = mu_data
		
		return odat
		
	def NAKSIN_k(self, dem, relh, T_ret, winter_temp_av=2, lapse_rate=-0.0065):
		"""
		Create the NASKIN friction coefficient raster for this DEM and ReleaseArea at return period T_ret
		winter_temp_av: winter temperature average at location deg C
		temp_lapse_rate: rate temperature decreases with elevation 
		"""
		
		if winter_temp_av is None: 
			winter_temp_av = 2
		
		if lapse_rate is None:
			lapse_rate = -0.0065
		
		temp_av = (winter_temp_av, lapse_rate)
		
		friction = Friction(dem, relh)
		k_data = friction.naksin_k(T_ret, temp_av=temp_av)
		
		odat = dem.copy()
		odat["data"] = k_data
		
		return odat
	

	# TODO: construct a function that enhances depth if inline with winddir
	def snow_accum_layer(
		self,
		dem,
		layerval=1.0,
		winddir=0.0,
		slope_threshold=(25.0, 45.0),
		aspect_opening_angle=45.0,
		modifiers=(0.5, 1.5),
	):
		"""Compute a layer over the top of DEM given a winddir"""

		penalty, gain = modifiers
		slopemin, slopemax = slope_threshold
		aspectmin = (winddir - aspect_opening_angle) % 360
		aspectmax = (winddir + aspect_opening_angle) % 360

		slope = self.rasterop._slope(dem)
		aspect = self.rasterop._aspect(dem)

		slope_mask = (slope["data"] >= slopemin) & (slope["data"] <= slopemax)

		# Handle wraps across 0
		if aspectmin < aspectmax:
			aspect_mask = (aspect["data"] >= aspectmin) & (aspect["data"] <= aspectmax)
		else:
			aspect_mask = (aspect["data"] >= aspectmin) | (aspect["data"] <= aspectmax)

		mask = slope_mask & aspect_mask

		basedat = np.ones_like(dem["data"])
		multiplier = np.where(mask, basedat * gain, basedat * penalty)

		outdat = np.full_like(basedat, layerval) * multiplier

		out = dem.copy()
		out["data"] = outdat
		return out
