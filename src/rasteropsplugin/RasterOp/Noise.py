#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raster Noise methods 
"""

import numpy as np
from perlin_noise import PerlinNoise

class NoiseAdder:
	def __init__(self, nodata=None, seed=None):
		np.random.seed(seed)
		self.seed = seed if seed is not None else np.random.randint(0, 1e9)
		self.nodata = nodata
		self._perlin_cache = {} # reuse noise instances for efficiency

	def __call__(self, method, dat, **kwargs):
		if not hasattr(self, method):
			raise ValueError(f"Unknown noise mode '{method}'")
		func = getattr(self, method)
		return self._apply_mask(func, dat, **kwargs)

	def _apply_mask(self, func, dat, **kwargs):
		"""If no nodata handling: apply function directly else mask nodata"""
		nodata = self.nodata
		if nodata is None:
			return func(dat, **kwargs)

		valid_mask = dat != nodata
		result = np.full(dat.shape, nodata, dtype=dat.dtype)
		result[valid_mask] = func(dat, **kwargs)[valid_mask]
		return result

	@classmethod
	def _available(cls):
		return [m for m in dir(cls) if not m.startswith("_") and callable(getattr(cls, m))]

	def normal(self, data, mean=0.0, std=1.0):
		"""Add normal noise to raster data"""
		noise = np.random.normal(loc=mean, scale=std, size=data.shape)
		return noise + data

	def uniform(self, data, low=0.0, high=1.0):
		"""Add uniform noise to raster data"""
		noise = np.random.uniform(low=low, high=high, size=data.shape)
		return noise + data

	def salt_and_pepper(self, data, amount=0.05, salt_value=1.0, pepper_value=0.0):
		"""Add salt and pepper noise to raster data"""
		out = data.copy()
		total = data.size
		num_salt = int(np.ceil(amount * total * 0.5))
		num_pepper = int(np.ceil(amount * total * 0.5))

		# Salt
		coords = tuple(np.random.randint(0, i, num_salt) for i in data.shape)
		out[coords] = salt_value

		# Pepper
		coords = tuple(np.random.randint(0, i, num_pepper) for i in data.shape)
		out[coords] = pepper_value

		return out

	def perlin(self, data, scale=10.0, octaves=1, persistence=0.5, lacunarity=2.0):
		"""Add perlin noise to raster data"""
		height, width = data.shape
		key = (octaves, persistence, lacunarity)
		if key not in self._perlin_cache:
			self._perlin_cache[key] = PerlinNoise(
				octaves=octaves,
				seed=self.seed
			)
		noise_gen = self._perlin_cache[key]
		noise = np.zeros_like(data, dtype=np.float32)

		for i in range(height):
			for j in range(width):
				x = i / scale
				y = j / scale
				noise[i, j] = noise_gen([x, y])

		return noise + data
		
	def coloured(self, data, colour="pink", exponent=None, scale=1.0):
		"""Add coloured noise"""
		colour_map = {
			"white": 0.0,
			"pink": 1.0,
			"brown": 2.0,
			"blue": -1.0,
			"violet": -2.0,
		}
		if exponent is None:
			if colour not in colour_map:
				raise ValueError(f"Unknown colour '{colour}'")
			exponent = colour_map[colour]

		ny, nx = data.shape
		noise = np.random.normal(size=(ny, nx))

		f = np.fft.fft2(noise)
		fy = np.fft.fftfreq(ny).reshape(-1, 1)
		fx = np.fft.fftfreq(nx).reshape(1, -1)
		radius = np.sqrt(fx**2 + fy**2)
		radius[0, 0] = 1e-6  # avoid divide by zero

		filt = 1.0 / (radius**exponent)
		f_filtered = f * filt
		coloured_noise = np.real(np.fft.ifft2(f_filtered))
		return data + scale * coloured_noise
