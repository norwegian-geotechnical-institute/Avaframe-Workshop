#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Friction methods

TODO: Update to use new RasterOp methods e.g. curvatures etc.
"""

import os
import numpy as np
from math import log10
import copy

from ..FileIO.FileReaders import ASCIIFileReader

class Friction:
	def __init__(self, dem, relh):
		self.store = {
			"dem": copy.deepcopy(dem),
			"relh": copy.deepcopy(relh),
		}
		self.cellsize = dem["cellsize"]
		self.nrows = dem["nrows"]
		self.ncols = dem["ncols"]
		self.nodata = dem.get("nodata", -9999)
		
		self.volume = self._calculaterelvol()
		
		
	def _calculaterelvol(self):
		"""
		Calculate the release volume.
		NB: This requires self.store["relh"] to be set prior.
		"""
		if self.store.get("relh") is None:
			return 0

		relh_data = self.store["relh"]["data"]
		nodata = self.store["relh"]["nodata"]

		# Build mask: finite, not nodata, not zero
		mask = np.isfinite(relh_data) & (relh_data != nodata) & (relh_data != 0)

		return np.sum(relh_data[mask] * self.cellsize**2)

		
		
	# NB: This is re-worked release_area.py:terrain_parameters() from NAKSIN
	def _terrain_parameters(self):
		"""Derives potential release areas based on slope and curvature.

		Input parameters:
			dem             Digital elevation model cut to the area of interest
							(aoi.asc)
			slope_min       Min. slope angle for release
			slope_max       Max. slope angle for release
			plancurv_max    Max. planform curvature within single release area

		Output:
			slope           Slope angle based on Horns' method (using 6 neighbours)
			plan_curvature  Planform curvature (curvature of contour lines)
			D8_dir			D8 flow direction (steepest path of 8 directions)
		"""

		if self.store.get("dem") is None:
			raise RuntimeError(
				"DEM must be set in Friction class prior to terrain_parameters call."
			)
		
		dem = self.store.get("dem", {}).get("data")

		xres = self.cellsize
		yres = self.cellsize * -1
		scale = 1

		# Geometric terrain properties are calculated on a basic 3x3 stencil. For
		# best use of NumPy's array operations, create 9 altitude fields shifted
		# in correspondence with the stencil points.
		window = []
		for row in range(3):
			for col in range(3):
				window.append(
					dem[row : (row + self.nrows - 3), col : (col + self.ncols - 3)]
				)

		# Process each cell
		dzdx = (
			(window[0] + window[3] + window[3] + window[6])
			- (window[2] + window[5] + window[5] + window[8])
		) / (8.0 * xres * scale)

		dzdy = (
			(window[6] + window[7] + window[7] + window[8])
			- (window[0] + window[1] + window[1] + window[2])
		) / (8.0 * yres * scale)

		# Calculate terrain parameters:
		slope = np.full(dem.shape, 0.0, np.double)
		restricted_slope = slope[1 : int(self.nrows - 2), 1 : int(self.ncols - 2)]

		# NB: Mutibility - this will effect slope!
		# This puts out=restricted_slope implicitly
		np.degrees(np.arctan(np.sqrt(dzdx * dzdx + dzdy * dzdy)), restricted_slope)
		# aspect = np.degrees(np.arctan2(dzdx, dzdy))

		D = ((window[3] + window[5]) / 2 - window[4]) / (yres * yres * scale)
		E = ((window[1] + window[7]) / 2 - window[4]) / (yres * yres * scale)
		F = (-window[0] + window[2] + window[6] - window[8]) / (4 * yres * yres * scale)
		G = (-window[3] + window[5]) / (2 * yres)
		H = (window[1] - window[7]) / (2 * yres)

		curvaturePlan = np.full(dem.shape, 0.0, np.double)
		restricted_curvaturePlan = curvaturePlan[1 : int(self.nrows - 2), 1 : int(self.ncols - 2)]

		curvatureGauss = np.full(dem.shape, 0.0, np.double)
		restricted_curvatureGauss = curvatureGauss[1 : int(self.nrows - 2), 1 : int(self.ncols - 2)]

		# NB. The following formulas compute the same curvatures as ArcGIS does,
		# but differ by a factor -100. Here, curvature is negative on a shoulder
		# and positive in a gully. The units are 1/m.

		# NB: MUTIBILITY: this effects curvature plan
		np.copyto(
			restricted_curvaturePlan,
			2 * (D * H * H + E * G * G - F * G * H) / np.maximum(0.01, (G * G + H * H)),
		)
		# curvatureProf = 200*(D*G*G + E*H*H + F*G*H)/np.maximum(0.01,(G*G + H*H))

		# NB: MUTIBILITY: this effects curvature Gauss
		np.copyto(restricted_curvatureGauss, 2.0 * (D + E))
		# curvature = -2*(D + E) *100;

		# D8 flow direction
		D8_dir = np.full(dem.shape, 0.0, np.double)
		restricted_D8 = D8_dir[1 : int(self.nrows - 2), 1 : int(self.ncols - 2)]

		allslopes = np.full([9, window[0].shape[0], window[0].shape[1]], np.nan)

		for i in (0, 2, 6, 8):
			allslopes[i] = np.arctan((window[4] - window[i]) / (self.cellsize * np.sqrt(2)))

		for i in (1, 3, 5, 7):
			allslopes[i] = np.arctan((window[4] - window[i]) / self.cellsize)

		# NB: MUTIBILITY: this effects curvature D8_dir
		np.copyto(restricted_D8, np.nanargmax(allslopes, 0))
		restricted_D8[restricted_slope < 1] = 4
		# Areas with slope angle < 1° are considered flat.

		# Set no-data values -- MoT-Voellmy wants 0, and this is OK for NAKSIN, too.
		# NB: This doesn't hit the border values that then get set to nodata and not 0
		for pane in window:
			restricted_slope[pane == self.nodata] = 0.0
			restricted_curvaturePlan[pane == self.nodata] = 0.0

		return slope, curvaturePlan, curvatureGauss, D8_dir
		
	# NB: This is a re-worked version of prepare_data.py:prepare_climate_data() from NAKSIN
	def _get_TA_params(self):
		"""Avg. winter temperature, elevation retrieved from SeNorge data
		For use in Step 5 (run-out calculations), the average elevation and
		winter temperature of all SeNorge cells in the AoI are clipped out
		from locally stored files. A linear regression of TA vs. Z yields
		the avg. winter temperature at sea level, TA_intc, and the lapse
		rate, TA_grad.
		"""

		root = os.path.dirname(os.path.realpath(__file__))
		TA_fn = str(os.path.join(root, "input", "NAKSIN_input", "TA_djf_N30.asc"))
		Z_SN_fn = str(os.path.join(root, "input", "NAKSIN_input", "DEMsenorge.asc"))

		reader = ASCIIFileReader()
		TA = reader.read(
			TA_fn, clip=[self.xllcorner, self.xurcorner, self.yllcorner, self.yurcorner]
		)
		Z = reader.read(
			Z_SN_fn, clip=[self.xllcorner, self.xurcorner, self.yllcorner, self.yurcorner]
		)

		# Interpolate the climate data from the SeNorge raster to the DEM raster:

		# Need to mask NaN and flatten the arrays for linear regression analysis:
		Zm = np.ma.masked_values(Z["data"].flatten(), Z["nodata"])
		TAm = np.ma.masked_values(TA["data"].flatten(), TA["nodata"])

		# Check whether there are enough data points for a meaningful regression
		# If not, ask the user to specify temperature at sea level and lapse rate
		if TAm.count() < 6 or Zm.count() < 6:
			error_msg = f"Not enough climate data found. Only {TAm.count()} valid SeNorge cells available. Consider using manual values."
			print(WARNING(error_msg))

		# Linear regression of winter temperature vs. altitude over entire area:

		if TAm.count() > 2:
			spanZ = np.max(Zm) - np.min(Zm)
		else:
			raise RuntimeError(
				"Not enough temperature data to perform regression. Use manual setting. '--temp_av= TA_intc TA_grad'."
			)

		if spanZ > 100.0:
			# TODO: Improve this to a trilinear fit using X, Y and Z-coordinates
			TA_grad, TA_intc = np.ma.polyfit(Zm, TAm, 1)
		else:
			# In rare cases (e.g., 14344 Auvær), all SeNorge cells have altitude 0
			# so that linear regression is impossible. Then take mean temperature
			# as sea-level temperature and assume lapse rate -0.001°C/m.
			TA_intc = np.mean(TAm)
			TA_grad = -0.001

		# 	print("TA = {:6.2f}°C + ({:6.4f}°C/m)".format(TA_intc, TA_grad))

		return TA_intc, TA_grad

	# This is reworked ./runout.py:friction_params() from NAKSIN
	def _friction_params(self, curv, vol, T_ret, mode, TAvals=None, Tthresholds=None):
		"""Calculation of locally interpolated friction parameters

		SLF has published a table of friction values for RAMMS with categories
		for avalanche size, topographic character of the path, altitude and return
		period. Moreover, the original table contains abrupt changes across
		category boundaries that cannot be justified on physical grounds and may
		give misleading results. For these reasons, the routine friction_params()
		implements an interpolation between categories.

		When applying RAMMS in Norway, the limits of the altitude categories have
		to be adjusted to different degrees for different parts of the country.
		To make the altitude interpolation more robust and objective, the altitude
		dependence is replaced by a dependence on the mean winter temperature
		(months Dec., Jan., Feb.), for which altitude was a proxy in the first
		place.

		The RAMMS::AVALANCHE manual v.1.7.0 provides a table of friction values
		for return periods of 10, 30, 100 and 300 years. Tests of RAMMS in Norway
		have shown that the values for T = 300 y give reasonable results for
		avalanches with nominal T = 1000 y; for T = 5000 y, one should presumably
		lower them a little. This routine allows to interpolate and extrapolate
		to user-selected return periods between 1 and 5000 y; linear relations
		between mu, k and log T are assumed.

		First, a table for the desired return period is generated from SLF's
		tables. Next, the values in this table are interpolated for the avalanche
		volume. Afterwards, the volume-interpolated category values are
		interpolated for each grid cell wrt. altitude, then wrt. the planform
		curvature. Finally, the raster is written to disk in ESRI ASCII Grid
		format.

		For increased flexibility, the drag coefficient can be written either as
		the dimensionless parameter k used in MoT-Voellmy or as xi = g/k, as used
		in RAMMS. For use in NAKSIN, mode 'k' is to be specified.


		Input parameters:
		dem         NumPy array of topography at simulation resolution
		curv        NumPy array of curvature at simulation resolution
		vol         Avalanche release volume
		mode        Output mode for drag coefficient, either 'k' for
					MoT-Voellmy or 'xi' for RAMMS
		TAvals		Tuple: Average winter temperate at sealevel, Temperature lapse rate
		Tthresholds	Tuple: Temp threshold go from altitude class 1 -> 2, Temp threshold go from altitude class 2 -> 3
		"""

		# Safety check: is the output mode variable set to an allowed value?
		# NB: Note command line parse should not allow this to get this far
		if not mode in ['k', 'xi']:
			raise RuntimeError(
				"The output mode of friction_params() must be either" + " 'k' or 'xi'.\n"
			)
			
		dem = self.store.get("dem",{}).get("data")

		# Create 4D lists (4x4x4x3) for mu and k from input table
		mu_tab = [[[[0.0 for a in range(3)] for c in range(4)] for s in range(4)] for t in range(4)]
		k_tab = [[[[0.0 for a in range(3)] for c in range(4)] for s in range(4)] for t in range(4)]

		# Create 3D tables for specified return period:
		mu_tab_T = [[[0.0 for a in range(3)] for c in range(4)] for s in range(4)]
		k_tab_T = [[[0.0 for a in range(3)] for c in range(4)] for s in range(4)]

		# Create 2D tables for specified avalanche volume:
		mu_tab_TV = [[0.0 for a in range(3)] for c in range(4)]
		k_tab_TV = [[0.0 for a in range(3)] for c in range(4)]

		# Create 2D arrays to hold the interpolated mu and k values
		nrows, ncols = self.nrows, self.ncols

		mu = np.array([[0.0 for _ in range(ncols)] for _ in range(nrows)])
		k = np.array([[0.0 for _ in range(ncols)] for _ in range(nrows)])

		# Supply the RAMMS table
		root = os.path.dirname(os.path.realpath(__file__))
		calib_fn = os.path.join(root, "assets", "fric_coeffs_RAMMS_10-300y.txt")

		# Read SLF's table of friction parameters and close it
		with open(calib_fn, 'r', encoding='utf-8') as pfp:
			zeile = pfp.readline()
			for t in range(4):  # Index for return period (10, 30, 100, 300y)
				for v in range(4):  # Index for size category
					for c in range(4):  # Index for curvature category
						while zeile[0] == '#':
							zeile = pfp.readline()
						mu_tab[t][v][c][0] = float(zeile.split()[0][1:-1])
						mu_tab[t][v][c][1] = float(zeile.split()[2][1:-1])
						mu_tab[t][v][c][2] = float(zeile.split()[4][1:-1])
						k_tab[t][v][c][0] = float(zeile.split()[1][0:-1])
						k_tab[t][v][c][1] = float(zeile.split()[3][0:-1])
						k_tab[t][v][c][2] = float(zeile.split()[5][0:-1])
						zeile = pfp.readline()

		# Interpolate/extrapolate the parameters to the specified return period

		if T_ret <= 30.0:
			L, H, lgTL, lgTH = 0, 1, 1.0, log10(30.0)
		elif T_ret <= 100.0:
			L, H, lgTL, lgTH = 1, 2, log10(30.0), 2.0
		else:
			L, H, lgTL, lgTH = 2, 3, 2.0, log10(300.0)

		# In this version, limit the return period to 10 y < T < 300 y
		# lgT = log10(T_ret)
		lgT = log10(max(10.0, min(300.0, T_ret)))
		w = (lgTH - lgT) / (lgTH - lgTL)

		for s in range(4):
			for c in range(4):
				for a in range(3):
					mu_tab_T[s][c][a] = mu_tab[L][s][c][a] * w + mu_tab[H][s][c][a] * (1.0 - w)
					k_tab_T[s][c][a] = k_tab[L][s][c][a] * w + k_tab[H][s][c][a] * (1.0 - w)

		# Determine the size class of the present avalanche and interpolate
		# SLF's parameter table according to the volume
		# NB: This is fixed from the table!
		V_ts = 5000.0
		V_sm = 25000.0
		V_ml = 60000.0

		if vol < V_ts:  # Tiny avalanche, no interpolation
			L, H = 0, 1
			w = 1.0
		elif vol < V_sm:  # Small avalanche, interpolate
			L, H = 1, 2
			w = (V_sm - vol) / (V_sm - V_ts)
		elif vol < V_ml:  # Medium-size avalanche, interpolate
			L, H = 2, 3
			w = (V_ml - vol) / (V_ml - V_sm)
		else:  # Large avalanche, no interpolation
			L, H = 3, 3
			w = 1.0

		for c in range(4):
			for a in range(3):
				mu_tab_TV[c][a] = w * mu_tab_T[L][c][a] + (1 - w) * mu_tab_T[H][c][a]
				k_tab_TV[c][a] = w * k_tab_T[L][c][a] + (1 - w) * k_tab_T[H][c][a]

		# For each cell, interpolate mu and k according to the temperature
		# and the curvature
		# NB: For test areas with no temperature data - set to some default values with a warning!

		if TAvals is None:
			# fetch TAvals
			try:
				TA_intc, TA_grad = self._get_TA_params()
			except (ValueError, RuntimeWarning) as e:
				TA_intc, TA_grad = 2.0, -0.0065
				print(
						f"Sea-level winter temperature / temperature lapse rate could not be fetched. Default values chosen {TA_intc=:.1f}°C {TA_grad=:.4f}°C."
				)
		else:
			TA_intc, TA_grad = (float(TAvals[0]), float(TAvals[1]))

		# Check whether the obtained values are reasonable - warn if not
		if TA_intc < -20.0 or TA_intc > 20.0:
			print("Unreasonable sea-level winter temperature:  {:.1f}°C. Consider setting manually.".format(TA_intc))

		if TA_grad > 0.0:
			print("Unreasonable winter temperature lapse rate set: {:.3f}°C/m. Consider setting manually.".format(TA_grad))

		# NB: These default values are from NASKIN input. These correspond to altitude class essentially
		# so I guess it should be possible just choose altitude class also...
		if Tthresholds is None:
			t12 = -2.0  # High-temperature threshold (deg C)
			t23 = -6.0  # Low-temperature threshold (deg C)
		else:
			t12, t23 = Tthresholds

		# NB: Curvature bounds for different classifications. From RAMMS??
		C_f = 0.0000  # Flat-slope curvature
		C_n = 0.0020  # Open-slope
		C_c = 0.0100  # Channelized curvature
		C_g = 0.0500  # Gully curvature

		for i in range(nrows):
			for j in range(ncols):
				TA, C = TA_intc + TA_grad * dem[i][j], curv[i][j]

				if TA > t12:  # Altitude/temperature class 1
					L, H, v = 0, 2, 1.0
				elif TA > t23:  # Altitude/temperature class 2
					L, H, v = 0, 2, (TA - t23) / (t12 - t23)
				else:  # Altitude/temperature class 3
					L, H, v = 0, 2, 0.0

				if C < C_f:  # Open (flat) slope
					S, R, w = 0, 0, 1.0
				elif C < C_n:  # Open slope to Non-channelized
					S, R, w = 0, 1, (C_n - C) / (C_n - C_f)
				elif C < C_c:  # Non-channelized to Channelized
					S, R, w = 1, 2, (C_c - C) / (C_c - C_n)
				elif C < C_g:  # Channelized to Gully
					S, R, w = 2, 3, (C_g - C) / (C_g - C_c)
				else:  # Gully
					S, R, w = 3, 3, 0.0

				mu[i][j] = w * (v * mu_tab_TV[S][L] + (1 - v) * mu_tab_TV[S][H]) + (1 - w) * (
					v * mu_tab_TV[R][L] + (1 - v) * mu_tab_TV[R][H]
				)

				k[i][j] = w * (v * k_tab_TV[S][L] + (1 - v) * k_tab_TV[S][H]) + (1 - w) * (
					v * k_tab_TV[R][L] + (1 - v) * k_tab_TV[R][H]
				)

				if mode == 'xi':
					k[i][j] = 9.81 / k[i][j]

		return mu, k
		
	#----------------------------------------------------
	# Methods 
	#----------------------------------------------------

	def naksin_mu(self, T_ret, volume=None, mode="k", temp_av=(2, -0.0065), temp_thresholds=(-2.0, -6.0)):
		"""
		The friction mu param generation method from NAKSIN
		"T_ret": 300,  # Return Period Years
		"volume": None,  # Release Area Volume - override - calculated from release area
		"mode": "k",  # k or xi
		"temp_av": (2, -0.0065),  # Average winter temp, temp lapse rate
		"temp_thresholds": (-2.0,-6.0),  # Default temperature thresholds to swap "altitude" parameterisation
		"""

		T_ret = T_ret
		
		# If volume overriden
		vol = volume
		if vol is None:
			vol = self.volume
			
		self.store.setdefault("mu", None)
		self.store.setdefault("k", None)

		if self.store["mu"] is None:

			# Planar Curvature
			_, curvaturePlan, _, _ = self._terrain_parameters()

			# Friction calc - using curvature, volume, return period, "k" or "xi", temp_av = (TA_intc, TA_grad),
			mu, k = self._friction_params(
				curvaturePlan, self.volume, T_ret, mode, temp_av, temp_thresholds
			)

			self.store["mu"] = mu

			# Sets k as well if it hasn't been set yet
			if self.store["k"] is None:
				self.store["k"] = k

		return self.store["mu"]


	def naksin_k(self, T_ret, volume=None, mode="k", temp_av=(2, -0.0065), temp_thresholds=(-2.0, -6.0)):
		"""
		The friction k param generation method from NAKSIN
		"T_ret": 300,  # Return Period Years
		"volume": None,  # Release Area Volume - override - calculated from release area
		"mode": "k",  # k or xi
		"temp_av": (2, -0.0065),  # Average winter temp, temp lapse rate
		"temp_thresholds": (-2.0,-6.0),  # Default temperature thresholds to swap "altitude" parameterisation
		"""

		T_ret = T_ret
		
		# If volume overriden
		vol = volume
		if vol is None:
			vol = self.volume
			
		self.store.setdefault("mu", None)
		self.store.setdefault("k", None)

		if self.store["k"] is None:

			# Planar Curvature
			_, curvaturePlan, _, _ = self._terrain_parameters()

			# Friction calc - using curvature, volume, return period, "k" or "xi", temp_av = (TA_intc, TA_grad),
			mu, k = self._friction_params(
				curvaturePlan, vol, T_ret, mode, temp_av, temp_thresholds
			)

			self.store["k"] = k

			# Sets k as well if it hasn't been set yet
			if self.store["mu"] is None:
				self.store["mu"] = mu

		return self.store["k"]

