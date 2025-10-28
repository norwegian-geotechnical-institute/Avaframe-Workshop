#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Reader/Writer for ESRII ASCII Grid format 
"""

import numpy as np

from .FileUtils import ReaderUtilities, File, TarFile

# ESRI-ASCII File Reader
class ASCIIFileReader(ReaderUtilities):
	"""
	Read an ascii-format terrain file

	Reads the header
	Parses headerstring into a dictionary
	Reads the data directly into a np.array
	Adds extras such as filename etc. to data dict and returns
	"""

	def __init__(self):
		self.name = "ASCIIFileReader"
		ReaderUtilities.__init__(self)

	def read_header(self, fid, stop=None, inclusive=False):
		"""Read the header"""

		from string import whitespace

		header = fid.readlines(stop=stop, inclusive=inclusive)

		value_list = []
		for v in header:
			tmp = []
			for k in v.split():
				# NB: want to make sure no leading whitespace here before # using k = k.lstrip(string.whitespace)
				if k.lstrip(whitespace)[0] == '#':
					break
				else:
					tmp.append(k)
				value_list.append(tmp)

		hvals = self.combine_header_values(value_list)

		return hvals, len(header) - 1

	def combine_header_values(self, vlist):
		"""
		This will take in the str:str value_list and return a list of
		coupled keyword value pairs with correct data type
		"""

		hvals = {}

		for k in vlist:
			if len(k) != 2:
				print(f'ERROR: header item not in k = [key = val] syntax - skipping item : \n {k=}')
				continue

			key = k[0]
			val = k[1]

			if key == 'ncols':
				ncols = int(val)
				hvals.setdefault('ncols', ncols)
			elif key == 'nrows':
				nrows = int(val)
				hvals.setdefault('nrows', nrows)
			elif key == 'xllcorner':
				xllcorner = float(val)
				hvals.setdefault('xllcorner', xllcorner)
			elif key == 'yllcorner':
				yllcorner = float(val)
				hvals.setdefault('yllcorner', yllcorner)
			elif key == 'cellsize':
				try:
					cellsize = int(val)
				except ValueError:
					try:
						cellsize = float(val)
					except ValueError:
						cellsize = val
				hvals.setdefault('cellsize', cellsize)
			elif key == "NODATA_value":
				try:
					nodata = int(val)
				except ValueError:
					try:
						nodata = float(val)
					except ValueError:
						nodata = val

				hvals.setdefault('nodata', nodata)
			else:
				print(f'Line content not recognized, skipping it:\n {k}')

		return hvals

	def read_data(self, fid, rstart=None, rstop=None, cstart=None, cstop=None):
		"""
		Reads a numerical data array from a text file.
		Supports row/column slicing.
		"""
		try:
			# If row/column slicing is required, process manually
			if any(slim is not None for slim in (rstop, cstart, cstop)):
				return np.array(
					[
						[float(i) for i in dataline.split()[cstart:cstop]]
						for dataline in fid.readlines(start=rstart, stop=rstop)
					]
				)

			# Use np.loadtxt when possible for performance
			return np.loadtxt(fid.file, skiprows=rstart or 0, dtype=float)

		except ValueError as e:
			print(f"ERROR: Failed to read array in {fid.file} - {e}")
			return None

	def read_head(self, fpath, tar=None):
		"""Retrieve the header data from a file"""

		fid = TarFile(tar) if tar else File(fpath)

		stopstr = 'NODATA'

		# Output dictionary
		data_dict = {}
		data_dict.setdefault('filename', fpath)
		tmp, nlines = self.read_header(fid, stop=stopstr, inclusive=True)
		for k, v in tmp.items():
			data_dict.setdefault(k, v)

		# Add just a couple of helpful extra entries here
		data_dict.setdefault(
			'xurcorner', data_dict["xllcorner"] + data_dict["cellsize"] * data_dict["ncols"]
		)
		data_dict.setdefault(
			'yurcorner', data_dict["yllcorner"] + data_dict["cellsize"] * data_dict["nrows"]
		)

		return data_dict

	def read(self, fpath, clip=None, tar=None):
		"""
		return dictionary {'filename': filename, 'header variable' : val,...,  'data' : ndarray}
		NB: Clip reads in the raster with extent provided by clip = [xmin, xmax, ymin, ymax]
		It then corrects the header information to the new extent
		NB: llcorner (and urcorner) values may/will be different to that specified by clip because
		clip calculates the minimum extent that contains the requested clip values on the existing grid.
		It is not resampled to force start and end points to match!
		"""

		fid = TarFile(tar) if tar else File(fpath)

		stopstr = 'NODATA'

		# Output dictionary
		data_dict = {}
		data_dict.setdefault('filename', fpath)

		tmp, nlines = self.read_header(fid, stop=stopstr, inclusive=True)
		for k, v in tmp.items():
			data_dict.setdefault(k, v)

		if clip is not None:
			# Read in array to clipped area and adjust header appropriately
			if not isinstance(clip, list) and len(clip) == 4:
				raise RuntimeError("Clip format failed. Clip format is [xmin, xmax, ymin, ymax]")

			xmin, xmax, ymin, ymax = clip

			xur = data_dict["xllcorner"] + data_dict["cellsize"] * data_dict["ncols"]
			yur = data_dict["yllcorner"] + data_dict["cellsize"] * data_dict["nrows"]

			if xmin < data_dict["xllcorner"]:
				print(
					"Warning: clipping raster to below xllcorner is not allowed. Using xllcorner."
				)
				xmin = data_dict["xllcorner"]

			if xmax < xmin:
				print(
					"Warning: clipping raster to xmax below xmin is not allowed. Using xurcorner."
				)
				xmax = data_dict["xllcorner"] + data_dict["cellsize"] * data_dict["ncols"]

			if xmax > xur:
				print(
					"Warning: clipping raster to above xurcorner is not allowed. Using xurcorner."
				)
				xmax = data_dict["xllcorner"] + data_dict["cellsize"] * data_dict["ncols"]

			if ymin < data_dict["yllcorner"]:
				print(
					"Warning: clipping raster to below yllcorner is not allowed. Using yllcorner."
				)
				ymin = data_dict["yllcorner"]

			if ymax < ymin:
				print(
					"Warning: clipping raster to ymax below ymin is not allowed. Using yurcorner."
				)
				ymax = data_dict["yllcorner"] + data_dict["cellsize"] * data_dict["nrows"]

			if ymax > yur:
				print(
					"Warning: clipping raster to above yurcorner is not allowed. Using yurcorner."
				)
				ymax = data_dict["yllcorner"] + data_dict["cellsize"] * data_dict["nrows"]

			# NB: python int a truncation so rounds towards 0
			# NB: Here we clamp to original bounding extent
			jmin = max(0, int((xmin - data_dict["xllcorner"]) / data_dict["cellsize"]))
			jmax = min(
				data_dict["ncols"], int((xmax - data_dict["xllcorner"]) / data_dict["cellsize"])
			)

			# 	int(min(in_rows, in_rows - math.floor((y_min-ly) / cellsize)))
			imin = max(
				0, data_dict["nrows"] - int((ymax - data_dict["yllcorner"]) / data_dict["cellsize"])
			)
			imax = min(
				data_dict["nrows"],
				data_dict["nrows"] - int((ymin - data_dict["yllcorner"]) / data_dict["cellsize"]),
			)

			# shift down by the number of header lines
			data = self.read_data(
				fid, rstart=nlines + 1 + imin, rstop=imax + nlines + 1, cstart=jmin, cstop=jmax
			)

			# NB: uy - row_max*cellsize instad??
			data_dict["yllcorner"] = ymin
			data_dict["xllcorner"] = xmin
			data_dict["nrows"] = int((ymax - ymin) / data_dict["cellsize"])
			data_dict["ncols"] = int((xmax - xmin) / data_dict["cellsize"])

		else:
			data = self.read_data(fid, rstart=nlines + 1)

		# Add some helpful extra entries here

		data_dict.setdefault(
			'xurcorner', data_dict["xllcorner"] + data_dict["cellsize"] * data_dict["ncols"]
		)
		data_dict.setdefault(
			'yurcorner', data_dict["yllcorner"] + data_dict["cellsize"] * data_dict["nrows"]
		)

		# NB: If data none something has gone terribly wrong... Not sure what I want to do here
		if data is not None:
			# 			data_dict.setdefault('mean', np.mean(data))
			# 			data_dict.setdefault('standard-deviation', np.std(data))
			data_dict.setdefault('min', np.min(data))
			data_dict.setdefault('max', np.max(data))
			data_dict.setdefault('data', data)
		else:
			# 			data_dict.setdefault('mean', None)
			# 			data_dict.setdefault('standard-deviation', None)
			data_dict.setdefault('min', None)
			data_dict.setdefault('max', None)
			data_dict.setdefault('data', None)

		return data_dict


class ASCIIFileWriter:
	def __init__(self, dparams={}):
		self.params = {
			"ncols": dparams.get("ncols"),
			"nrows": dparams.get("nrows"),
			"xllcorner": dparams.get("xllcorner"),
			"yllcorner": dparams.get("yllcorner"),
			"cellsize": dparams.get("cellsize"),
			"nodata": dparams.get("nodata"),
		}

	# NB: Deprecated
	def set_default_params(self, dparams):
		"""Alias for update for backwards compatibility"""
		self.update(dparams)

	def update(self, params):
		"""Update params"""
		self.params.update(params)

	def gen_h_str(self):
		"""Generate the header string"""
		out = ""
		out += f"ncols        {self.params['ncols']}\n"
		out += f"nrows        {self.params['nrows']}\n"
		out += f"xllcorner    {self.params['xllcorner']:.2f}\n"
		out += f"yllcorner    {self.params['yllcorner']:.2f}\n"
		out += f"cellsize     {self.params['cellsize']:.1f}\n"
		out += f"NODATA_value {self.params['nodata']}\n"
		return out

	def gen_data_str(self, func):
		"""Generate the data string from an arbitrary func f(i,j)"""
		out = ""
		for i in range(self.params["nrows"]):
			for j in range(self.params["ncols"]):
				try:
					out += f"{func(i,j)}"
					out += " " if j != self.params["ncols"] - 1 else "\n"
				except RuntimeError:
					raise OSError("nrows or ncols not correctly set for output array.")
		return out

	def gen_str(self, func):
		"""Generate the full file string from an arb. func f(i,j)"""
		if self.params is None:
			raise RuntimeError("Header parameters must be set prior to writeout.")
		return self.gen_h_str() + self.gen_data_str(func)

	def makefile(self, fn, ft, func):
		"""Write an ASCII file from a function f(i,j)"""
		fid = File(fn)
		string = self.gen_str(func)
		fid.write(string)

	def write(self, fn, dat):
		"""
		Write an ASCII file from the ASCIIFileReader output structure.
		`dat` should be a dictionary containing header fields and a 'data' key
		with the 2D data array.
		"""

		required_keys = {"ncols", "nrows", "xllcorner", "yllcorner", "cellsize", "nodata", "data"}
		if not required_keys.issubset(dat):
			raise ValueError(f"Missing required keys: {required_keys - dat.keys()}")

		# Update writer params with header from dat
		self.set_default_params(dat)

		# Validate shape
		nrows, ncols = dat["data"].shape
		if nrows != self.params["nrows"] or ncols != self.params["ncols"]:
			raise ValueError(
				f"Data shape mismatch: expected {(self.params['nrows'], self.params['ncols'])}, got {(nrows, ncols)}"
			)

		# Write header
		out = self.gen_h_str()

		# Write data
		arr = dat["data"]
		for i in range(nrows):
			row = arr[i]
			out += (
				" ".join(
					f"{val:.2f}" if not np.isnan(val) else str(self.params["nodata"]) for val in row
				)
				+ '\n'
			)

		# Save to file
		fid = File(fn)
		fid.write(out)
