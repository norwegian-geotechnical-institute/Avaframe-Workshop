#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read all paths filter and sort appropriately and simply...
"""

import os
import re
import numpy as np

from .FileUtils import ReaderUtilities
from .FileReaders import *


# File parser for solution sets from MoT Codes
class FileParser(ReaderUtilities):
	"""
	File Reader provides methods for reading a solver directory tree from an entry point
	and returns the structured data -- see header to this file

	Essentially this class lays down the routine for pulling together the dataset for animations
	and analysis in the plotting tools
	"""

	def __init__(self):
		ReaderUtilities.__init__(self)

	def filterodirfn(self, paths, odirfn):
		"""Filter paths by the directories that match the output directory fieldname value"""

		# TODO: filterodirfn
		# Split paths to find the odir with the fieldname present
		# If fn matches we keep

		pass

	def filtertimesteps(self, paths, timesteps):
		"""Filter paths of timestep data files to match given range - keep any that match"""

		keep = lambda x: self.istimestepfile(x) and self.properties(x)[2] in timesteps
		paths = [p for p in paths if keep(p)]
		return paths

	def filterfieldnames(self, paths, fieldnames):
		"""
		Keep any files with a fieldname that matches keys

		Partial strip of paths where we can find out if fieldname present
		Fieldname will be in the path if not timestep pattern txt file
		in which case we have to read the header to know.
			- can do this here then if not in header - remove
		where though is complicated - schema is subdir but we also want to catch input ava_h and ava_dem
		or in the case of txt file - the name of a variable we can calculate e.g. u,v,p etc. will not be in the header
		"""

		keep = lambda x: self.properties(x)[0] in fieldnames
		paths = [p for p in paths if keep(p)]
		return paths

	def filterfiletypes(self, paths, filetypes):
		"""Filter by filetype - keep any that match"""

		keep = lambda x: self.properties(x)[1] in filetypes
		paths = [p for p in paths if keep(p)]
		return paths

	def filterignore(self, paths, ignore):
		"""Filter by simple ignore pattern e.g. *.svg or a filename"""

		# Hardcoded always ignore...
		ignorelist = [".DS_Store", "cmdsummary.txt", "*.svg"]

		if ignore is not None:
			ignorelist.extend(ignore)

		# Add a simple pattern to catch e.g. ignore all svg's with "*.svg"
		ignorepattern = [i.lstrip('*') for i in ignorelist if i[0] == "*"]

		ignore = lambda x: os.path.basename(x) in ignorelist or any(
			x.find(ip) > 0 for ip in ignorepattern
		)

		return [p for p in paths if not ignore(p)]

	def order_files(self, paths):
		'''
		Sort by non-timestep files - timestep - filetype - fieldname
		'''
		pathinfo = [(p, *self.properties(p)) for p in paths]

		out = [comb[0] for comb in sorted(pathinfo, key=lambda x: (x[3], x[2], x[1]))]
		return out

	def filterpaths(self, paths, **filters):
		'''
		Here we strip out any paths that do not contain the data then order the paths
		'''

		timesteps = filters.get("timesteps", None)
		fieldnames = filters.get("fieldnames", None)
		filetypes = filters.get("filetypes", None)
		level = filters.get("level", None)
		ignore = filters.get("ignore", None)
		odirfn = filters.get("odirfn", None)

		# Filter by output dir name field name
		if odirfn is not None:
			paths = self.filterodirfn(paths, odirfn)

		# Filter by timesteps
		if timesteps is not None:
			paths = self.filtertimesteps(paths, timesteps)

		# Filter by fieldnames
		if fieldnames is not None:
			paths = self.filterfieldnames(paths, fieldnames)

		# Filter by filetypes
		if filetypes is not None:
			paths = self.filterfiletypes(paths, filetypes)

		# Always ignore some paths - .DS_Store etc.
		paths = self.filterignore(paths, ignore)

		# If no files left return empty list
		if paths == []:
			return []

		# Order the files
		return self.order_files(paths)

	def parsepaths(self, ipath, **filters):
		"""Get "zipped" lists of paths, timesteps, filenames, filetypes after filters applied"""

		if os.path.isdir(ipath):

			# If filtering to a specific directory depth from root
			level = filters.get("level", None)

			# Walk the directory
			paths = []
			for root, dirs, files in self.walk(ipath, level=level):
				for f in files:
					paths.append(os.path.join(root, f))

			filteredpaths = self.filterpaths(paths, **filters)
		else:
			# If path is a file
			filteredpaths = [ipath]

		# Split output to paths, timesteps, fieldnames, filetypes
		opaths, ots, ofn, oft = [], [], [], []
		for fp in filteredpaths:
			fn, ft, ts = self.properties(fp)
			opaths.append(fp)
			ots.append(ts)
			ofn.append(fn)
			oft.append(ft)

		return opaths, ots, ofn, oft

	def nframes(self, root, **filters):
		"""
		Return the number of frames read in with corresponding filters
		Input must match that of read to ensure the same number of frames!
		NB: frames may be different to timesteps e.g. with the case where we
		load in the start and end timesteps

		frame 0 - dem, maxh, maxs, maxp, ava_h_0000.asc -> ts = 0
		frame 1 - ava_h_0300.asc -> ts = 300
		"""

		_, ots, _, _ = self.parsepaths(root, **filters)
		uniquetimesteps = list({ts for ts in ots})
		return len(uniquetimesteps)

	def timestepmap(self, root, **filters):
		"""Get the timestep to frame map of read in with corresponding filters"""

		_, ots, _, _ = self.parsepaths(root, **filters)
		uniquetimesteps = list({ts for ts in ots})
		return {ts: i for i, ts in enumerate(uniquetimesteps)}

	def reader(self, ftype):
		"""Factory method for various file readers"""

		if ftype == 'asc':
			return ASCIIFileReader()
		elif ftype == 'txt':
			return TXTFileReader()
		elif ftype == 'bt':
			return BinaryTerrainFileReader()
		elif ftype == 'tif' or ftype == "tiff":
			return TIFFileReader()
		elif ftype == 'cmf' or ftype == "rcf":
			return CMDFileReader()
		elif ftype == '1D':
			return OneDOutputFileReader()
		else:
			return BaseFileReader()

	def readhead(self, fpath, filetype=None):
		"""
		Read a particular file header auto determining reader from filetype
		return dictionary with header vals
		"""

		# Choose reader from filetype
		if filetype is None:
			_, ft, _ = self.properties(fpath)
		else:
			ft = filetype

		reader = self.reader(ft)
		return reader.readhead(fpath)

	def readfile(self, fpath, filetype=None):
		"""
		Read particular file auto determining read from filetype
		return dictionary with header info and data array
		"""

		# Optional: Choose reader from filetype else derived
		if filetype is None:
			_, ft, _ = self.properties(fpath)
		else:
			ft = filetype

		reader = self.reader(ft)
		return reader.read(fpath)

	def read(
		self,
		ipath,
		timesteps=None,
		fieldnames=None,
		filetypes=None,
		level=None,
		ignore=None,
		odirfn=None,
	):
		'''
		read() : return a dat[timestep][fieldname][var] ordered data structure

		Filters strip the possible paths to read by their value e.g. filter files with options
		chosen by timesteps, fieldnames, filetypes, level (dir search depth), ignore

		if ipath is file :
			read file to get dat[0][fieldname | filename][var]

		if ipath is dir :
			read dir to get dat[ts][fieldnames | filenames][var]

		if ipath is file list :
			read each file to get [dat[0][fieldnames | filename][var], dat[0][fieldnames | filename][var], ...]

		if ipath is dir list :
			read each file to get [dat[ts][fieldnames | filenames][var], dat[ts][fieldnames | filenames][var], ...]
		'''

		# Collect filters
		filters = {
			"timesteps": timesteps,
			"fieldnames": fieldnames,
			"filetypes": filetypes,
			"level": level,
			"ignore": ignore,
			"odirfn": odirfn,
		}

		# Fetch the zipped paths and path info from input tree
		paths, tslist, fnlist, ftlist = self.parsepaths(ipath, **filters)

		# Construct a map from timestep to frame
		uniquetimesteps = list({ts for ts in tslist})
		tsmap = {ts: i for i, ts in enumerate(uniquetimesteps)}

		dat = [{} for _ in range(len(uniquetimesteps))]
		for path, ts, fn, ft in zip(paths, tslist, fnlist, ftlist):
			try:
				data = self.readfile(path)
			except RuntimeError:
				print(f"could not read file {path}")
				continue

			# Map the timestep to the right placement
			dat[tsmap[ts]].setdefault(fn, None)

			if dat[tsmap[ts]][fn] is None:
				dat[tsmap[ts]][fn] = data
			else:
				# Non-unique fieldname for this file -
				raise RuntimeError(f"Non-unique fieldname for path {path}")

		return dat
