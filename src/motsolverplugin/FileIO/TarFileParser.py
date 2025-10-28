#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Read all paths filter and sort appropriately and simply...
"""

import os
import numpy as np

from .FileUtils import ReaderUtilities
from .FileReaders import *

# File parser for tar archives
class TarFileParser(ReaderUtilities):
	"""
	Make a routine for parsing a large dataset from a tar archive
	"""

	def __init__(self):
		ReaderUtilities.__init__(self)

	def reader(self, ftype):
		"""
		Factory method for reader types
		"""

		if ftype == 'asc':
			return ASCIIFileReader()
		elif ftype == 'txt':
			return TXTFileReader()
		elif ftype == 'bt':
			return BinaryTerrainFileReader()
		elif ftype == 'tif':
			return TIFFileReader()
		elif ftype == 'cmf':
			return CMDFileReader()
		elif ftype == '1D':
			return OneDOutputFileReader()
		else:
			return BaseFileReader()

	# TODO: This is now in Batch postprocess methods
	def stripinfo(self, dirpath, lfid=None):
		"""
		Get all the info out of the path string

		e.g. plane_dh500_theta30_vol16766d0_fricconst_eronone_depnone_forestnone

		Also read log and find out how sim exited
		"""

		# Local namespace (Needed for eval)
		from numpy import array

		info = {}

		strinfo = dirpath.split("_")

		# Hack to fix channeled_parabola _ mistake
		if strinfo[0] == "channeled":
			entry = strinfo.pop(0)
			strinfo[0] = entry + strinfo[0]

		simexit = "UNKNOWN"
		methodparams = {}

		try:
			logfile = os.path.join(dirpath, "log.lg")

			#### FIX HERE ######
			if lfid is not None:
				logs = fid.readlines()
			else:
				with open(logfile, "r") as fid:
					logs = fid.readlines()

			for line in logs:
				words = line.split()
				if words[0] == "MoTVoellmy:":
					simexit = " ".join(words[1:])
				elif words[0] == "Method":
					dictstr = " ".join(words[2:])
					methodparams = eval(dictstr)

		except FileNotFoundError as e:
			# 			print(f"Log file not found in {dirpath}!")
			pass

		info.setdefault("fname", str(dirpath))
		info.setdefault("SimulationExit", simexit)
		info.setdefault("Topography", strinfo[0])
		info.setdefault("Cellsize", float(strinfo[1][2:]))
		info.setdefault("Dropheight", float(strinfo[2][2:]))
		info.setdefault("ReleaseAngleDegs", float(strinfo[3][5:]))
		info.setdefault("ReleaseVolume", float(strinfo[4].replace("d", ".")[3:]))
		info.setdefault("Friction", str(strinfo[5][4:]))
		info.setdefault("Erosion", str(strinfo[6][3:]))
		info.setdefault("b", float(strinfo[7][1:]))
		info.setdefault("tauc", int(strinfo[8][4:]))
		info.setdefault("Deposition", str(strinfo[9][3:]))
		info.setdefault("Forest", str(strinfo[10][6:]))

		return info, methodparams

	def getlog(self, tinfo):
		# TODO: return the logfile fid for this filename
		lname = os.path.join(os.path.dirname(tinfo.name), "log.lg")
		return None

	def filter(self, tinfo, loadinput=False, expfilter={}):
		"""
		e.g. I only need the maxes and dem h
		"""

		# NB: MAC adds a strange ._ file to tar archives so finder can position elements
		ismacosdouble = os.path.basename(tinfo.name)[:2] == '._'

		# Skip over macos doubles
		if ismacosdouble:
			return False

		# Skip over directories
		if tinfo.isdir():
			return False

		name = tinfo.name
		nele = name.split(os.path.sep)
		fmt = nele[-1].split(".")[-1]

		# ignore any extra files stored in the tar e.g. images etc.
		isnotasciiorlogorcmd = fmt not in {"cmf", "lg", "asc"}
		if isnotasciiorlogorcmd:
			return False

		# Here we want to restrict load to a global output e.g. max file or an Input file e.g. dem or rel area
		# TODO: to generalise this probs need a different schema...
		isnotinputfile = len(nele) > 2 and nele[-2] != "Input"
		isinputfile = len(nele) > 2 and nele[-2] == "Input"

		# Skip over all input files if desired
		if isinputfile and not loadinput:
			return False

		# ignore file that is neither a max file or a input file
		isextrafile = (
			all(i not in os.path.basename(name) for i in expfilter["filetypes"]) and isnotinputfile
		)
		if isextrafile:
			return False

		# TODO: Get log info for this file
		# NB: currently tricky to get at the logfile here because we are parsing by file and not by group
		# so can't currently filter by info contained in the logfile
		# I kinda wanted to be able to do it to read in the minimal amount of files I can and restrict further filtering...
		# so I guess doing it this way round does that for most things as we only get the requested files by filtering existing
		# files
		lfid = self.getlog(tinfo)

		info, _ = self.stripinfo(nele[1], lfid=lfid)

		# Filter anything from the experiment naming convention
		for k, v in expfilter.items():
			if k == "filetypes":
				continue
			if info[k] != v:
				return False

		# Keep any file that fullfils this schema
		return True

	def parsetar(self, tar, expfilter={}):
		"""Generator yielding tarfile handles one by one"""

		for tinfo in tar:
			if self.filter(tinfo, expfilter=expfilter):
				yield tinfo

	def extract(self, tarpath, expfilter={}):
		"""Extract any members according to a filter"""

		with tarfile.open(tarpath, "r:gz") as tar:
			tar.extractall(members=self.parsetar(tar, expfilter=expfilter))

	def readfile(self, fpath, tfile, filetype=None):
		"""Read file within the tar archive"""

		if filetype is None:
			_, ft, _ = self.properties(fpath)
		else:
			ft = filetype

		reader = self.reader(ft)
		return reader.read(fpath, tar=tfile)

	def execfunc(self, args):

		# This gives us a handle to the file within the tar archive
		i, tar, tinfo = args

		f = tar.extractfile(tinfo)
		data = self.readfile(tinfo.name, f)

		# extract the info from the directory naming convention
		nele = tinfo.name.split(os.path.sep)
		info, _ = self.stripinfo(nele[1])

		data.setdefault("info", info)

		return data

	def igen(self, tar, gen):
		# Generate all args and a run number
		for i, out in enumerate(gen):
			yield (i, tar, out)

	def read(self, tarpath, expfilter={}):
		"""Open and read files in the tar archive"""

		# To be nice
		niceness = 3

		dat = []
		with tarfile.open(tarpath, "r:gz") as tar:

			# Attempt at multiprocessing this
			# 			with Pool(6, initializer=os.nice, initargs=(niceness,)) as pool:
			# 				genargs = self.igen(tar,self.parsetar(tar,expfilter))
			##				for res in pool.imap_unordered(self.execfunc,genargs) :
			# 				for res in pool.map(self.execfunc,genargs) :
			# 					dat.append(res)

			for i, tinfo in enumerate(self.parsetar(tar, expfilter)):
				# This gives us a handle to the file within the tar archive
				f = tar.extractfile(tinfo)
				data = self.readfile(tinfo.name, f)

				# extract the info from the directory naming convention
				nele = tinfo.name.split(os.path.sep)
				info, _ = self.stripinfo(nele[1])

				data.setdefault("info", info)

				# Show progress in terminal
				print(i, ": ", tinfo.name)
				dat.append(data)

		return np.array(dat)
