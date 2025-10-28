#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Reader for BinaryTerrain (.bt) filetypes
"""

import numpy as np
import struct

from .FileUtils import ReaderUtilities, File, TarFile

# NB: Untested
class BinaryTerrainFileReader(ReaderUtilities):
	def __init__(self):
		self.name = "BinaryTerrainFileReader"
		ReaderUtilities.__init__(self)

	def read_header(self, fpath, tar=None):
		"""Reads the 256-byte header of a .bt file and returns a metadata dictionary."""
		if tar is not None:
			fid = TarFile(tar)
		else:
			fid = File(fpath)

		header = fid.read_bytes(0)[:256]

		# Unpack known structure from header
		metadata = {}
		(
			metadata["ncols"],
			metadata["nrows"],
			metadata["bytes_per_sample"],
			datatype,
		) = struct.unpack("<4H", header[:8])
		metadata["datatype"] = {1: 'int16', 2: 'float32'}.get(datatype, 'unknown')
		(
			metadata["west"],
			metadata["east"],
			metadata["south"],
			metadata["north"],
			metadata["vscale"],
		) = struct.unpack("<5d", header[8:48])

		return metadata

	def read(self, fpath, tar=None):
		"""Reads both the header and elevation data from a .bt file."""
		meta = self.read_header(fpath, tar)

		fid = TarFile(tar) if tar else File(fpath)

		raw = fid.read_bytes(256)  # skip 256-byte header

		shape = (meta["nrows"], meta["ncols"])
		dtype = {"int16": np.int16, "float32": np.float32}.get(meta["datatype"], np.float32)
		data = np.frombuffer(raw, dtype=dtype).reshape(shape)

		out = {**meta, "filename": fpath, "data": data}
		return out
