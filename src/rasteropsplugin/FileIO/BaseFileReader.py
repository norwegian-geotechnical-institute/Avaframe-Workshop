#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BaseFileReader tries to read file (binary or ascii text) and output all contents - no structuring
"""

import os, re
import codecs
import numpy as np
import struct
import csv

from .FileUtils import ReaderUtilities, File, TarFile


class BaseFileReader(ReaderUtilities):
	def __init__(self):
		self.name = "BaseFileReader"
		ReaderUtilities.__init__(self)

	def read(self, fpath, tar=None):
		if tar is not None:
			fid = TarFile(tar)
		else:
			fid = File(fpath)
		try:
			return fid.read()
		except UnicodeDecodeError:
			try:
				return codecs.decode(fid.read_bytes(), encoding='utf-8')
			except UnicodeDecodeError as e:
				raise RuntimeError(f"Could not read {fpath}") from e
