#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Reader Utilities
"""

import os, re
import codecs

# File wrapper for read write methods
class File:
	'''
	File class: Simplifies file reading / writing tasks with commands and context control
	Specifically overload readlines for extra control
	'''

	def __init__(self, path):
		self.file = path

	def read(self):
		"""Read whole file"""

		with open(self.file, 'r') as fid:
			filestr = fid.read()
		return filestr

	def readlines(self, start=0, stop=None, pos=False, inclusive=False):
		'''
		Returns: list of line strings

		The idea of this function is we can split file into sections to be read separately (e.g. header then data)
		without reading (and appending) the whole file each time - as opposed to the built-in readlines function.

		Notes on functionality :

		start = int, stop = (int | str), pos = bool, inclusive = bool

		If start not specified we start at line 0.
		If start is specified we seek to line start and then append from line start.

		If stop not specified we continue to end of file.
		If stop is an integer
			- we break from loop at line stop.
		If stop is a string
			- we enter while loop and stop at a conditional specified by the string
			e.g. stop at first instance of '# 	X' (note not inclusive of this line by default).
		NB: stop is specified s.t. we stop at 1 less as counting from 0 (i.e. like range(num)).
		If inclusive (default false) is true then we do include the stopping entry but the position pointer
		stays at the same place

		If pos is selected then we additionally return the bytes position the read ends. The enables us
		to swap readin methods if we have a human readable header, for example.
		'''

		lines = []
		if isinstance(stop, int):

			if inclusive == True:
				stop += 1

			with open(self.file, 'r') as fid:
				count = 0
				for i in range(stop):
					line = fid.readline()
					if not line:
						break
					if count >= start:
						lines.append(line)
					count += 1
				pos_ = fid.tell()
		else:
			if isinstance(stop, str):
				condition = lambda line: stop in line
			else:
				condition = lambda line: False

			count = 0
			with open(self.file, 'r') as fid:
				while True:
					line = fid.readline()
					if not line or condition(line):
						if inclusive and count >= start:
							lines.append(line)
						break
					if count >= start:
						lines.append(line)
					count += 1
				pos_ = fid.tell()

		if pos == True:
			return lines, pos_
		else:
			return lines

	# Writes a string to file
	def write(self, str2write):
		with open(self.file, 'w') as fid:
			fid.write(str2write)

	# Appends a strin to file
	def append(self, str2write):
		with open(self.file, 'a') as fid:
			fid.write(str2write)

	# Read bytes from a file
	def read_bytes(self, startbyte=0):
		with open(self.file, 'rb') as fid:
			fid.seek(startbyte, 0)
			out = fid.read()
		return out

	# Write bytes to a file
	def write_bytes(self, str2write):
		with open(self.file, 'wb') as fid:
			fid.write(str2write)

	# Append bytes to a file
	def append_bytes(self, str2write):
		with open(self.file, 'ab') as fid:
			fid.write(str2write)


# File wrapper for tar archives
class TarFile:
	'''
	Specialises file reading tasks when reading a file from within a tar archive
	Overloads read and readlines methods to allow decoding
	'''

	# NB tfiles are already open so here we need to preserve this
	def __init__(self, tfile):
		self.fid = tfile

	# Read whole file
	def read(self):
		return codecs.decode(self.fid.read(), encoding='utf-8')

	# Read line by line
	def readlines(self, start=0, stop=None, pos=False, inclusive=False):
		'''
		Returns: list of line strings

		The idea of this function is we can split file into sections to be read separately (e.g. header then data)
		without reading (and appending) the whole file each time - as opposed to the built-in readlines function.

		Notes on functionality :

		start = int, stop = (int | str), pos = bool, inclusive = bool

		If start not specified we start at line 0.
		If start is specified we seek to line start and then append from line start.

		If stop not specified we continue to end of file.
		If stop is an integer
			- we break from loop at line stop.
		If stop is a string
			- we enter while loop and stop at a conditional specified by the string
			e.g. stop at first instance of '# 	X' (note not inclusive of this line by default).
		NB: stop is specified s.t. we stop at 1 less as counting from 0 (i.e. like range(num)).
		If inclusive (default false) is true then we do include the stopping entry but the position pointer
		stays at the same place

		If pos is selected then we additionally return the bytes position the read ends. The enables us
		to swap readin methods if we have a human readable header, for example.
		'''

		lines = []
		if isinstance(stop, int):

			if inclusive == True:
				stop += 1

			count = 0
			for i in range(stop):
				line = codecs.decode(self.fid.readline(), encoding='utf-8')
				if not line:
					break
				if count >= start:
					lines.append(line)
				count += 1
			pos_ = fid.tell()
		else:
			if isinstance(stop, str):
				condition = lambda line: stop in line
			else:
				condition = lambda line: False

			count = 0

			while True:
				line = codecs.decode(self.fid.readline(), encoding='utf-8')
				if not line or condition(line):
					if inclusive and count >= start:
						lines.append(line)
					break
				if count >= start:
					lines.append(line)
				count += 1
			pos_ = self.fid.tell()

		if pos == True:
			return lines, pos_
		else:
			return lines


# Base class providing utilities for file readers
class ReaderUtilities:
	"""Class defines utility functions shared between readers"""

	def __init__(self):
		self.asctimesteppattern = re.compile(r'.*\w\_(\d{4}).asc')
		self.txttimesteppattern = re.compile(r'.*(\d{6}).txt')

	def isnumber(self, s):
		"""Trick to check if is number - handles 1e-6 etc. as well without special treatment"""
		try:
			float(s)
			return True
		except ValueError:
			return False

	def istimestepfile(self, fpath):
		"""Check if file path matches timestep file patterns"""
		return any(v.search(fpath) for v in [self.asctimesteppattern, self.txttimesteppattern])

	def properties(self, fpath):
		"""Get the properties from the file name e.g. fieldname, ftype, timestep"""

		ftype = fpath.split('.')[-1]

		filename = os.path.basename(fpath)[: -len(ftype) - 1]

		ascval = self.asctimesteppattern.search(fpath)
		txtval = self.txttimesteppattern.search(fpath)

		if ascval:
			timestep = int(ascval.group(1))
		elif txtval:
			timestep = int(txtval.group(1))
		else:
			timestep = 0

		# Next we search for a field name
		# If timestep file its the (sub) directory name
		# If not then its file name (basename)
		if ascval or txtval:
			fname = os.path.split(os.path.dirname(fpath))[-1]
		else:
			fname = filename

		return fname, ftype, timestep

	def walk(self, dirpath, level=None):
		"""
		Generator for os walk method which restricts to a directory level
		returns (root, dirs, files) tuple iterator
		"""

		assert os.path.isdir(dirpath)

		# CHECK: remove trailing '/' or go up one level?
		dirpath = dirpath.rstrip(os.path.sep)

		# count the number of separators for base level
		num_sep = dirpath.count(os.path.sep)

		# Do walk and yield results
		# Following calls will update the base count for next path, calc. difference and overwrite dirs
		# thus restricting output to the requested level
		for root, dirs, files in os.walk(dirpath):
			yield root, dirs, files
			num_sep_this = root.count(os.path.sep)
			if level is not None and num_sep + level <= num_sep_this:
				del dirs[:]
