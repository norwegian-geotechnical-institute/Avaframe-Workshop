#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Reader for the old txt and binary output files for the SDKT code
"""

import numpy as np
import struct

from .FileUtils import ReaderUtilities


class TXTFileReader(ReaderUtilities):
	"""
	Reads the C++ solver txt file output for both ASCII-text and binary formats

	Format: 10 human readable metadata lines followed by ncols * nrows * nfields
			data arrays
	"""

	def __init__(self):
		self.name = "TXTFileReader"
		self.nhl = 10  # header lines
		ReaderUtilities.__init__(self)

	def read_header(self, fid):
		"""Reads text header from binary stream, returns metadata and byte offset."""
		header_lines = []
		byte_offset = 0

		for _ in range(self.nhl):
			line = b""
			while True:
				c = fid.read_bytes(1)
				byte_offset += 1
				if not c:
					break  # EOF
				line += c
				if c == b"\n":
					break
			try:
				decoded = line.decode("utf-8").strip()
			except UnicodeDecodeError as e:
				raise ValueError(f"Header decoding failed: {e}")
			header_lines.append(decoded)

		value_list = [line.split() for line in header_lines if line.strip()]
		hvals = self.combine_header_values(value_list)
		return hvals, byte_offset

	def combine_header_values(self, vlist):
		"""
		This will take in the value_list and return a list of
		coupled keyword value pairs with correct data type
		"""

		hvals = {}

		for k in vlist:
			key = k[0]
			val = k[1:]

			if key == '1':
				hvals.setdefault('nfields', int(val[0]))
				hvals.setdefault('nx', int(val[1]))  # ncols
				hvals.setdefault('ny', int(val[2]))  # nrows

			elif key == '2':
				xmin = float(val[0])
				ymin = float(val[1])
				xsize = float(val[2])
				ysize = float(val[3])
				time = float(val[4])
				cflno = float(val[5])

				xGrid = [xmin + xsize / hvals["ncols"] * (i + 0.5) for i in range(hvals["ncols"])]
				yGrid = [ymin + ysize / hvals["nrows"] * (i + 0.5) for i in range(hvals["nrows"])]

				hvals.setdefault('xmin', xmin)
				hvals.setdefault('ymin', ymin)
				hvals.setdefault('xsize', xsize)
				hvals.setdefault('ysize', ysize)
				hvals.setdefault('time', time)
				hvals.setdefault('cflno', cflno)
				hvals.setdefault('xGrid', xGrid)
				hvals.setdefault('yGrid', yGrid)

			elif key == '3':
				hvals.setdefault('File written', ' '.join(val))  # Timestamp

			elif key == '4':
				hvals.setdefault(
					'Code run', ' '.join(val)
				)  # Simulation time stamp - code start time

			elif key == '5':
				hvals.setdefault('fileindex', int(val[0]))

			elif key == '6':
				# From old implementation - oldFr - ignore
				pass

			elif key == '7':
				params = {}
				if len(val) != 0:
					assert len(val) % 2 == 0, 'Parameters not in keyword value pairs'

					for i in range(0, len(val), 2):
						if val[i] == 'TransitionType':
							params.setdefault(val[i], val[i + 1])
							continue
						if val[i + 1] not in 'true false':
							params.setdefault(val[i], float(val[i + 1]))
						else:
							params.setdefault(val[i], val[i + 1])
				hvals.setdefault('params', params)

			elif key == '8':
				if val[0] == '0':
					hvals.setdefault('nExtras', 0)
				elif val[0] == '-1':
					# indicates nextras comes after data in binary format
					hvals.setdefault('nExtras', int(val[1]))
				elif int(val[0]) > 0:
					hvals.setdefault('nExtras', int(val[0]))
					hvals.setdefault('Extras', [float(i) for i in val[1:]])
				else:
					pass

			elif key == '9':
				hvals.setdefault('names', [i[1:-1] for i in val])

			elif key == '10':
				if val[0] == '0':
					hvals.setdefault('datareadmode', "text")
				else:
					hvals.setdefault('datareadmode', "binary")
			else:
				print(f'Line content not recognized, skipping it:\n {k}')

		return hvals

	def read_data(self, fid, header, start=0):
		"""Read data based on format: 'text' or 'binary'"""

		ncols, nrows, nfields = header['nx'], header['ny'], header['nfields']

		if header.get("datareadmode", "text") == "text":
			lines = fid.readlines(start=start)
			floats = list(map(float, " ".join(lines).split()))
		else:
			num_values = ncols * nrows * nfields
			data_bytes = fid.read_bytes(num_values * 4)  # float32 = 4 bytes
			floats = struct.unpack(f"{num_values}f", data_bytes)

		return np.array(floats, dtype=np.float32).reshape((ncols, nrows, nfields), order='F')

	def read(self, fpath, tar=None):
		"""Read the original C++ results file in either text or binary modes"""

		fid = TarFile(tar) if tar else File(fpath)

		hvals = self.read_header(fid)
		data = self.read_data(fid, hvals, start=nhlines)
		hvals.update({"filename": fpath, "data": data})

		return hvals
