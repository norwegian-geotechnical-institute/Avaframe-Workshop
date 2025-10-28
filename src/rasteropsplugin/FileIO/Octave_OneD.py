#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File Reader for OneD outputs of the OneD Octave and python codes based on SDKT
"""

import numpy as np
import struct

from .FileUtils import ReaderUtilities, File, TarFile


# TODO: convert output format as .asc1d files - or at least make distinct somehow...
class OneDOutputFileReader(ReaderUtilities):
	def __init__(self, path):
		self.name = "OneDOutputFileReader"
		self.datareadmode = 'text'
		ReaderUtilities.__init__(self)

	def read_header(self, fid, start=0, stop=None):
		header = fid.readlines(start=start, stop=stop)

		value_list = []
		for v in header:
			value_list.append(v.split())

		hvals = self.combine_header_values(value_list)

		return hvals, len(header)

	def combine_header_values(self, vlist):
		'''
		This will take in the value_list and return a list of
		coupled keyword value pairs with correct data type
		Header must have keys
			inx, iny inames infields
			onx, ony onames onfields
		Format below

		fname = ./Input/...
		friction = Voellmy
		em_f = none
		em_b = none
		curv = no
		baseodir = ''
		plt_list = ['h0']
		g = 9.81
		rho_b = 140
		rho_f = 200
		rho_a = 1
		ds = 5
		t_ini = 0
		t_end = 100
		cfl = 0.24
		dt_max = 1
		dt_min = 1e-10
		mmtheta = 1.3
		dt_plt = 1
		dt_out = 1
		iarray = 110 x 10
		X         Y         Z         W         tau_c     b0        h0        u0        mu        k
		oarray = 860 x 14
		X         Y         Z         L         W         sint      cost      kappa     tau_c     b0        h0        u0        mu        k
		'''

		hvals = {}
		for k in vlist:

			key = k[0]

			if key == 'X':
				val = k[:]
			else:
				val = k[2:]

			if key in {"fname", "fiction", "em_f", "em_b", "curv", "baseodir", "plt_list"}:
				hvals.setdefault(key, val[0])
			elif key in {
				"g",
				"rho_b",
				"rho_f",
				"rho_a",
				"ds",
				"t_ini",
				"t_end",
				"cfl",
				"dt_max",
				"dt_min",
				"mmtheta",
				"dt_plt",
				"dt_out",
			}:
				hvals.setdefault(key, float(val[0]))
			elif key == 'iarray':
				hvals.setdefault('inx', int(val[0]))
				hvals.setdefault('iny', int(val[2]))
				hvals.setdefault('infields', int(val[4]))
			elif key == 'oarray':
				hvals.setdefault('onx', int(val[0]))
				hvals.setdefault('ony', int(val[2]))
				hvals.setdefault('onfields', int(val[4]))
			elif key == 'X':
				if len(val) == hvals['infields']:
					hvals.setdefault('inames', val)
				elif len(val) == hvals['onfields']:
					hvals.setdefault('onames', val)
			else:
				print(f'Line content not recognized, skipping it:\n {k}')

		return hvals

	def read_data_input(self, fid, header, start=0, stop=None):

		data = fid.readlines(start=start, stop=stop)

		var = header['inx'] * header['iny'] * header['infields']
		names = header["inames"]

		# if var != len(data) * len(data[0].split()):
		#      print(f'ERROR: number of variable elements in {header["inames"]} does not match data array size')
		#      exit()

		darray = np.empty((header['inx'], header['iny'], header['infields']))
		for i in range(header['inx']):
			d = np.array([float(n) for n in data[i].split()])
			darray[i, 0, :] = d

		return darray

	def read_data_output(self, fid, header, start=0, stop=None):
		data = fid.readlines(start=start, stop=stop)

		var = header['onx'] * header['ony'] * header['onfields']
		names = header["onames"]

		# if var != len(data) * len(data[0].split()):
		# print(f'ERROR: number of variable elements in {header["onames"]} does not match data array size')
		# exit()

		darray = np.empty((header['onx'], header['ony'], header['onfields']))
		for i in range(header['onx']):
			d = np.array([float(n) for n in data[i].split()])
			darray[i, 0, :] = d

		return darray

	def read(self, fpath, tar=None, nhl=24):
		"""Reads the OneD  (python and Matlab / Octave) solver output ascii text type results files"""
		fid = TarFile(tar) if tar else File(fpath)

		data_dict, nlines = self.read_header(fid, stop=nhl)

		idata = self.read_data_input(fid, header, start=nlines, stop=nhl + out['inx'])
		odata = self.read_data_output(
			fid, header, start=nlines + out['inx'] + 1
		)  # stop=nlines + out['inx'] + 1 + out['onx']

		data_dict.setdefault('idata', idata)
		data_dict.setdefault('odata', odata)

		return out
