#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper for CSV file reading with a small amount of formatting to convert numeric strings etc.
"""

import csv

from .FileUtils import ReaderUtilities, File, TarFile


class CSVReader(ReaderUtilities):
	def __init__(self):
		self.name = "CSVReader"
		ReaderUtilities.__init__(self)

	def convert_value(self, value):
		"""Converts a string to float if numeric, otherwise returns the string or None if empty."""
		value = value.strip()  # Remove leading/trailing whitespace
		if value == "":
			return None

		if self.isnumber(value):
			return float(value)
		else:
			return value

	def read(self, fpath, **kwargs):
		"""Wrapper for pythons own csv reader - just packages into dict and tries to simply handle numeric types"""

		data = {}
		with open(fpath, mode='r', newline='', encoding='utf-8') as file:
			reader = csv.DictReader(file, **kwargs)
			# Initialize dictionary keys with empty lists
			for field in reader.fieldnames:
				data[field] = []

			# Populate dictionary with column values
			for row in reader:
				for key, value in row.items():
					val = self.convert_value(value)
					if val is not None:
						data[key].append(val)

		return data
