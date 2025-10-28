#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Registry for available file readers 
"""

from .BaseFileReader import BaseFileReader
from .BinaryTerrain import BinaryTerrainFileReader
from .CMDFile import CMDFileReader
from .CSV import CSVReader
from .ESRI_ASCII_Grid import ASCIIFileReader
from .GeoTiff import TIFFileReader
from .Octave_OneD import OneDOutputFileReader
from .SDKT_TXT_File import TXTFileReader
from .GeoFile import GeoFileReader


__all__ = [
	"BaseFileReader",
	"BinaryTerrainFileReader",
	"CMDFileReader",
	"CSVReader",
	"ASCIIFileReader",
	"TIFFileReader",
	"OneDOutputFileReader",
	"TXTFileReader",
	"GeoFileReader",
]
