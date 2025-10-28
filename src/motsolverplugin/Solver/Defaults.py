#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Include default file types parameters
"""

import os
import numpy as np
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _BASE_DIR.parent.parent


# large ava domain: 540, 616, 30235.0, 6707145.0, 10.0, -9999

# Default ESRI_ASCII_Grid File Parameters
DefaultASCIIParams = {
	"ncols": 30,  # Number of columns
	"nrows": 100,  # Number of rows
	"xllcorner": 0.0,  # x in lower left corner
	"yllcorner": 0.0,  # y in lower left corner
	"cellsize": 5,  # square cell edge length
	"nodata": -9999,  # nodata value
}


# Default Command File Parameters for all cmdfile types
DefaultCMDParams = {
	"file_version": "2024-09-10",  # MoT codes (v.) particular about cmdfile format
	"area": "Testland",  # Specify area
	"utm": "33N",  # UTM code
	"epsg": 25833,  # EPSG code
	"name": "Avalanche",  # Run name
	#
	"dempath": str(
		_ROOT_DIR / "data" / "input" / "Test" / "dem_5m_sn1.asc"
	),  # choose filename or - for IC dem mask
	"hpath": str(
		_ROOT_DIR / "data" / "input" / "Test" / "rel_sn1.asc"
	),  # choose filename or - for IC h mask
	"bpath": "-",  # choose filename or - for IC bed depth mask
	"taucpath": "-",  # choose filename or - for IC tauc = bed shear strength mask
	"forestnDpath": "-",  # choose filename or - for IC forest density mask
	"foresttDpath": "-",  # choose filename or - for IC tree diameter mask
	"upath": "-",  # choose filename or - for IC u	mask
	"vpath": "-",  # choose filename or - for IC v mask
	#
	"outroot": str(_ROOT_DIR / "data" / "results" / "Test") + "/ava",  # root results directory
	"outfmt": "ESRI_ASCII_Grid",  # results format choice
	"fmt": "w",  # if text write or binary write
	#
	"g": 9.81,  # gravity
	"rho": 250.0,  # const. density of snow for pressure calc.
	"rheo": "Voellmy",  # friction law specification
	"params": "variable",  # if filenames use "variable" else "constant"
	#
	"mu": 0.3,  # choose either filename or const mu val
	"k": 0.001,  # choose either filename or const k val
	"zeta1": 21.4,  # choose either filename or const zeta1 val
	"zeta2": 34.7,  # choose either filename or const zeta2 val
	"zeta3": 40.0,  # choose either filename or const zeta3 val
	"beta": 0.9,  # choose either filename or const beta val
	"betastar": 0.9,  # choose either filename or const betastar val
	"L": 0.31,  # choose either filename or const L val
	"kappa": 1.0,  # choose either filename or const kappa val
	"Gamma": 0.0,  # choose either filename or const Gamma val
	#
	"effdragh": 0.0,  # effective drag height
	"earthp": 1.0,  # earth pressure k
	"centrifugal": "no",  # switch additional centrifugal forces from curvature
	#
	"granularviscosity": "no",  # switch granular viscosity
	#
	"deposition": "no",  # switch deposition effects
	"evo_geo": "no",  # swith evolving geometry
	#
	"foresteff": "no",  # switch forest effects
	"treedrag": 1.0,  # tree drag val if forest effects
	"modrupture": 50.0,  # modulus of rupture if forest effects
	"forestdecay": 0.15,  # forest decay rate if forest effects
	#
	"entrainment": "none",  # switch entrainment model (TJEM, RAMMS, JoIs)
	"erocoeff": 0.0,  # erosion rate
	"bedstrength": "global",  # bed strength (const, global, local)
	"bedfriction": 0.25,  # bed friction val (const: 0), (global,val), (local, filename)
	"bedrho": 140.0,  # bed density
	"deprho": 400.0,  # deposit density
	#
	"t_end": 200,  # simultion end
	"mindt": 1e-12,  # minimum time step threshold
	"maxdt": 0.1,  # maximum time step threshold
	"t_out": 1.0,  # output interval
	#
	"writevel": "no",  # write velocity output - (u,v) fields
	"writemaxp": "yes",  # write max pressure field at end
	"writeinstp": "no",  # write inst pressure field
	#
	"hthreshold": 0.01,  # thickness threshold below which 0
	"uthreshold": 0.01,  # vel threshold below which 0
	"momthreshold": 0.05,  # mom threshold to dictate early sim end
	"cfl": 0.25,  # CFL number
	#
	# Extras for MoT PSA
	#
	"h1path": "-",  # IC thickness of dense part
	"h2path": "-",  # IC thickness of powder part
	"bdeppath": "-",  # existing deposit mask
	"bdrag_01": 0.008,  # basal drag coeff 0 - 1
	"bdrag_12": 0.04,  # basal drag coeff 1 - 2
	"topdrag": 0.0001,  # top drag coeff
	#
	"phi_L2_0": 1.3333,  # phi powder order 0
	"phi_L2_1": -0.66667,  # phi powder order 1
	"phi_L2_2": 0.0,  # phi powder order 2
	"speed_L2_0": 1.4,  # speed powder order 0
	"speed_L2_1": 0.13333,  # speed powder order 1
	"speed_L2_2": -1.4,  # speed powder order 2
	#
	"entrainment_L1": "none",  # entraiment model e.g. TJEM
	"entrainment_L2": "Base+Air",  # suspension entraiment modes
	"erocoeff_L1": 0.0,  # erosion rate
	"m12": 0.0,  # m from dense to powder ?
	"decaycoeffsus": 1.0,  # decay coeff sus
	"decaycoeffdep": 1.0,  # decay coeff dep
	"deprate_21": 0.50,  # deposition rate sus to dense
	#
	"constdensity_L1": "yes",  # constant density layer 1
	#
	"suss_model": "TJSM",  # suspension model
	"ava_shear_strength": 0.0,  # avalanche shear strength
	"ava_shear_strength_dep": 0.0,  # avalanche shear strength deposit
}


def Get_default_ASCII_params():
	return {**DefaultASCIIParams}


def Get_default_cmd_params():
	return {**DefaultCMDParams}


# Return all filetype options available to generate
# Split as friction params are either a filetype or a numeral
def Get_filetype_options(restrict=None):
	"""Fetch filetype options"""

	ifiletypes = {"dem", "h", "h1", "h2", "u", "v"}
	efiletypes = {"b", "tauc", "bedfriction"}
	Ffiletypes = {"forestnD", "foresttD"}
	ffiletypes = {
		"mu",
		"k",
		"zeta1",
		"zeta2",
		"zeta3",
		"beta",
		"betastar",
		"L",
		"kappa",
		"Gamma",
	}
	cmdfiletypes = {"MoTV", "MoTPSA", "MoTmuI"}

	if restrict == "fric":
		return ffiletypes
	elif restrict == "input":
		return ifiletypes
	elif restrict == "cmd":
		return cmdfiletypes
	elif restrict == "erosion":
		return efiletypes
	elif restrict == "forest":
		return Ffiletypes
	else:
		return cmdfiletypes | ifiletypes | ffiletypes | efiletypes | Ffiletypes


def Get_path_map():
	"""Generate a map between the filetype label and the label in cmdfileparams"""
	out = {}
	for ft in Get_filetype_options():
		if ft in Get_filetype_options(restrict="fric") | Get_filetype_options(restrict="cmd"):
			out.setdefault(ft, ft)
		else:
			out.setdefault(ft, ft + "path")
	return out
