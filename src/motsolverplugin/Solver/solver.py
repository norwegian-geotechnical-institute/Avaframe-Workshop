#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entry point to access a wrapper for the solvers in the framework and handle some directory management. Options are provided for manual command configuration specification and adjustment.
"""

import sys
import argparse, textwrap, shutil
import subprocess as sp
from subprocess import CalledProcessError, TimeoutExpired
import traceback
from pathlib import Path
from rich.pretty import pprint
from importlib.metadata import version
import tempfile

from ..FileIO.FileReaders import CMDFileReader
from ..FileIO.FileWriters import CMDFileWriter
from .Defaults import DefaultCMDParams, Get_filetype_options

_BASE_DIR = Path(__file__).resolve().parent


class Solve:

	SOLVERS = {
		"MoTV-linux": _BASE_DIR /  "solvers" / "MoT-Voellmy",
		"MoTV-static": _BASE_DIR /  "solvers" / "MoT-Voellmy-static",
		"MoTV-windows": _BASE_DIR / "solvers" / "MoT-Voellmy-win64.exe",
		"MoTV-mac": _BASE_DIR / "solvers" / "MoT-Voellmy-macOS",
		"MoTPSA-linux": _BASE_DIR /  "solvers" / "MoT-PSA",
		"MoTPSA-windows": _BASE_DIR / "solvers" / "MoT-PSA-win64.exe",
		"MoTPSA-mac": _BASE_DIR / "solvers" / "MoT-PSA-macOS",
		"MoTmuI-linux": _BASE_DIR /  "solvers" / "MoT-muI",
		"MoTmuI-windows": _BASE_DIR / "solvers" / "MoT-muI-win64.exe",
		"MoTmuI-mac": _BASE_DIR / "solvers" / "MoT-muI-macOS",
	}

	def __init__(self, solver="MoTV", os="linux", verbose=False):
	
		self.solvername = solver
		self.solver = self.SOLVERS.get(f"{solver}-{os}")
		if not self.solver or not self.solver.is_file():
			raise RuntimeError(f"{solver}-{os} is not available")
	
		self.reader = CMDFileReader()
		self.writer = CMDFileWriter()
		self.log = []
		self.verbose=verbose
		

	def __call__(self, overrides={}, timeout=None):
		
		# Create temporary config file
		tmpfile = tempfile.NamedTemporaryFile(suffix=".rcf", delete=False)
		tmp_path = tmpfile.name
		tmpfile.close()
		
		# Fetch the cmdparams and update with any overrides e.g. paths to specific rasters etc. 
		cmdfile_dat = DefaultCMDParams.copy()
		cmdfile_dat.update(overrides)
		cmdfile_dat["filename"] = tmp_path
		
		outdir = Path(cmdfile_dat["outroot"]).parent
		outidir = outdir / "Input"

		self.writer(tmpfile, self.solvername, cmdfile_dat)
		
		# Construct the result directories
		try:
			outdir.mkdir(parents=True, exist_ok=True)
			outidir.mkdir(parents=True, exist_ok=True)
		except OSError:
			raise RuntimeError(f"Failed to create output directory {outdir}. Check permissions.")

		# Run solver and cleanup
		sres = self._solve(solvername, solver_path, cmdfile_dat.get("filename"), timeout=timeout)
		print(f"Results written to {outdir}")

		# Indicate how it went
		if sres:
			print("Simulation completed successfully!")
		else:
			print("Simulation failed...")
			
		return outdir


	def _solve(self, solvername, solver_path, cmdfile, timeout=None):
		"""Run solver as a subprocess - indicate success"""
		
		try:
			out = sp.run(
				[str(solver_path), str(cmdfile)],
				cwd=str(solver_path.parent),
				stdin=sp.DEVNULL,
				stdout=sp.PIPE,
				stderr=sp.PIPE,
				timeout=timeout,
			)
			if self.verbose:
				print(out.stdout.decode("utf-8", errors="replace"))
			out.check_returncode()
			
		except CalledProcessError as e:
			if e.returncode == 1:
				msg = f"{solvername}: Internal Timeout Exit (code 1)"
				self.log.append(msg)
				if self.verbose:
					print(msg)
				return True  # still considered a "normal" stop
			else:
				msg = f"{solvername}: Solver process failed. Return code {e.returncode}. {e}"
				self.log.append(msg)
				if self.verbose:
					print(msg)
				return False
				
		except TimeoutExpired:
			msg = f"{solvername}: External Timeout Exit"
			self.log.append(msg)
			if self.verbose:
				print(msg)
			return False
			
		except UnicodeDecodeError:
			msg = f"{solvername}: Output contained undecodable bytes"
			self.log.append(msg)
			if self.verbose:
				print(msg)
				
		except Exception:
			msg = f"{solvername}: Unexpected error in solver run\n{traceback.format_exc()}"
			self.log.append(msg)
			if self.verbose:
				print(msg)
			return False

		msg = f"{solvername}: Normal Exit"
		self.log.append(msg)
		if self.verbose:
			print(msg)
		return True

