#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dialog with QGIS to the Plugin functionality
"""

from qgis.PyQt.QtWidgets import (
	QDialog,
	QVBoxLayout,
	QHBoxLayout,
	QLabel,
	QPushButton,
	QComboBox,
	QMessageBox,
	QWidget,
	QLineEdit,
)
from qgis.core import QgsProject, QgsRasterLayer, QgsVectorLayer
from qgis.utils import iface

import tempfile
from rich.pretty import pprint

from .FileIO.FileReaders import GeoFileReader
from .FileIO.FileWriters import TIFFileWriter
from .RasterOp.RasterMethods import RasterMethods
from .RasterOp.Recipes import Recipes


class RasterOpsDialog(QDialog):
	def __init__(self, parent=None):
		super().__init__(parent)
		self.setWindowTitle("RasterOps Tool")

		main_layout = QVBoxLayout()
		self.setLayout(main_layout)

		self.resize(700, 500)  # width x height
		self.center_on_qgis(parent)

		# Operation selection
		main_layout.addWidget(QLabel("Select Operation:"))
		self.op_combo = QComboBox()
		self.op_combo.addItems(RasterMethods._available())
		main_layout.addWidget(self.op_combo)

		# Dynamic options placeholder
		self.options_widget = QWidget()
		self.options_layout = QVBoxLayout()
		self.options_widget.setLayout(self.options_layout)
		main_layout.addWidget(self.options_widget)
		
		# Dynamic suboptions placeholder
		self.suboptions_widget = QWidget()
		self.suboptions_layout = QVBoxLayout()
		self.suboptions_widget.setLayout(self.suboptions_layout)
		main_layout.addWidget(self.suboptions_widget)

		# Run button
		btn_layout = QHBoxLayout()
		self.run_btn = QPushButton("Run")
		self.run_btn.clicked.connect(self.run_operation)
		btn_layout.addWidget(self.run_btn)
		main_layout.addLayout(btn_layout)

		# Connect operation change
		self.op_combo.currentTextChanged.connect(self.update_options)
		self.update_options(self.op_combo.currentText())

	def center_on_qgis(self, parent=None):
		"""Center this dialog over the QGIS main window."""
		if parent:
			window = iface.mainWindow()
			geom = window.geometry()
			x = geom.x() + geom.width() / 2 - self.width() / 2
			y = geom.y() + geom.height() / 2 - self.height() / 2
			self.move(int(x), int(y))

	def clear_options(self, layout=None):
	
		if layout is None:
			layouts = [self.options_layout, self.suboptions_layout]
		else:
			layouts = [layout]
	
		for layout in layouts:
			if layout is not None:
				while layout.count():
					item = layout.takeAt(0)
					w = item.widget()
					if w:
						w.deleteLater()
	
	def add_to_layout(self, layout, args, store):
		# Add the option selectors from metadata
		for arg_name, arg_info in args.items():
			label = QLabel(arg_info.get("label", arg_name))
			layout.addWidget(label)

			t = arg_info["type"]

			if t == "raster":
				combo = QComboBox()
				for lyr in QgsProject.instance().mapLayers().values():
					if isinstance(lyr, QgsRasterLayer):
						combo.addItem(lyr.name(), lyr)
				layout.addWidget(combo)
				store[arg_name] = combo

			elif t == "vector":
				combo = QComboBox()
				for lyr in QgsProject.instance().mapLayers().values():
					if isinstance(lyr, QgsVectorLayer):
						combo.addItem(lyr.name(), lyr)
				layout.addWidget(combo)
				store[arg_name] = combo

			elif t == "enum":
				combo = QComboBox()
				combo.addItems(arg_info.get("choices", []))
				layout.addWidget(combo)
				store[arg_name] = combo

			elif t in ("float", "int", "string"):
				line = QLineEdit()
				layout.addWidget(line)
				store[arg_name] = line
	
	def update_options(self, op_name):
		
		self.clear_options()
		self.input_widgets = {}

		meta = RasterMethods._meta.get(op_name, {})
		args = meta.get("args", {})

		# SPECIAL CASE: recipe operation
		if op_name == "recipe":
		
			self.recipe_widgets = {}
			
			# Stage 1: show recipe selector
			label = QLabel("Select Recipe:")
			self.options_layout.addWidget(label)
			self.recipe_combo = QComboBox()
			
			# Assume we can fetch available recipe names from Recipes
			self.recipe_combo.addItems(Recipes._available())
			self.options_layout.addWidget(self.recipe_combo)

			# Connect signal: when user selects a recipe, update stage 2
			self.recipe_combo.currentTextChanged.connect(self.update_recipe_options)
			
			return  # stop here so normal args aren't added yet

		self.add_to_layout(self.options_layout, args, self.input_widgets)


	def update_recipe_options(self, recipe_name):
		"""Populate recipe-specific metadata options."""
		
		self.clear_options(self.suboptions_layout)
		self.recipe_widgets = {}
		
		meta = Recipes._meta.get(recipe_name, {})
		args = meta.get("args", {})
		self.add_to_layout(self.suboptions_layout, args, self.recipe_widgets)


	def run_operation(self):
		"""Algorithm to connect with QGIS"""
		
		# Fetch the operation
		op_name = self.op_combo.currentText()
		op = getattr(RasterMethods(), op_name, None)
		if not op:
			QMessageBox.warning(self, "Error", f"No implementation for {op_name}")
			return

		# Create temporary file
		tmpfile = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
		tmp_path = tmpfile.name
		tmpfile.close()

		# Collect function arguments from selectors
		kwargs = {"output_file": tmp_path}
		
		def collect_widgets(widgets_dict, metaclass):
			for arg, widget in widgets_dict.items():
				if isinstance(widget, QComboBox):
					data = widget.currentData()
					if isinstance(data, (QgsRasterLayer, QgsVectorLayer)):
						fname = data.dataProvider().dataSourceUri()
						kwargs[arg] = fname
					else:
						kwargs[arg] = widget.currentText()
				elif isinstance(widget, QLineEdit):
					type_converter = {
						"float": lambda x: float(x),
						"int": lambda x: int(x),
						"string": lambda x: str(x),
					}
					text = widget.text()
					if text == '':
						kwargs[arg] = None
					else:
						argtype = metaclass._meta.get(op_name, {}).get("args", {}).get(arg, {}).get("type", "string")
						kwargs[arg] = type_converter.get(argtype, lambda x: str(x))(text)
						
						# HACK
						if arg == "T_ret":
							kwargs["T_ret"] = float(text)
							
		
		# Collect main op widgets
		collect_widgets(self.input_widgets, RasterMethods)
		
		# Collect recipe stage if applicable
		if op_name == "recipe" and hasattr(self, "recipe_widgets"):
			kwargs["recipe_name"] = self.recipe_combo.currentText()
			collect_widgets(self.recipe_widgets, Recipes)
			
			out_name = f"{op_name}_{kwargs['recipe_name']}_result"
		else:
			out_name = f"{op_name}_result"
			
		# Call operation with selected arguments
		print(f"Running {op_name} with {kwargs}")
		op(**kwargs)

		# Get the current map canvas CRS
		iface_crs = iface.mapCanvas().mapSettings().destinationCrs()

		# Create the layer
		out_layer = QgsRasterLayer(tmp_path, out_name)

		# If the layer loaded correctly
		if out_layer.isValid():
			# Compare CRS and assign if different
			if out_layer.crs() != iface_crs:
				out_layer.setCrs(iface_crs)
				
			QgsProject.instance().addMapLayer(out_layer)
		else:
			raise Exception(f"Failed to load output raster {tmp_path}")
