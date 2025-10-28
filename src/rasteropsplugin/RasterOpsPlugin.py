#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create a QGIS plugin for RasterOps entry point
"""

from qgis.PyQt.QtCore import QCoreApplication
from qgis.PyQt.QtWidgets import QAction, QMessageBox
from qgis.utils import iface
from qgis.core import QgsProject

from .RasterOpsDialog import RasterOpsDialog


class RasterOpsPlugin:
	def __init__(self, iface):
		self.iface = iface
		self.action = None
		self.dlg = None

	def initGui(self):
		self.action = QAction("RasterOps", self.iface.mainWindow())
		self.action.triggered.connect(self.run)
		self.iface.addToolBarIcon(self.action)
		self.iface.addPluginToMenu("&RasterOps Plugin", self.action)

	def unload(self):
		self.iface.removeToolBarIcon(self.action)
		self.iface.removePluginMenu("&RasterOps Plugin", self.action)

	def run(self):
		self.dlg = RasterOpsDialog()
		self.dlg.exec_()
