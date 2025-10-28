from .RasterOpsPlugin import RasterOpsPlugin


def classFactory(iface):
	return RasterOpsPlugin(iface)
