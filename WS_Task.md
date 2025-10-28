# Snøskredkonferanse AvaFrame Workshop

## Download the materials for the course

Go to the following repositry

```text
http://www.github.com:NGI AvaFrameWS... 
```

Download (or clone the repo) to your machine. 


## Installation

First step is to check that we have QGIS installed and the appropriate dependencies are available for AvaFrame (and any extra tools)

### Windows

OSGEO4W installation - advanced - GDAL and QGIS

Open the OSGEO4W_Shell.exe and install the dependencies for QGIS

```powershell
pip install -r requirements.txt
```

as the shell couples the python version to QGIS.

We can then open QGIS using the command

```powershell
qgis
```  

Then we can install the plugin for AvaFrame through

```text
Plugins -> Manage and install plugins -> search for Avaframe -> install
```

Avaframe should then be available in the processing toolbox. 


### Linux & MacOS

Install GDAL and QGIS via apt (/ homebrew). 

QGIS by default uses system python if opened from desktop (your shell python if opened from a terminal). Due to this we need to make sure the python version being used is the correct one and expose the dependencies accordingly. 

Install pipx (pip install for different python systems). Use apt / brew as appropriate.

```bash
apt install pipx
``` 

Then we can install poetry (python package manager)

```bash
pipx install poetry
```

We can then install the python project locally

```bash
poetry install
```

The script qgis-poetry.sh then opens QGIS linking the correct python dependencies without affecting your system python. 

```bash
chmod +x qgis-poetry.sh
./qgis-poetry.sh
```

For ease of future use you can make an alias to this script and open QGIS (if using zsh on mac swap to .zshrc as appropriate) 

```bash
echo "alias avaframeQGIS='<path to script>/qgis-poetry.sh'" >> ~/.bashrc
```

the command `avaframeQGIS` will then open QGIS appropriately whenever you need it.


## Running your first simulation

Add the DEM raster and release_area polygon to the staged area in QGIS. 

Open the attribute table for the RA polygon and check it has field 'd0' or 'thickness'

Open the avaframe processing tool

 - Select the DEM
 - Select the RA
 - Click Run

Results (Pressure, Thickness and Speed) rasters will appear in the staged area.


## Advanced routines

### Preprocessing tools

For this we need to use the pre-processing tools to generate rasters for different simulations

Plugins -> Manage & install plugins -> Install via zip -> select the rasterops zip in the workshop folder -> install

The 'RasterOps' plugin tool should appear in toolbar


### Advanced routines: Variable friction

#### Generating const. custom rasters for friction over the AoI

First check all the CRS for both the rasters and QGIS display is consistent.

 - Select the full_like tool -> select dem -> select value for mu -> run
 - Rename the generated raster
 - Repeat for k

Open the avaframe tool - select your DEM, RA and friction rasters and run.

#### Generating variable custom rasters for frction over the AoI

First check all the CRS for both the rasters and QGIS display is consistent.

First rasterise the release area

 - Select the rasterise tool -> select DEM template -> choose band (1) -> choose value ('d0')
 - Rename the generated raster

Generate the NAKSIN mu and k

 - Select the recipe tool -> select NAKSIN_mu -> provide DEM & RA rasters -> run
 - Rename the generated raster
 - Repeat for k

Open the avaframe tool - select your DEM, RA and friction rasters and run.


### Advanced routines: Erosion 

For erosion we need more aditional information namely a erodible layer (b0) and a shear strength layer (tau c) 

First check all the CRS for both the rasters and QGIS display is consistent.

 - Select the full_like tool -> select dem -> select value for b0 -> run
 - Rename the generated raster
 - Repeat for tau c (300 is normal)

Open the avaframe tool - select your DEM, RA and erosion rasters and run.


### Advanced routines: Forest

Get data for forest for your area (e.g. SR16) - normally this comes as tree count and breast height diameter (bhd)

Resample to the DEM resolution (e.g. 5 or 10m) using the 'resample' and 'match' tools
Combine rasters using the treecount2nD recipe

We can then add the nD and bhd rasters to avaframe menu and run.


## Other queries for avalanche simulations within the QGIS framework?  

We are happy to answer questions


---

## Super advanced - full control over simulations

### Avaframe

To have full control of the simulations we can run them from a scripting environment. 

Install avaframe in advanced editable mode.
Using pixi set the correct environment
Copy and adjust the config `local_<name>.ini` files 
Run the simulations directly with a python call 
Inspect the results

### MoT Codes

Download the MoT Code (MoT-Voellmy or MoT-PSA) directly from the NGI repo. 
Adjust the config file stating the paths to the rasters you want to use

```bash
./MoT-Voellmy.2025... myconfig.rcf 
```

Results will be output into the requested directory which can then be pulled back into GIS software directly.

