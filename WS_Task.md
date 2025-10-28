# Snøskredkonferanse 2025: AvaFrame Workshop

## Download the materials for the course

Go to the following repositry

```url
https://github.com/norwegian-geotechnical-institute/Avaframe-Workshop
```

__download zip__

Download (or clone the repo) to your machine by navigating to the the above url, then clicking the green <> Code button where you should see a Download ZIP. Move the Zip to an appropritate location on your computer and expand to get the materials. 

__clone__

If you have git setup on your machine you can clone the repository using the command

```bash
git clone https://github.com/norwegian-geotechnical-institute/Avaframe-Workshop.git
```

This will download the repository and maintain the version control so you may recieve updates without needing re-download the full repository. 

## Installation

First step is to check that we have QGIS, GDAL and the avaframe/rasterops plugins installed. Please follow the [README.md](README.md) instructions for installation for your system. 

--- 

## Running your first simulation

Add the DEM raster and release_area polygon to the staged area in QGIS. 

Open the attribute table for the RA polygon and check it has field 'd0' or 'thickness'

Open the avaframe processing tool

 - Select the DEM
 - Select the RA
 - Click Run

Results (Pressure, Thickness and Speed) rasters will appear in the staged area.

---

## Advanced routines

### Ensure Preprocessing Tools Installed

For this we need to use the pre-processing tools to generate rasters for different simulations

Plugins -> Manage & install plugins -> Install via zip -> select the rasterops zip in the workshop folder -> Install

The 'RasterOps' plugin tool should appear in toolbar


### Advanced routines: Variable friction

#### Generating const. custom rasters for friction over the AoI

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


### Advanced routines: Forest

Fetch data for forest for your area (e.g. SR16) - normally this comes as tree count and breast height diameter (bhd)

Resample to the DEM resolution (e.g. 5 or 10m) using the 'resample' and 'match' tools.
Combine rasters using the treecount2nD recipe

We can then add the 'nD' and 'bhd' rasters to avaframe menu and run.

__Alternative method__

Draw a polygon where a forest exists then use the 'rasterise', 'full-like' and 'treecount2nD' tools to generate synthetic forest data on the avalanche path.  


## Other queries for avalanche simulations within the QGIS framework?  

We are happy to answer questions / requests for functionality. Let us know your thoughts.


---

## Super advanced - full control over simulations

### AvaFrame

To have full control of the simulations we can run them from a scripting environment. 

Install avaframe in advanced editable mode.
Using pixi set the correct environment
Copy and adjust the config `local_<name>.ini` files 
Run the simulations directly with a python call 
Inspect the results

### MoT Codes

Download the MoT Code (MoT-Voellmy or MoT-PSA) directly from the NGI repo. 
Adjust the config file stating the paths to the rasters you want to use

```bash
./MoT-Voellmy myconfig.rcf 
```

Results will be output into the requested directory which can then be pulled back into GIS software directly, as required, or used in the further post-processing.

