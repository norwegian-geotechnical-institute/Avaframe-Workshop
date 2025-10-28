# QGIS Plugin Installation

## Install QGIS

__Windows__  

We recommend installing Python, GDAL and QGIS from
[OSGeo4W](https://trac.osgeo.org/osgeo4w/). If you have OSGeo4W
and GDAL installed but just need the extra Python tools you can rerun the following method.

Run `osgeo4w-setup.exe`, located by default in `C:\OSGeo4W\bin`.

- Run the Installer application
- Select _Advanced Setup_ and continue.
- Select _Install from Internet_ and continue.
- Select your existing _Root Directory_ and preferences until you
  reach a screen showing lots of packages that can be installed.
- Search for _gdal_, toggle to Full View. The first column will
  show you your installed version – __note down the number__ – it
  should look like e.g. "gdal 3.11.3" (Note the version number on
  your system is likely to be different). We will need the version
  number of your GDAL installation in a later step.
- (If you have GDAL already installed the version number can be checked with the command `gdalinfo --version` from the OSGEO4W shell)
- Enter the Advanced section.
- Choose `python3-gdal` and `python3-gdal-dev` from the advanced section and swap
  from "skip" to "keep".
- Likewise from the packages section choose `qgis` from "skip" to "keep".
- Check whether `pip` is installed; if not, mark it for installation.
- Complete the installation.  

([See here](https://gis.stackexchange.com/questions/307850/osgeo4w-checking-gdal-version-with-gdalinfo-version-returns-nothing) or `Install/QGIS_Avaframe_MoT.docx` for additional info and images if the above is not clear.)


__Linux__

To check which version of gdal is on the system (and that the development tools are in place) use

```bash
sudo apt install gdal-bin libgdal-dev python3-gdal qgis
```

we can check the version of gdal with

```bash
gdal-config --version
```

__macOS__

GDAL is available via [homebrew](https://brew.sh/).

To install `homebrew`, open _Terminal_ and use the following command

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

We then install GDAL and QGIS with

```bash
brew install gdal qgis
```

Check the version with

```bash
gdal-config --version
```

---

## Install Dependencies

For Avaframe to work we need to open QGIS with additional packages installed so the python inside QGIS can use them. 

### requirements.txt

Firstly we need to tell python which version of GDAL is installed on your system. To do this edit the line in `requirements.txt` so it matches the version of GDAL installed i.e. if your GDAL is 3.11.3 then change the line

```text
gdal==3.8.4 ; python_version >= "3.11" and python_version < "3.13"
```

to

```text
gdal==3.11.3 ; python_version >= "3.11" and python_version < "3.13"
```

### Windows

Open the OSGEO4W shell. Run the following line

```powershell
python -m pip install -r <path to project directory>\requirements.txt
```

This will install the dependencies for both Avaframe and the preprocessing tools. Now the python used by QGIS can access the installed packages.

Next we can now open QGIS with the command

```powershell
qgis
```

### Linux and Mac

QGIS uses different python versions depending on whether you have opened it via the desktop or from an environment (e.g. from the terminal). To get around this and ensure that the packages are available for whichever python QGIS wants to use we need to expose the packages to our PATH variable.  

If you know for sure which version of python QGIS is using on your system (you can check this through the python terminal within QGIS: `import sys; print(sys.executable)`). We can then install requirements directly to this python e.g. `/usr/bin/python3 -m pip install --force -r requirements.txt`. However, this is not recommended as it can break your system python and other packages may break due to the requirements for this package. 

The workaround for this is to create an __isolated virtual environment__ for the project so that all required dependencies are installed locally and don’t interfere with your system Python. This happens in the OSGEO4W shell already in the Windows installation as the python QGIS uses is isolated there. For those on Linux and Mac we need to expose the packages to whichever python QGIS wants to use.


__poetry__

To install poetry we do (either using apt or brew)

```bash
sudo apt install pipx
pipx install poetry
```

Once poetry is installed you can simply type

```bash
poetry install --no-root
```

and all dependencies will be fetched into an environment for you.


__Python std venv__

If poetry is not available for you then we can do the following.
Create a new virtual environment (named .venv)

```bash
python -m venv .venv
```

Activate the environment:

```bash
source .venv/bin/activate
```

Once activated, install all dependencies listed in `requirements.txt` into the new environment:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Finally, confirm that the installation worked by listing the installed packages:

```bash
python -m pip list
```


#### Open QGIS with environment 


Now we can run the script `qgis-poetry.sh` which will expose the site-packages in the virtual environment so any python QGIS is using can see them.

```bash
chmod +x qgis-poetry.sh
./qgis-poetry.sh
```

This will open QGIS with the specific dependencies for Avaframe and the preprocessing tools exposed so the plugin can use them.

For ease of future use you can make an alias to this script which will make a global command to open QGIS from the terminal. Edit the following line to where you have installed the project. If using zsh on mac swap to .zshrc as appropriate. 

```bash
echo "alias avaframeQGIS='<path to script>/qgis-poetry.sh'" >> ~/.bashrc
```

The command `avaframeQGIS` will then open QGIS with the avaframe requirements whenever you need it.


___

## Installing the Avaframe and RasterOps Plugins

Open QGIS using the method described above that works for you. Install the plugin via

```text
QGIS top menu -> Plugins -> Manage and Install Plugins -> Install from ZIP
```

Drag and drop the ZIP file `Install/RasterOpsPlugin.zip` into the path and click install.


For Avaframe search for `Avaframe` from the install from Web tab. Click install.

Now both plugin will be installed and able to run.



## Using the Plugins


### Avaframe

Load any layers you wish to manipulate into QGIS (drag and drop files into the staging area).

Open the processing toolbox. Select the MoT Voellmy tool

```text
Avaframe - NGI_Experimental - MoTVoellmy (Com9)
```

In the popup menu select the options you wish and add relevant rasters then click run.
Results will appear in the staging area.


### RasterOps

Load any layers you wish to manipulate into QGIS (drag and drop files into the staging area).

Click the `RasterOps` button in the top menu.

- Select an operation.
- Select relevant layers
- Hit Run

Your processed raster results will appear in the staging area.


#### Recipes

The tool is designed such that general operations are available at top level but specific "recipes" (combinations of operations that are likely more frequently used or more complex operations than simple raster combinations e.g. calculating variable friction parameters for the terrain or combining forest data files into the correct format and scale) are stored under the recipes subheading. This allows us to write specific more complex routines which can take in - process and return rasters. 