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

([See here](https://gis.stackexchange.com/questions/307850/osgeo4w-checking-gdal-version-with-gdalinfo-version-returns-nothing) for additional info and
images if the above is not clear.)

__Linux__

To check which version of gdal is on the system (and that the
development tools are in place) use

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

We then install gdal and QGIS with

```bash
brew install gdal qgis
```

Check the version with

```bash
gdal-config --version
```

---

## Opening QGIS with additional python packages

For the plugin to work we need to open QGIS with additional packages installed. Firstly we need to tell python which version of GDAL is installed on your system. To do this edit the line in `requirements.txt` so it matches the version of GDAL installed i.e. if your GDAL is 3.11.3 then change the line

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
python -m pip install -r requirements.txt
```

Now the python used by QGIS can use the installed packages. We can now open QGIS with the command

```powershell
qgis
```

### Linux and Mac

First, create an __isolated virtual environment__ for the project so that all required dependencies are installed locally and don’t interfere with your system Python.

```bash
# Create a new virtual environment (named .venv)
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

Now we can run the script `qgis-poetry.sh` which will expose the site-packages to QGIS so QGIS's python can use them

```bash
chmod +x qgis-poetry.sh
./qgis-poetry.sh
```

This will open QGIS with the specific dependencies for this project available so the plugin can use them.

## Installing the Plugin

Open QGIS using the method descibed above. Install the pluging via

```text
QGIS top menu -> Plugins -> Manage and Install Plugins -> Install from ZIP
```

Drag and drop the ZIP file `RasterOpsPlugin.zip` into the path and click install.

Now the plugin will be able to run the tools using additional dependencies to QGIS defaults.

## Using the Plugin

Load any layers you wish to manipulate into QGIS (drag and drop files into the stage area).

Click the `RasterOps` button in the top menu.

- Select an operation.
- Select relevant layers
- Hit Run

Your processed raster should appear in the staging area.
