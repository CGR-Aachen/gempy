
from matplotlib.cm import ScalarMappable as SM
from gempy.plot._visualization_2d import PlotData2D
import numpy as np
import os
import itertools as it


def export_geomap2geotiff(path, geo_model, geo_map=None, geotiff_filepath=None):
    """

    Args:
        path (str): Filepath for the exported geotiff, must end in .tif
        geo_map (np.ndarray): 2-D array containing the geological map
        cmap (matplotlib colormap): The colormap to be used for the export
        geotiff_filepath (str): Filepath of the template geotiff

    Returns:
        Saves the geological map as a geotiff to the given path.
    """
    from osgeo import gdal

    plot = PlotData2D(geo_model)
    cmap = plot._cmap
    norm = plot._norm

    if geo_map is None:
        geo_map = geo_model.solutions.geological_map[0].reshape(geo_model._grid.topography.resolution)

    if geotiff_filepath is None:
        # call the other function
        print('stupid')

    # **********************************************************************
    geo_map_rgb = SM(norm=norm, cmap=cmap).to_rgba(geo_map.T) # r,g,b,alpha
    # **********************************************************************
    # gdal.UseExceptions()
    ds = gdal.Open(geotiff_filepath)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    [cols, rows] = arr.shape

    outFileName = path
    driver = gdal.GetDriverByName("GTiff")
    options = ['PROFILE=GeoTiff', 'PHOTOMETRIC=RGB', 'COMPRESS=JPEG']
    outdata = driver.Create(outFileName, rows, cols, 3, gdal.GDT_Byte, options=options)

    outdata.SetGeoTransform(ds.GetGeoTransform())  # sets same geotransform as input
    outdata.SetProjection(ds.GetProjection())  # sets same projection as input
    outdata.GetRasterBand(1).WriteArray(geo_map_rgb[:, ::-1, 0].T * 256)
    outdata.GetRasterBand(2).WriteArray(geo_map_rgb[:, ::-1, 1].T * 256)
    outdata.GetRasterBand(3).WriteArray(geo_map_rgb[:, ::-1, 2].T * 256)
    outdata.GetRasterBand(1).SetColorInterpretation(gdal.GCI_RedBand)
    outdata.GetRasterBand(2).SetColorInterpretation(gdal.GCI_GreenBand)
    outdata.GetRasterBand(3).SetColorInterpretation(gdal.GCI_BlueBand)
    # outdata.GetRasterBand(4).SetColorInterpretation(gdal.GCI_AlphaBand)  # alpha band

    # outdata.GetRasterBand(1).SetNoDataValue(999)##if you want these values transparent
    outdata.FlushCache()  # saves to disk
    outdata = None  # closes file (important)
    band = None
    ds = None

    print("Successfully exported geological map to  " +path)

def export_moose_input(geo_model, path=None, filename='geo_model_units_moose_input.i'):
    """
    Method to export a 3D geological model as MOOSE compatible input. 

    Args:
        path (str): Filepath for the exported input file
        filename (str): name of exported input file

    Returns:
        
    """
    # get model dimensions
    nx, ny, nz = geo_model.grid.regular_grid.resolution
    xmin, xmax, ymin, ymax, zmin, zmax = geo_model.solutions.grid.regular_grid.extent
    
    # get unit IDs and restructure them
    ids = np.round(geo_model.solutions.lith_block)
    ids = ids.astype(int)
    
    liths = ids.reshape((nx, ny, nz))
    liths = liths.flatten('F')

    # create unit ID string for the fstring
    idstring = '\n  '.join(map(str, liths))

    # create a dictionary with unit names and corresponding unit IDs
    sids = dict(zip(geo_model._surfaces.df['surface'], geo_model._surfaces.df['id']))
    surfs = list(sids.keys())
    uids = list(sids.values())
    # create strings for fstring, so in MOOSE, units have a name instead of an ID
    surfs_string = ' '.join(surfs)
    ids_string = ' '.join(map(str, uids))
    
    fstring = f"""[MeshGenerators]
  [./gmg]
  type = GeneratedMeshGenerator
  dim = 3
  nx = {nx}
  ny = {ny}
  nz = {nz}
  xmin = {xmin}
  xmax = {xmax}
  yim = {ymin}
  ymax = {ymax}
  zmin = {zmin}
  zmax = {zmax}
  block_id = '{ids_string}'
  block_name = '{surfs_string}'
  [../]
  
  [./subdomains]
    type = ElementSubdomainIDGenerator
    input = gmg
    subdomain_ids = '{idstring}'
  [../]
[]

[Mesh]
  type = MeshGeneratorMesh
[]
"""
    if not path:
        path = './'
    if not os.path.exists(path):
      os.makedirs(path)

    f = open(path+filename, 'w+')
    
    f.write(fstring)
    f.close()
    
    print("Successfully exported geological model as moose input to "+path)

def export_shemat_suite_input_file(geo_model, path: str=None, filename: str='geo_model_SHEMAT_input'):
    """
    Method to export a 3D geological model as SHEMAT-Suite input-file for a conductive HT-simulation. 

    Args:
        path (str): Filepath for the exported input file (default './')
        filename (str): name of exported input file (default 'geo_model_SHEMAT_input')
    """
    # get model dimensions
    nx, ny, nz = geo_model.grid.regular_grid.resolution
    xmin, xmax, ymin, ymax, zmin, zmax = geo_model.solutions.grid.regular_grid.extent
    
    delx = (xmax - xmin)/nx
    dely = (ymax - ymin)/ny
    delz = (zmax - zmin)/nz
    
    # get unit IDs and restructure them
    ids = np.round(geo_model.solutions.lith_block)
    ids = ids.astype(int)
    
    liths = ids.reshape((nx, ny, nz))
    liths = liths.flatten('F')

    # group litho in space-saving way
    sequence = [len(list(group)) for key, group in it.groupby(liths)]
    unit_id = [key for key, group in it.groupby(liths)]
    combined = ["%s*%s" % (pair) for pair in zip(sequence,unit_id)]

    combined_string = " ".join(combined)

    # get number of units and set units string
    units = geo_model.surfaces.df[['surface', 'id']]
    
    unitstring = ""
    for index, rows in units.iterrows():
        unitstring += f"0.01d-10    1.d0  1.d0  1.e-14	 1.e-10  1.d0  1.d0  3.74	0.  2077074.  10  2e-3	!{rows['surface']} \n" 	

    # input file as f-string
    fstring = f"""!==========>>>>> INFO
# Title
{filename}

# linfo
1 2 1 1

# runmode
1

# timestep control
0
1           1           0           0

# tunit
1
 
# time periods, records=1
0      60000000    200      lin
           
# output times, records=10
1
6000000
12000000
18000000
24000000
30000000
36000000
42000000
48000000
54000000
    
# file output: hdf vtk

# active temp

# PROPS=bas

# USER=none


# grid
{nx} {ny} {nz}

# delx
{nx}*{delx}

# dely
{ny}*{dely}

# delz
{nz}*{delz}

!==========>>>>> NONLINEAR SOLVER
# nlsolve
50 0

!==========>>>>> FLOW
# lsolvef (linear solver control)
1.d-12 64 500
# nliterf (nonlinear iteration control)
1.0d-10 1.0

!==========>>>>> TEMPERATURE
# lsolvet (linear solver control)
1.d-12 64 500
# nlitert (nonlinear iteration control)
1.0d-10 1.0

!==========>>>>> INITIAL VALUES
# temp init
{nx*ny*nz}*15.0d0  

# head init
{nx*ny*nz}*7500

!==========>>>>> UNIT DESCRIPTION
!!
# units
{unitstring}

!==========>>>>>   define boundary properties
# temp bcd, simple=top, value=init

# temp bcn, simple=base, error=ignore
{nx*ny}*0.06

# uindex
{combined_string}"""

    if not path:
        path = './'
    if not os.path.exists(path):
        os.makedirs(path)

    f = open(path+filename, 'w+')
    
    f.write(fstring)
    f.close()
    
    print("Successfully exported geological model as SHEMAT-Suite input to "+path)        