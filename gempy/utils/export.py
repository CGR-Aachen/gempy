
from matplotlib.cm import ScalarMappable as SM
from gempy.plot._visualization_2d import PlotData2D
import numpy as np
import os
import rasterio as rio

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

    plot = PlotData2D(geo_model)
    cmap = plot._cmap
    norm = plot._norm

    if geo_map is None:
        geo_map = geo_model.solutions.geological_map[0].reshape(geo_model._grid.topography.resolution)

    geo_map_rgb = SM(norm=norm, cmap=cmap).to_rgba(geo_map.T) # r,g,b,alpha

    with rio.open(geotiff_filepath, 'r') as src:
        profile = src.profile
        profile.update(
            count=3,
            dtype=rio.uint8,
            photometric="RGB"
        )
        with rio.open(path, 'w', **profile) as dst:
            dst.write(geo_map_rgb[:, ::-1, 0].T * 256, 1)
            dst.write(geo_map_rgb[:, ::-1, 1].T * 256, 2)
            dst.write(geo_map_rgb[:, ::-1, 2].T * 256, 3)
            

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
