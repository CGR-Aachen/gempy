
from matplotlib.cm import ScalarMappable as SM
from gempy.plot._visualization_2d import PlotData2D
import numpy as np
import os


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

def export_pflotran_input(geo_model, path=None, filename='pflotran.ugi',
                          mesh_format=None):
    import h5py
    """
    Method to export a 3D geological model as PFLOTRAN implicit unstructured grid
    (see pflotran.org)

    Args:
        path (str): Filepath for the exported input file
        filename (str): name of exported input file
        mesh_format (str): format of exported mesh ('ascii' or 'hdf5')
                           if not provided, deduced from file extension

    Returns:
        
    """
    #
    # Added by Moise Rousseau, December 8th, 2020
    # see https://www.pflotran.org/documentation/user_guide/cards/subsurface/grid_card.html
    #
    # create vertices and elements
    vertices, elements = __build_vertices_and_elements__(geo_model)
    
    #create mesh
    if not path:
        path = './'
    if not os.path.exists(path):
        os.makedirs(path)
    
    if mesh_format is None:
        ext = filename.split('.')[-1]
        if ext == 'ugi': mesh_format = 'ascii'
        elif ext == 'h5': mesh_format = 'hdf5'
        else: #assume ugi mesh
            mesh_format = 'ugi'
            filename += '.ugi'
    
    if mesh_format == 'ascii': #export as ugi
        out = open(path+filename, 'w')
        out.write(f"{n_elements} {n_vertices}")
        for element in elements:
            out.write('\nH')
            for x in element[1:]:
              out.write(f" {int(x)}")
        for vertice in vertices:
            out.write('\n')
            out.write(f"{vertice[0]} {vertice[1]} {vertice[2]}")
        out.close()
    elif mesh_format == 'hdf5':
        out = h5py.File(path+filename, 'w')
        out.create_dataset("Domain/Cells", data=elements)
        out.create_dataset("Domain/Vertices", data=vertices)
        out.close()
    
    #make groups
    lith_ids = np.round(geo_model.solutions.lith_block)
    lith_ids = lith_ids.astype(int)
    sids = dict(zip(geo_model._surfaces.df['surface'], geo_model._surfaces.df['id']))
    for region_name,region_id in sids.items():
        cell_ids = np.where(lith_ids == region_id)[0] + 1
        if not len(cell_ids): continue
        if mesh_format == 'ascii':
            out = open(path+region_name+'.vs','w')
            for x in cell_ids:
                out.write(f"{x}\n")
            out.close()
        if mesh_format == "hdf5":
            out = h5py.File(path+filename,'r+')
            out.create_dataset(f"Regions/{region_name}/Cell Ids", data=cell_ids)
            out.close()
            
    print("Successfully exported geological model as PFLOTRAN input to "+path)
    return

def export_flac3D_input(geo_model, path=None, filename='geomodel.f3grid'):
    """
    Method to export a 3D geological model FLAC3D readable format

    Args:
        path (str): Filepath for the exported input file
        filename (str): name of exported input file

    Returns:
        
    """
    #
    # Added by Moise Rousseau, December 9th, 202
    #
    # create vertices and elements
    vertices, elements = __build_vertices_and_elements__(geo_model)
    
    #open output file
    if not path:
        path = './'
    if not os.path.exists(path):
        os.makedirs(path)
    out = open(path+filename, 'w')
    
    #write gridpoints
    out.write("*GRIDPOINTS")
    for i,vertice in enumerate(vertices):
        out.write(f"\nG {i+1} {vertice[0]} {vertice[1]} {vertice[2]}")
    
    #write elements
    out.write('\n*ZONES')
    for i,elem in enumerate(elements):
        out.write('\nB8')
        for x in elem[1:]:
            out.write(f" {x}")
    
    #make groups
    out.write('\n*GROUPS\n')
    lith_ids = np.round(geo_model.solutions.lith_block)
    lith_ids = lith_ids.astype(int)
    sids = dict(zip(geo_model._surfaces.df['surface'], geo_model._surfaces.df['id']))
    for region_name,region_id in sids.items():
        cell_ids = np.where(lith_ids == region_id)[0] + 1
        if not len(cell_ids): continue
        out.write(f"*ZGROUP {region_name}\n")
        count = 0
        for x in cell_ids:
            out.write(f"{x} ")
            count += 1
            if count == 8:
                out.write("\n")
                count = 0
        if count != 0: out.write("\n")
    
    out.close()
    print("Successfully exported geological model as FLAC3D input to "+path)
    return

def __build_vertices_and_elements__(geo_model):
    # get model information
    nx, ny, nz = geo_model.grid.regular_grid.resolution
    xmin, xmax, ymin, ymax, zmin, zmax = geo_model.solutions.grid.regular_grid.extent
    
    # create vertices array
    dx, dy, dz = (xmax-xmin)/nx, (ymax-ymin)/ny, (zmax-zmin)/nz
    n_vertices = (nx+1) * (ny+1) * (nz+1)
    vertices = np.zeros((n_vertices, 3), dtype='f8')
    vertices_ids = np.arange(n_vertices) #used to generate coordinate
    vertices[:,0] = vertices_ids % (nx+1) * dx
    vertices[:,1] = ( vertices_ids % ( (nx+1)*(ny+1) ) ) // (nx+1) * dy
    vertices[:,2] = vertices_ids // ( (nx+1)*(ny+1) ) * dz
    
    #build elements
    n_elements = nx*ny*nz
    element_ids = np.arange(n_elements) #used to generate elems
    elements = np.zeros((n_elements,9), dtype='i8')
    i = element_ids % nz
    j = element_ids // nz % ny
    k = element_ids // (nz*ny)
    elements[:,0] = 8 #all hex
    elements[:,1] = 1 + i*(nx+1)*(ny+1) + j*(nx+1) + k
    elements[:,2] = elements[:,1] + 1
    elements[:,3] = elements[:,2] + (nx+1)
    elements[:,4] = elements[:,3] - 1
    elements[:,5] = elements[:,1] + ( (nx+1)*(ny+1) )
    elements[:,6] = elements[:,5] + 1
    elements[:,7] = elements[:,6] + (nx+1)
    elements[:,8] = elements[:,7] - 1
    
    return vertices, elements
