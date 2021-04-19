"""
This file is part of gempy.

Created on 16.04.2019

@author: Elisa Heim
"""
import logging
logger = logging.getLogger('gempy')
import numpy as np
from scipy import fftpack


import rasterio as rio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling

import matplotlib.pyplot as plt


class LoadDEMGDAL:
    """Class to include height elevation data (e.g. DEMs) with the geological grid """
    dem: rio.DatasetReader

    def __init__(self, path_dem, grid=None, extent=None):
        """
        Args:
            path_dem: path where dem is stored. file format: GDAL raster formats
            if grid: cropped to geomodel extent
        """
        self.dem = rio.open(path_dem)

        if isinstance(self.dem, type(None)):
            raise AttributeError('Raster file could not be opened {}. Check if the filepath is correct. If yes,'
                                 'check if your file fits the requirements of GDALs raster file formats.'.format(
                path_dem))

        try:
            self.dem_zval = self.dem.read(1)
        except AttributeError:
            raise AttributeError('Filepath seems to be wrong.')

        self._get_raster_dimensions()

        if extent is not None:
            self.regular_grid_extent = extent
            self.crop2grid()
        elif grid is not None:
            self.regular_grid_extent = grid.extent
            self.crop2grid()
        else:
            logger.info('pass geo_model to automatically crop the DEM to the grid extent')
        logger.info('depending on the size of the raster, this can take a while...')
        self.convert2xyz()

    def _get_raster_dimensions(self):
        """calculates DEM extent, resolution, and max. z extent (d_z)"""
        ulx, xres, xskew, uly, yskew, yres = self.dem.transform.to_gdal()
        z = self.dem_zval
        if np.any(np.array([xskew, yskew])) != 0:
            logger.info('DEM is not north-oriented.')
        lrx = ulx + (self.dem.res[0] * xres)
        lry = uly + (self.dem.res[0] * yres)
        self.resolution = np.array([(uly - lry) / (-yres), (lrx - ulx) / xres]).astype(int)
        self.extent = np.array([ulx, lrx, lry, uly]).astype(int)
        self.d_z = np.array([z.min(), z.max()])

    def get_values(self):
        return self.values_3D

    def info(self):
        ulx, xres, xskew, uly, yskew, yres = self.dem.transform.to_gdal()
        logger.info('raster extent:  {}\n raster resolution: {}\n Pixel X size {}, Pixel Y size {}'.format(
            self.extent, self.resolution, xres, yres))
        plt.imshow(self.dem_zval, extent=self.extent)  # plot raster as image
        plt.colorbar()

    def crop2grid(self):
        """
        Crops raster to extent of the geomodel grid.
        """
        cornerpoints_geo = self._get_cornerpoints(self.regular_grid_extent)
        cornerpoints_dtm = self._get_cornerpoints(self.extent)
        if np.any(cornerpoints_geo[:2] - cornerpoints_dtm[:2]) != 0:
            new_bounds = self.regular_grid_extent[[0, 2, 1, 3]]
            crs = self.dem.crs
            self.dem = WarpedVRT(self.dem, crs=crs, resampling=Resampling.cubic, outputBounds=new_bounds)
            self.dem_zval = self.dem.read()
            self._get_raster_dimensions()
        logger.info('Cropped raster to geo_model.grid.extent.')

    def check(self):
        # TODO make this usable
        test = np.logical_and.reduce((self.regular_grid_extent[0] <= self.extent[0],
                                      self.regular_grid_extent[1] >= self.extent[1],
                                      self.regular_grid_extent[2] <= self.extent[2],
                                      self.regular_grid_extent[3] >= self.extent[3]))
        if test:
            cornerpoints_geo = self._get_cornerpoints(self.regular_grid_extent)
            cornerpoints_dtm = self._get_cornerpoints(self.extent)
            plt.scatter(cornerpoints_geo[:, 0], cornerpoints_geo[:, 1], label='grid extent')
            plt.scatter(cornerpoints_dtm[:, 0], cornerpoints_dtm[:, 1], label='raster extent')
            plt.legend(frameon=True, loc='upper left')
            raise AssertionError('The model extent is too different from the raster extent.')

    def convert2xyz(self):
        """
        Translates the gdal raster object to a numpy array of xyz coordinates.
        """
        shape = self.dem_zval.shape
        if len(shape) == 3:
            shape = shape[1:]
        
        b = self.dem.bounds
        xr, yr = self.dem.res
        X, Y = np.meshgrid(np.linspace(b.left, b.right, xr), np.linspace(b.top, b.bottom, yr))
        xyz = np.hstack((X.ravel(), Y.ravel(), self.dem_zval))
        self.values_3D = xyz.reshape((*shape, 3), order='C')
        # This is for 3D going from xyz to ijk
        self.values_3D = self.values_3D.swapaxes(0, 1)
        self.values_3D = np.flip(self.values_3D, 1)

        return self.values_3D

    def _get_cornerpoints(self, extent):
        """Get the coordinates of the bounding box.

        Args:
            extent: np.array([xmin, xmax, ymin, ymax)]

        Returns: np.ndarray with corner coordinates

        """
        upleft = ([extent[0], extent[3]])
        lowleft = ([extent[0], extent[2]])
        upright = ([extent[1], extent[3]])
        lowright = ([extent[1], extent[2]])
        return np.array([upleft, lowleft, upright, lowright])


class LoadDEMArtificial:

    def __init__(self, grid=None, fd=2.0, extent=None, resolution=None, d_z=None):
        """Class to create a random topography based on a fractal grid algorithm.

        Args:
            fd:         fractal dimension, defaults to 2.0
            d_z:        maximum height difference. If none, last 20% of the model in z direction
            extent:     extent in xy direction. If none, geo_model.grid.extent
            resolution: desired resolution of the topography array. If none, geo_model.grid.resolution
        """
        self.values_2d = np.array([])

        self.resolution = grid.resolution[:2] if resolution is None else resolution

        assert all(np.asarray(self.resolution) >= 2), 'The regular grid needs to be at least of size 2 on all ' \
                                                      'directions.'
        self.extent = grid.extent if extent is None else extent

        if d_z is None:
            self.d_z = np.array(
                [self.extent[5] - (self.extent[5] - self.extent[4]) * 1 / 5, self.extent[5]])
            logger.info(self.d_z)
        else:
            self.d_z = d_z

        topo = self.fractalGrid(fd, n=self.resolution.max())
        topo = np.interp(topo, (topo.min(), topo.max()), self.d_z)

        self.dem_zval = topo[:self.resolution[0], :self.resolution[1]]  # crop fractal grid with resolution
        self.create_topo_array()

    def fractalGrid(self, fd, n=256):
        """
        Modified after https://github.com/samthiele/pycompass/blob/master/examples/3_Synthetic%20Examples.ipynb

        Generate isotropic fractal surface image using
        spectral synthesis method [1, p.]
        References:
        1. Yuval Fisher, Michael McGuire,
        The Science of Fractal Images, 1988

        (cf. http://shortrecipes.blogspot.com.au/2008/11/python-isotropic-fractal-surface.html)
        **Arguments**:
         -fd = the fractal dimension
         -N = the size of the fractal surface/image

        """
        h = 1 - (fd - 2)
        # X = np.zeros((N, N), complex)
        a = np.zeros((n, n), complex)
        powerr = -(h + 1.0) / 2.0

        for i in range(int(n / 2) + 1):
            for j in range(int(n / 2) + 1):
                phase = 2 * np.pi * np.random.rand()

                if i != 0 or j != 0:
                    rad = (i * i + j * j) ** powerr * np.random.normal()
                else:
                    rad = 0.0

                a[i, j] = complex(rad * np.cos(phase), rad * np.sin(phase))

                if i == 0:
                    i0 = 0
                else:
                    i0 = n - i

                if j == 0:
                    j0 = 0
                else:
                    j0 = n - j

                a[i0, j0] = complex(rad * np.cos(phase), -rad * np.sin(phase))

                a.imag[int(n / 2)][0] = 0.0
                a.imag[0, int(n / 2)] = 0.0
                a.imag[int(n / 2)][int(n / 2)] = 0.0

        for i in range(1, int(n / 2)):
            for j in range(1, int(n / 2)):
                phase = 2 * np.pi * np.random.rand()
                rad = (i * i + j * j) ** powerr * np.random.normal()
                a[i, n - j] = complex(rad * np.cos(phase), rad * np.sin(phase))
                a[n - i, j] = complex(rad * np.cos(phase), -rad * np.sin(phase))

        itemp = fftpack.ifft2(a)
        itemp = itemp - itemp.min()

        return itemp.real / itemp.real.max()

    def create_topo_array(self):
        """for masking the lith block"""
        x = np.linspace(self.extent[0], self.extent[1], self.resolution[0])
        y = np.linspace(self.extent[2], self.extent[3], self.resolution[1])
        self.x = x
        self.y = y
        xx, yy = np.meshgrid(x, y, indexing='ij')
        self.values_2d = np.dstack([xx, yy, self.dem_zval])

    def get_values(self):
        return self.values_2d
