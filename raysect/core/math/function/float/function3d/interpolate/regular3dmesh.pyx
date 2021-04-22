# cython: language_level=3

# Copyright (c) 2014-2020, Dr Alex Meakins, Raysect Project
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright notice,
#        this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#
#     3. Neither the name of the Raysect Project nor the names of its
#        contributors may be used to endorse or promote products derived from
#        this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numpy as np
cimport numpy as np
from raysect.core cimport Point3D
from raysect.core.math.function.float.function3d cimport Function3D
from libc.math cimport sqrt, atan2, M_PI
cimport cython


cdef class RegularCartesianMesh(Function3D):
    """
    Discrete interpolator for data on a 3d regular Cartesian mesh.
    
    The mesh is defined by the lower and upper bounds along each axis.

    The shape of the data array sets the number of mesh cells along each axis.

    By default, requesting a point outside the bounds of the mesh will cause
    a ValueError exception to be raised. If this is not desired the limit
    attribute (default True) can be set to False. When set to False, a default
    value will be returned for any point lying outside the mesh. The value
    return can be specified by setting the default_value attribute (default is
    0.0).

    :param Point3D lower: The lower boundary of the mesh along each axis.
    :param Point3D upper: The upper boundary of the mesh along each axis.
    :param ndarray data: A 3D array containing the data for each mesh cell.
    :param bool limit: Raise an exception outside mesh limits - True (default) or False.
    :param float default_value: The value to return outside the mesh limits if limit is set to False.
    """

    def __init__(self, Point3D lower not None, Point3D upper not None, object data not None, bint limit=True, double default_value=0.0):

        data = np.asarray(data, dtype=np.float64)

        if data.ndim != 3:
            raise ValueError("Argument 'data' must be a 3D array.")

        if data.shape[0] == 0 or data.shape[1] == 0 or data.shape[2] == 0:
            raise ValueError("Argument 'data' must have non-zero shapes.")

        if lower.x >= upper.x or lower.y >= upper.y or lower.z >= upper.z:
            raise ValueError("The lower limits of the mesh must be less than the upper limits.")

        self._xmin = lower.x
        self._ymin = lower.y
        self._zmin = lower.z
        self._xmax = upper.x
        self._ymax = upper.y
        self._zmax = upper.z
        self._data = data

        self._data_mv = data
        self._default_value = default_value
        self._limit = limit

        self._dx = (self._xmax - self._xmin) / self._data.shape[0]
        self._dy = (self._ymax - self._ymin) / self._data.shape[1]
        self._dz = (self._zmax - self._zmin) / self._data.shape[2]

    def __getstate__(self):
        return (
            self._xmin,
            self._ymin,
            self._zmin,
            self._xmax,
            self._ymax,
            self._zmax,
            self._data,
            self._limit,
            self._default_value
        )

    def __setstate__(self, state):
        (
            self._xmin,
            self._ymin,
            self._zmin,
            self._xmax,
            self._ymax,
            self._zmax,
            self._data,
            self._limit,
            self._default_value
        ) = state
        self._data_mv = self._data
        self._dx = (self._xmax - self._xmin) / self._data.shape[0]
        self._dy = (self._ymax - self._ymin) / self._data.shape[1]
        self._dz = (self._zmax - self._zmin) / self._data.shape[2]

    def __reduce__(self):
        return self.__new__, (self.__class__, ), self.__getstate__()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cdef double evaluate(self, double x, double y, double z) except? -1e999:

        cdef:
            np.int32_t ix, iy, iz

        if self._xmin <= x <= self._xmax and self._ymin <= y <= self._ymax and self._zmin <= z <= self._zmax:
            # if the point is at the upper mesh borders, it resides in the mesh
            ix = min(self._data.shape[0] - 1, <np.int32_t>((x - self._xmin) / self._dx))
            iy = min(self._data.shape[1] - 1, <np.int32_t>((y - self._ymin) / self._dy))
            iz = min(self._data.shape[2] - 1, <np.int32_t>((z - self._zmin) / self._dz))
            return self._data_mv[ix, iy, iz]

        if not self._limit:
            return self._default_value

        raise ValueError("Requested value outside mesh bounds.")


cdef class RegularCylindricalMesh(Function3D):
    """
    Discrete interpolator for data on a regular cylindrical mesh.

    The mesh is defined by the lower and upper bounds along each axis: (R, phi, Z).
    Namely, lower.x, lower.y and lower.z define the radial, azimuthal and axial lower
    bounds of the mesh, respectively. The azimuthal coordinate is given in degrees.

    The lower.y must be in the range [-180, 180), the upper.y in the range (-180, 540).
    The azimuthal sector defined by the difference between upper.y and lower.y must not
    exceed 360.

    The shape of the data array sets the number of mesh cells along each axis.

    The point at which the function is evaluated is given in Cartesian coordinates. 

    By default, requesting a point outside the bounds of the mesh will cause
    a ValueError exception to be raised. If this is not desired the limit
    attribute (default True) can be set to False. When set to False, a default
    value will be returned for any point lying outside the mesh. The value
    return can be specified by setting the default_value attribute (default is
    0.0).

    In axisymmetric case use PeriodicRegularCylindricalMesh for better performance.

    :param Point3D lower: The lower boundary of the mesh along each axis.
    :param Point3D upper: The upper boundary of the mesh along each axis.
    :param ndarray data: A 3D array containing the data for each mesh cell.
    :param bool limit: Raise an exception outside mesh limits - True (default) or False.
    :param float default_value: The value to return outside the mesh limits if limit is set to False.
    """

    def __init__(self, Point3D lower not None, Point3D upper not None, object data not None,
                 bint limit=True, double default_value=0.0):

        if lower.x > upper.x or lower.y > upper.y or lower.z > upper.z:
            raise ValueError("The lower limits of the mesh must be less than the upper limits.")

        if lower.x < 0 or upper.x < 0:
            raise ValueError("The radial limits of the mesh must be positive.")

        if not -180. <= lower.y < 360.:
            raise ValueError("The lower limit for the sector angle must be defined in [-180, 180) deg.")

        if not -180. < upper.y < 540.:
            raise ValueError("The upper limit for the sector angle must be defined in (-180, 540) deg.")

        if upper.y - lower.y > 360.:
            raise ValueError("The sector angle must be less or equal to 360.")

        data = np.asarray(data, dtype=np.float64)

        if data.ndim != 3:
            raise ValueError("Argument 'data' must be a 3D array.")

        if data.shape[0] == 0 or data.shape[1] == 0 or data.shape[2] == 0:
            raise ValueError("Argument 'data' must have non-zero shapes.")

        self._rmin = lower.x
        self._phimin = lower.y
        self._zmin = lower.z
        self._rmax = upper.x
        self._phimax = upper.y
        self._zmax = upper.z
        self._data = data
        self._data_mv = data
        self._default_value = default_value
        self._limit = limit

        self._dr = (self._rmax - self._rmin) / self._data.shape[0]
        self._dphi = (self._phimax - self._phimin) / self._data.shape[1]
        self._dz = (self._zmax - self._zmin) / self._data.shape[2]

    def __getstate__(self):
        return (
            self._rmin,
            self._phimin,
            self._zmin,
            self._rmax,
            self._phimax,
            self._zmax,
            self._data,
            self._limit,
            self._default_value
        )

    def __setstate__(self, state):
        (
            self._rmin,
            self._phimin,
            self._zmin,
            self._rmax,
            self._phimax,
            self._zmax,
            self._data,
            self._limit,
            self._default_value
        ) = state
        self._data_mv = self._data
        self._dr = (self._rmax - self._rmin) / self._data.shape[0]
        self._dphi = (self._phimax - self._phimin) / self._data.shape[1]
        self._dz = (self._zmax - self._zmin) / self._data.shape[2]

    def __reduce__(self):
        return self.__new__, (self.__class__, ), self.__getstate__()

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cdef double evaluate(self, double x, double y, double z) except? -1e999:

        cdef:
            np.int32_t ir, iphi, iz
            double r, phi

        r = sqrt(x * x + y * y)

        phi = (180. / M_PI) * atan2(y, x)
        if phi < self._phimin:
            phi += 360.

        if self._rmin <= r <= self._rmax and phi <= self._phimax and self._zmin <= z <= self._zmax:
            # if the point is at the upper mesh borders, it resides in the mesh
            ir = min(self._data.shape[0] - 1, <np.int32_t>((r - self._rmin) / self._dr))
            iphi = min(self._data.shape[1] - 1, <np.int32_t>((phi - self._phimin) / self._dphi))
            iz = min(self._data.shape[2] - 1, <np.int32_t>((z - self._zmin) / self._dz))
            return self._data_mv[ir, iphi, iz]

        if not self._limit:
            return self._default_value

        raise ValueError("Requested value outside mesh bounds.")


DEF _EPSILON = 1.e-6


cdef class PeriodicRegularCylindricalMesh(RegularCylindricalMesh):
    """
    Discrete interpolator for data on a regular cylindrical mesh periodic in azimuthal
    direction.

    The mesh is defined by the lower and upper bounds along each axis: (R, phi, Z).
    Namely, lower.x, lower.y and lower.z define the radial, azimuthal and axial lower
    bounds of the mesh, respectively. The azimuthal coordinate is given in degrees.

    The lower.y must be in the range [-180, 180), the upper.y in the range (-180, 540).
    The azimuthal sector defined by the difference between upper.y and lower.y must
    me a multiple of 360.

    The shape of the data array sets the number of mesh cells along each axis.

    Note that this interpolator works faster than RegularCylindricalMesh in axisymmetric
    case, namely when the number of mesh cells in azimuthal direction is equal to 1.

    The point at which the function is evaluated is given in Cartesian coordinates.

    By default, requesting a point outside the bounds of the mesh will cause
    a ValueError exception to be raised. If this is not desired the limit
    attribute (default True) can be set to False. When set to False, a default
    value will be returned for any point lying outside the mesh. The value
    return can be specified by setting the default_value attribute (default is
    0.0).

    :param Point3D lower: The lower boundary of the mesh along each axis.
    :param Point3D upper: The upper boundary of the mesh along each axis.
    :param ndarray data: A 3D array containing the data for each mesh cell.
    :param bool limit: Raise an exception outside mesh limits - True (default) or False.
    :param float default_value: The value to return outside the mesh limits if limit is set to False.
    """

    def __init__(self, Point3D lower not None, Point3D upper not None, object data not None,
                 bint limit=True, double default_value=0.0):

        cdef:
            double priod, num_sectors

        period = upper.y - lower.y
        num_sectors = 360. / period
        if abs(round(num_sectors) - num_sectors) > _EPSILON:
            raise ValueError("The sector angle {} deg is not a multiple of 360.".format(period))

        super().__init__(lower, upper, data, limit, default_value)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.initializedcheck(False)
    @cython.cdivision(True)
    cdef double evaluate(self, double x, double y, double z) except? -1e999:

        cdef:
            np.int32_t ir, iphi, iz
            double r, phi

        r = sqrt(x * x + y * y)

        if self._data.shape[1] == 1:  # optimisation for axisymmetric case
            iphi = 0
        else:
            phi = (180. / M_PI) * atan2(y, x) + 360.
            phi = (phi - self._phimin) % (self._phimax - self._phimin)  # moving into the [0, period) sector
            iphi = <np.int32_t>(phi / self._dphi)

        if self._rmin <= r <= self._rmax and self._zmin <= z <= self._zmax:
            # if the point is at the upper mesh borders, it resides in the mesh
            ir = min(self._data.shape[0] - 1, <np.int32_t>((r - self._rmin) / self._dr))
            iz = min(self._data.shape[2] - 1, <np.int32_t>((z - self._zmin) / self._dz))
            return self._data_mv[ir, iphi, iz]

        if not self._limit:
            return self._default_value

        raise ValueError("Requested value outside mesh bounds.")
