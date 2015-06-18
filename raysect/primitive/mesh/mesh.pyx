# cython: language_level=3

# Copyright (c) 2015, Dr Alex Meakins, Raysect Project
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

from raysect.core.math.affinematrix cimport AffineMatrix
from raysect.core.math.normal cimport Normal, new_normal
from raysect.core.math.point cimport new_point
from raysect.core.classes cimport Material, new_intersection
from raysect.core.acceleration.boundingbox cimport BoundingBox
from libc.math cimport fabs
cimport cython

# cython doesn't have a built-in infinity constant, this compiles to +infinity
DEF INFINITY = 1e999

# bounding box is padded by a small amount to avoid numerical accuracy issues
DEF BOX_PADDING = 1e-9

# additional ray distance to avoid re-hitting the same surface point
DEF EPSILON = 1e-9

cdef class Mesh(Primitive):

    def __init__(self, object vertices, object polygons, object normals=None, #object texture_coords=None, bint validate=True, bint optimise_now=True,
                 object parent = None, AffineMatrix transform not None = AffineMatrix(), Material material not None = Material(), unicode name not None= ""):

        # convert to internal numpy arrays and obtain memory views

        # validate
            # check vertices
            # check normals are valid
            # check polygons are not dangling

        # build kdtree

        pass


    # todo: get/set attributes must return copies of arrays to protect internals of the mesh object

    cdef _build_kdtree():

        pass


    cpdef Intersection hit(self, Ray ray):
        """
        Virtual method - to be implemented by derived classes.

        Calculates the closest intersection of the Ray with the Primitive
        surface, if such an intersection exists.

        If a hit occurs an Intersection object must be returned, otherwise None
        is returned. The intersection object holds the details of the
        intersection including the point of intersection, surface normal and
        the objects involved in the intersection.
        """

        raise NotImplementedError("Primitive surface has not been defined. Virtual method hit() has not been implemented.")

    cpdef Intersection next_intersection(self):
        """
        Virtual method - to be implemented by derived classes.

        Returns the next intersection of the ray with the primitive along the
        ray path.

        This method may only be called following a call to hit(). If the ray
        has further intersections with the primitive, these may be obtained by
        repeatedly calling the next_intersection() method. Each call to
        next_intersection() will return the next ray-primitive intersection
        along the ray's path. If no further intersections are found or
        intersections lie outside the ray parameters then next_intersection()
        will return None.

        If any geometric elements of the primitive, ray and/or scenegraph are
        altered between a call to hit() and calls to next_intersection() the
        data returned by next_intersection() may be invalid. Primitives may
        cache data to accelerate next_intersection() calls which will be
        invalidated by geometric alterations to the scene. If the scene is
        altered the data returned by next_intersection() is undefined.
        """

        raise NotImplementedError("Primitive surface has not been defined. Virtual method next_intersection() has not been implemented.")

    cpdef bint contains(self, Point p) except -1:
        """
        Virtual method - to be implemented by derived classes.

        Must returns True if the Point lies within the boundary of the surface
        defined by the Primitive. False is returned otherwise.
        """

        raise NotImplementedError("Primitive surface has not been defined. Virtual method inside() has not been implemented.")

    cpdef BoundingBox bounding_box(self):
        """
        Virtual method - to be implemented by derived classes.

        When the primitive is connected to a scenegrpah containing a World
        object at its root, this method should return a bounding box that
        fully encloses the primitive's surface (plus a small margin to
        avoid numerical accuracy problems). The bounding box must be defined in
        the world's coordinate space.

        If this method is called when the primitive is not connected to a
        scenegraph with a World object at its root, it must throw a TypeError
        exception.
        """

        raise NotImplementedError("Primitive surface has not been defined. Virtual method bounding_box() has not been implemented.")
