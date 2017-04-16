# my_vector = Vector([1,2,3])
# print(my_vector) = vector: 1,2,3
# my_vector1 == my_vector2 : True or False
from math import sqrt, acos, pi

#####

# VECTOR CLASS

#####

class Vector(object):

    CANNOT_NORMALIZE_ZERO_VECTOR = 'cannot normalize zero vector'
    VECTOR_LENGTHS_NOT_EQUAL = 'vector lengths not equal'
    NO_UNIQUE_PARALLEL_COMPONENT = 'no unique parallel component!'

    def __init__(self, coordinates):
        try:
            if not coordinates:
                raise ValueError
            self.coordinates = tuple(coordinates)
            self.dimension = len(coordinates)

        except ValueError:
            raise ValueError('coordinates must be non-empty')

        except TypeError:
            raise TypeError('coordinates must be an iterable')

        def __str__(self):
            return 'vector: {}'.format(self.coordinates)

        def __eq__(self, v):
            return self.coordinates == v.coordinates

    # vector addition
    def plus(self, vec):
        try:
            if len(self.coordinates) != len(vec.coordinates):
                raise ValueError

            result = [x+y for x, y in zip(self.coordinates, vec.coordinates)]
            return(Vector(result))

        except ValueError:
            raise ValueError('vector lengths not equal')

    # vector subtraction
    def minus(self, vec):
        try:
            if len(self.coordinates) != len(vec.coordinates):
                raise ValueError

            result = [x-y for x, y in zip(self.coordinates, vec.coordinates)]
            return(Vector(result))

        except ValueError:
            raise ValueError('vector lengths not equal')

    # scalar multiplication
    def scalarmultiply(self, scalar):
        result = [x*scalar for x in self.coordinates]
        return(Vector(result))
        
    def scalarmult(self, scalar):
        return(self.scalarmultiply(scalar))

    # vector magnitude: square root of sum of squares
    def magnitude(self):
        denom = [x**2 for x in self.coordinates]
        result = sqrt(sum(denom))
        return(result)

    # normalized vector (inputs considered points)
    def normalized(self):
        try:
            denom = self.magnitude()
            result = [x/denom for x in self.coordinates]
            return(Vector(result))
            
        except ZeroDivisionError:
            raise Exception('cannot normalize zero vector')

    # direction unit vector (as values)
    def direction(self):
        dvec = self.normalized()
        return(dvec.coordinates)

    # vector inner/dot product (sum of piecewise multiplication)
    def inner(self, vec):
        try:
            if len(self.coordinates) != len(vec.coordinates):
                raise ValueError

            temp = [x*y for x, y in zip(self.coordinates, vec.coordinates)]
            return(sum(temp))

        except ValueError:
            raise ValueError('vector lengths not equal')

    def dot(self, vec):
        return(self.inner(vec))

    # angle between two vectors as cosine
    def cosineto(self, vec):
        try:
            if len(self.coordinates) != len(vec.coordinates):
                raise ValueError

            mags = self.magnitude() * vec.magnitude()
            if mags == 0:
                print('notice: attempting to find cosine of 0 vector')
                return(0.0)
            else:
                return(self.dot(vec) / mags)

        except ValueError:
            raise ValueError('vector lengths not equal')
        
    # angle between two vectors as radians
    # a dot b = mag a * mag b * cos theta
    def radiansto(self, vec):
        try:
            if len(self.coordinates) != len(vec.coordinates):
                raise ValueError

            rads = acos(self.cosineto(vec))
            return(rads)

        except ValueError:
            raise ValueError('vector lengths not equal')
        except Exception as e:
            if str(e) == self.CANNOT_NORMALIZE_ZERO_VECTOR:
                raise Exception('cannot compute angle with zero vector')

    def radsto(self, vec):
        return(self.radiansto(vec))
        
    # angle between two vectors as degrees
    # degrees = rads * 180 / pi
    def degreesto(self, vec):
        return(self.radiansto(vec)*180/pi)
    def degsto(self, vec):
        return(self.degreesto(vec))

    # check for parallel (cos = -1, 1)
    def isparallel(self, vec):
        if vec.magnitude() == 0 or self.magnitude() == 0:
            return(True)
        cos = self.cosineto(vec)
        if (0.999 < cos < 1.001) or (-1.001 < cos < -0.999):
            return(True)
        else:
            return(False)

    # check for orthogonal (cos = 0)
    def isorthogonal(self, vec):
        cos = self.cosineto(vec)
        if (-0.001 < cos < 0.001):
            return(True)
        else:
            return(False)

    def isright(self, vec):
        return(self.isorthogonal(vec))

    # projection of v to b: proj_b(v) = (v dot unit_b) * unit_b
    def projection(self, vec):
        try:
            unit_vec = vec.normalized()
            return(unit_vec.scalarmult(self.dot(unit_vec)))

        except Exception as e:
            if str(e) == self.CANNOT_NORMALIZE_ZERO_VECTOR:
                raise Exception('no unique parallel component!')
            else:
                raise e

    def componentparallelto(self, vec):
        return(self.projection(vec))

    # orthogonal of v from b: v minus proj_b(v)
    def orthogonal(self, vec):
        try:
            proj_vec = self.projection(vec)
            return(self.minus(proj_vec))

        except Exception as e:
            if str(e) == self.NO_UNIQUE_PARALLEL_COMPONENT:
                raise Exception('no unique orthogonal component!')
            else:
                raise e

    def componentorthogonalto(self, vec):
        return(self.orthogonal(vec))

    # cross product: v x w orthogonal to v and w
    #                ||v x w|| = ||v|| ||w|| sin theta
    #                direction found by 'right hand rule'
    #                'anticommutative': order matters
    def cross(self, vec):
        try:
            if not (1 < len(self.coordinates) < 4) or not (1 < len(
                self.coordinates) < 4):
                raise ValueError
            elif len(self.coordinates) != len(vec.coordinates):
                raise ValueError

            v1 = list(self.coordinates)
            v2 = list(vec.coordinates)

            if len(self.coordinates) == 2:
                v1.insert(0, 0)
                v2.insert(0, 0)
            
            a = v1[1]*v2[2] - v2[1]*v1[2]
            b = v1[0]*v2[2] - v2[0]*v1[2]
            c = v1[0]*v2[1] - v2[0]*v1[1]
            result = [a, -b, c]
            return(Vector(result))

        except ValueError:
            raise ValueError('vector lengths must be equal; dim 2 or 3')

    # area of triangle spanned by two vectors = 1/2 mag (a cross b)
    def trianglearea(self, vec):
        rvec = self.cross(vec)
        return(0.5*rvec.magnitude())
    
#####

# LINE CLASS

#####

# if separating classes by file...
# from vector import Vector

class Line(object):

    NON_NONZERO_ELMTS_FOUND_MSG = 'No nonzero elements found'

    def __init__(self, normal_vector=None, constant_term=None):
        self.dimension = 2

        if not normal_vector:
            all_zeros = ['0']*self.dimension
            normal_vector = Vector(all_zeros)
        self.normal_vector = normal_vector

        if not constant_term:
            constant_term = 0.0
        self.constant_term = constant_term


# quiz questions 1
print('addition, subtraction and scalar multiplication')
a = Vector([8.218, -9.341])
b = Vector([-1.129, 2.111])
print("Q1:",a.plus(b).coordinates)
a = Vector([7.119, 8.215])
b = Vector([-8.223, 0.878])
print("Q2:",a.minus(b).coordinates)
a = Vector([1.671,-1.012,-0.318])
b = 7.41
print("Q3:",a.scalarmult(b).coordinates)
print('')

# quiz questions 2
print('magnitude and direction')
a = Vector([-0.221,7.437])
print("mag of",a.coordinates,":",a.magnitude())
a = Vector([8.813,-1.331,-6.247])
print("mag of",a.coordinates,":",a.magnitude())
a = Vector([5.581,-2.136])
print("dir of",a.coordinates,":",a.direction())
a = Vector([1.996,3.108,-4.554])
print("dir of",a.coordinates,":",a.direction())
print('')

#quiz questions 3
print('dot product and angle')
a = Vector([7.887,4.138])
b = Vector([-8.802, 6.776])
print("Q1 dot:",a.dot(b))
a = Vector([-5.955, -4.904, -1.874])
b = Vector([-4.496, -8.755, 7.103])
print("Q2 dot:",a.dot(b))
a = Vector([3.183,-7.627])
b = Vector([-2.668,5.319])
print("Q3 rads:",a.radiansto(b))
a = Vector([7.35,0.221,5.188])
b = Vector([2.751,8.259,3.985])
print("Q4 degs:", a.degreesto(b))
print('')

# quiz questions 4
print('parallelism & orthogonality')
a = Vector([-7.579,-7.88])
b = Vector([22.737,23.64])
print("Para:",a.isparallel(b),"Orth:",a.isright(b))
a = Vector([-2.029,9.97,4.172])
b = Vector([-9.231,-6.639,-7.245])
print("Para:",a.isparallel(b),"Orth:",a.isright(b))
a = Vector([-2.328,-7.284,-1.214])
b = Vector([-1.821,1.072,-2.94])
print("Para:",a.isparallel(b),"Orth:",a.isright(b))
a = Vector([2.118,4.827])
b = Vector([0.0,0.0])
print("Para:",a.isparallel(b),"Orth:",a.isright(b))
print('')

# quiz questions 5
print('vector projections & orthogonals')
v = Vector([3.039, 1.879])
b = Vector([0.825, 2.036])
print("A proj_b(v):",v.projection(b).coordinates)
v = Vector([-9.88, -3.264, -8.159])
b = Vector([-2.155, -9.353, -9.473])
print("B orth_b(v):",v.orthogonal(b).coordinates)
v = Vector([3.009, -6.172, 3.692, -2.51])
b = Vector([6.404, -9.144, 2.759, 8.718])
print("C proj_b(v):",v.projection(b).coordinates)
print("C orth_b(v):",v.orthogonal(b).coordinates)
print('')

# quiz questions 6
print('cross products')
a = Vector([8.462, 7.893, -8.187])
b = Vector([6.984, -5.975, 4.778])
c = a.cross(b)
print('cross product vector:', c.coordinates)
print('check dot prods == 0:', 'a',a.dot(c), 'b',b.dot(c))
a = Vector([-8.987, -9.838, 5.031])
b = Vector([-4.268, -1.861, -8.866])
print('area of parallelogram:', a.trianglearea(b) * 2)
print('area of parallelogram:', a.cross(b).magnitude())
a = Vector([1.5, 9.547, 3.691])
b = Vector([-6.007, 0.124, 5.772])
print('area of triangle:', a.trianglearea(b))
print('area of triangle:', a.cross(b).magnitude()*0.5)
a = Vector([8.462, 7.893])
b = Vector([6.984, -5.975])
c = a.cross(b)
print('2D vector test!!:', c.coordinates)
