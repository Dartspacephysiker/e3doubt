import numpy as np
def dotprod(a,b):
    """
    dp = dotprod(a,b)

    Assuming a and b are arrays of N three-dimensional vectors 
    (i.e., they have shape (N, 3)), return an array of N values.

    Each value corresponding to the dot products of a pair of vectors in a and b.

    SMH
    Birkeland Centre for Space Science
    2020-10-02
    """

    return np.einsum('ij,ij->i',a,b)

def crossprod(a,b):
    """
    As in dotprod, here we assume that the vectors themselves 
    are indexed by the zeroth axis (i.e., rows), and the vector 
    components by the first axis (i.e., columns).
    """
    return np.cross(a,b,axis=1)


def vecmag(a):
    return np.sqrt(dotprod(a,a))
