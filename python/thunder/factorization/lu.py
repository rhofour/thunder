"""
Class for performing LU-decomposition
"""

from thunder.rdds.matrices import RowMatrix


class LU(object):
    """
    LU decomposition on a distributed matrix

    Parameters
    ----------
    nb : int, optional, default = 3200
        Size of the largest matrix which we decompose locally

    Attributes
    ----------
    `l` : RowMatrix, n x n
        A pivot matrix
    `l` : RowMatrix, n x n
        Lower triangular matrix
    `u` : RowMatrix, n x n
        Upper triangular matrix
    """
    def __init__(self, nb=3200):
        self.nb = nb
        self.p = None
        self.l = None
        self.u = None

    def calc(self, mat):
        """
        Calculate LU-decomposition using a recursive block method with pivoting

        Parameters
        ----------
        mat : RowMatrix
            A square matrix to compute LU decomposition of

        Returns
        ----------
        self : returns an instance of self.
        """

        from scipy.linalg import lu

        if not (isinstance(mat, RowMatrix)):
          raise Exception('Input must be a RowMatrix')
        if not (mat.nrows == mat.ncols):
          raise Exception('Input matrix must be square')

        if mat.nrows <= self.nb:
          p, l, u = lu(mat.collectValuesAsArray())
          self.p = RowMatrix(mat.rdd.context.parallelize(enumerate(p)))
          self.l = RowMatrix(mat.rdd.context.parallelize(enumerate(l)))
          self.u = RowMatrix(mat.rdd.context.parallelize(enumerate(u)))

        return self
