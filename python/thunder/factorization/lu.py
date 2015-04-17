"""
Class for performing LU-decomposition
"""

from thunder.rdds.matrices import RowMatrix


class LU(object):
    """
    LU decomposition on a distributed matrix

    Attributes
    ----------
    `l` : RowMatrix, n x n
        Lower triangular matrix
    `u` : RowMatrix, n x n
        Upper triangular matrix
    """
    def __init__(self):
        self.l = None
        self.u = None

    def calc(self, mat):
        """
        Calculate LU-decomposition using a recursive block method

        Parameters
        ----------
        mat : RowMatrix
            Matrix to compute LU decomposition of

        Returns
        ----------
        self : returns an instance of self.
        """
        if not (isinstance(mat, RowMatrix)):
          raise Exception('Input must be a RowMatrix')
