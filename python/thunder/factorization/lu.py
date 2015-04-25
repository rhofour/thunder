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
    `p` : matrix, n x 1
        The indices from a permutation matrix
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

    def _permute(self, p, mat):
        """
        Given permutation vector and RowMatrix, permute the matrix.

        Assumes the matrix RDD's keys are its rows' 0-indexed indices,
        and does NOT sort the RDD by key afterward.
        """
        return mat.applyKeys(lambda x: p.item(x))

    def _times(self, a, b):
        """
        Given a RowMatrix A and RowMatrix B, compute A*B^T

        Assumes the matrix RDD's keys are its rows' 0-indexed indices
        """
        from numpy import add, vdot, zeros
        nrows = a.nrows
        glommedA = a.rdd.glom()
        glommedB = b.rdd.glom()
        prod = glommedA.cartesian(glommedB)
        def computeElems(x):
          return [(ai, (bi, vdot(ar, br))) for (ai, ar) in x[0] for (bi, br) in x[1]]
        def addCoord(col, x):
          col[x[0]] = x[1]
          return col
        def combCols(col1, col2):
          add(col1, col2, col1)
          return col1
        resRdd = prod.flatMap(computeElems).aggregateByKey(zeros(nrows), addCoord, combCols)
        return RowMatrix(resRdd)

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

        from numpy import arange, matrix, vdot, zeros
        from scipy.linalg import lu

        if not (isinstance(mat, RowMatrix)):
          raise Exception('Input must be a RowMatrix')
        if not (mat.nrows == mat.ncols):
          raise Exception('Input matrix must be square')

        # Do all the computation on the master node if everything is small
        # enough
        if mat.nrows <= self.nb:
          p, l, u = lu(mat.collectValuesAsArray(), overwrite_a=True, permute_l=False)
          self.p = p * matrix(arange(0, len(p))).transpose()
          self.l = RowMatrix(mat.rdd.context.parallelize(enumerate(l), mat.rdd.getNumPartitions()))
          self.u = RowMatrix(mat.rdd.context.parallelize(enumerate(u), mat.rdd.getNumPartitions()))
          return self

        mat = mat.keysToIndices()
        halfRows = mat.nrows / 2
        aTop = mat.filterOnKeys(lambda k: k < halfRows)
        a1 = aTop.between(0, halfRows - 1)
        a2 = aTop.between(halfRows, mat.ncols)
        aBot = mat.filterOnKeys(lambda k: k >= halfRows).keysToIndices()
        a3 = aBot.between(0, halfRows - 1)
        a4 = aBot.between(halfRows, mat.ncols)

        lup1 = LU(nb=self.nb).calc(a1)

        # Permute A1 and A2 using P1
        a1 = self._permute(lup1.p, a1)
        a2 = self._permute(lup1.p, a2)

        # Take the transpose of A2
        a2.rdd = a2.rdd.sortByKey() # Fix the order of A2 first
        a2t = a2.transpose(replaceKeys=False)

        nA2Rows = a2.nrows # Store this so we don't query to RDD inside of applyValues
        # Combine values of A2T with empty rows that will become U2T
        a2t_u2t = a2t.applyValues(lambda x: (x, zeros(nA2Rows)))

        def computeElementFactory(l1Row, colIdx):
          # This is a horrible hack because Python closures are late binding
          def computeElement(pairedRows):
            (a2tRow, u2tRow) = pairedRows
            x = (1.0 / l1Row.value[colIdx.value]) * (a2tRow[colIdx.value] - vdot(l1Row.value, u2tRow))
            u2tRow[colIdx.value] = x
            return (a2tRow, u2tRow)
          return computeElement
        for i in xrange(nA2Rows):
          l1Row = mat.rdd.context.broadcast(lup1.l.get(i))
          colIdx = mat.rdd.context.broadcast(i)
          a2t_u2t = a2t_u2t.applyValues(computeElementFactory(l1Row, colIdx))

        # Extract the U2T rows
        u2t = a2t_u2t.applyValues(lambda x: x[1])
        u2t.repartition(mat.rdd.getNumPartitions())
        u2t.rdd = u2t.rdd.sortByKey()

        return self, a1, a2, a3, a4, lup1, u2t
