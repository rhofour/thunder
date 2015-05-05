from numpy import asarray, mean, sqrt, ndarray, amin, amax, concatenate, sum, zeros, maximum, \
    argmin, newaxis, ones, delete, NaN, inf, isnan, clip

from thunder.utils.serializable import Serializable
from thunder.utils.common import checkParams, aslist
from thunder.rdds.images import Images
from thunder.rdds.series import Series


class Source(Serializable, object):
    """
    A single source, represented as a list of coordinates and other optional specifications.

    A source also has a set of lazily computed attributes useful for representing and comparing
    its geometry, such as center, bounding box, and bounding polygon. These properties
    will be computed lazily and made available as attributes when requested.

    Parameters
    ----------
    coordinates : array-like
        List of 2D or 3D coordinates, can be a list of lists or array of shape (n,2) or (n,3)

    values : list or array-like
        Value (or weight) associated with each coordiante

    id : int or string
        Arbitrary specification per source, typically an index or string label

    Attributes
    ----------
    center : list or array-like
        The coordinates of the center of the source

    polygon : list or array-like
        The coordinates of a polygon bounding the region (a convex hull)

    bbox : list or array-like
        Boundaries of the source (with the lowest values for all axes followed by the highest values)

    area : scalar
        The area of the region
    """
    from zope.cachedescriptors import property

    def __init__(self, coordinates, values=None, id=None):
        self.coordinates = asarray(coordinates)

        if self.coordinates.ndim == 1:
            self.coordinates = asarray([self.coordinates])

        if values is not None:
            self.values = asarray(values)
            if self.values.ndim == 0:
                self.values = asarray([self.values])

        if id is not None:
            self.id = id

    @property.Lazy
    def center(self):
        """
        Find the region center using a mean.
        """
        # TODO Add option to use weights
        return mean(self.coordinates, axis=0)

    @property.Lazy
    def polygon(self):
        """
        Find the bounding polygon as a convex hull
        """
        # TODO Add option for simplification
        from scipy.spatial import ConvexHull
        if len(self.coordinates) >= 4:
            inds = ConvexHull(self.coordinates).vertices
            return self.coordinates[inds]
        else:
            return self.coordinates

    @property.Lazy
    def bbox(self):
        """
        Find the bounding box.
        """
        mn = amin(self.coordinates, axis=0)
        mx = amax(self.coordinates, axis=0)
        return concatenate((mn, mx))

    @property.Lazy
    def area(self):
        """
        Find the region area.
        """
        return len(self.coordinates)

    def restore(self, skip=None):
        """
        Remove all lazy properties, will force recomputation
        """
        if skip is None:
            skip = []
        elif isinstance(skip, str):
            skip = [skip]
        for prop in LAZY_ATTRIBUTES:
            if prop in self.__dict__.keys() and prop not in skip:
                del self.__dict__[prop]
        return self

    def distance(self, other, method='euclidean'):
        """
        Distance between the center of this source and another.

        Parameters
        ----------
        other : Source, or array-like
            Either another source, or the center coordinates of another source

        method : str
            Specify a distance measure to used for spatial distance between source
            centers. Current options include Euclidean distance ('euclidean') and 
            L1-norm ('l1'). 

        """
        from numpy.linalg import norm

        checkParams(method, ['euclidean', 'l1'])

        if method == 'l1':
            order = 1
        else:
            order = 2

        if isinstance(other, Source):
            return norm(self.center - other.center, ord=order)
        elif isinstance(other, list) or isinstance(other, ndarray):
            return norm(self.center - asarray(other), ord=order)

    def overlap(self, other, method='support', counts=False):
        """
        Compute the overlap between this source and other, in terms
        of either support or similarity of coefficients.

        Support computes the number of overlapping pixels relative
        to the union of both sources. Correlation computes the similarity
        of the weights (not defined for binary masks).

        Parameters
        ----------
        other : Source
            The source to compute overlap with.

        method : str
            Compare either support of source coefficients ('support'), or the 
            source spatial filters (not yet implemented).

        counts : boolean, optional, default = True
            Whether to return raw counts when computing support, otherwise
            return a fraction.

        """
        checkParams(method, ['support', 'corr'])

        coordsSelf = aslist(self.coordinates)
        coordsOther = aslist(other.coordinates)

        intersection = [a for a in coordsSelf if a in coordsOther]
        complementLeft = [a for a in coordsSelf if a not in intersection]
        complementRight = [a for a in coordsOther if a not in intersection]
        hits = len(intersection)
        misses = len(complementLeft + complementRight)

        if method == 'support':
            if counts:
                return hits, misses
            else:
                return hits/float(hits+misses)

        if method == 'corr':
            from scipy.stats import spearmanr

            if not (hasattr(self, 'values') and hasattr(other, 'values')):
                raise Exception('Sources must have values to compute correlation')
            else:
                valuesSelf = aslist(self.values)
                valuesOther = aslist(other.values)
            if len(intersection) > 0:
                rho, _ = spearmanr(valuesSelf[intersection], valuesOther[intersection])
            else:
                rho = 0.0
            return (rho * hits)/float(hits + misses)

    def merge(self, other):
        """
        Combine this source with other
        """
        self.coordinates = concatenate((self.coordinates, other.coordinates))

        if hasattr(self, 'values'):
            self.values = concatenate((self.values, other.values))

        return self

    def tolist(self):
        """
        Convert array-like attributes to list
        """
        import copy
        new = copy.copy(self)
        for prop in ["coordinates", "values", "center", "bbox", "polygon"]:
            if prop in self.__dict__.keys():
                val = new.__getattribute__(prop)
                if val is not None and not isinstance(val, list):
                    setattr(new, prop, val.tolist())
        return new

    def toarray(self):
        """
        Convert array-like attributes to ndarray
        """
        import copy
        new = copy.copy(self)
        for prop in ["coordinates", "values", "center", "bbox", "polygon"]:
            if prop in self.__dict__.keys():
                val = new.__getattribute__(prop)
                if val is not None and not isinstance(val, ndarray):
                    setattr(new, prop, asarray(val))
        return new

    def transform(self, data, collect=True):
        """
        Extract series from data using a list of sources.

        Currently only supports averaging over coordinates.

        Params
        ------
        data : Images or Series object
            The data from which to extract

        collect : boolean, optional, default = True
            Whether to collect to local array or keep as a Series
        """

        if not (isinstance(data, Images) or isinstance(data, Series)):
            raise Exception("Input must either be Images or Series (or a subclass)")

        # TODO add support for weighting
        if isinstance(data, Images):
            output = data.meanByRegions([self.coordinates]).toSeries()
        else:
            output = data.meanOfRegion(self.coordinates)

        if collect:
            return output.collectValuesAsArray()
        else:
            return output

    def mask(self, dims=None, binary=True, outline=False, color=None):
        """
        Construct a mask from a source, either locally or within a larger image.

        Parameters
        ----------
        dims : list or tuple, optional, default = None
            Dimensions of large image in which to draw mask. If none, will restrict
            to the bounding box of the region.

        binary : boolean, optional, deafult = True
            Whether to incoporate values or only show a binary mask

        outline : boolean, optional, deafult = False
            Whether to only show outlines (derived using binary dilation)

        color : str or array-like
            RGB triplet (from 0 to 1) or named color (e.g. 'red', 'blue')
        """
        from thunder import Colorize

        coords = self.coordinates

        if dims is None:
            extent = self.bbox[len(self.center):] - self.bbox[0:len(self.center)] + 1
            m = zeros(extent)
            coords = (coords - self.bbox[0:len(self.center)])
        else:
            m = zeros(dims)

        if hasattr(self, 'values') and self.values is not None and binary is False:
            m[coords.T.tolist()] = self.values
        else:
            m[coords.T.tolist()] = 1

        if outline:
            from skimage.morphology import binary_dilation
            m = binary_dilation(m, ones((3, 3))) - m

        if color is not None:
            m = Colorize(cmap='indexed', colors=[color]).transform([m])

        return m

    def __repr__(self):
        s = self.__class__.__name__
        for opt in ["id", "center", "bbox"]:
            if hasattr(self, opt):
                o = self.__getattribute__(opt)
                os = o.tolist() if isinstance(o, ndarray) else o
                s += '\n%s: %s' % (opt, repr(os))
        return s


class SourceModel(Serializable, object):
    """
    A source model as a collection of extracted sources.

    Parameters
    ----------
    sources : list or Sources or a single Source
        The identified sources

    See also
    --------
    Source
    """
    def __init__(self, sources):
        if isinstance(sources, Source):
            self.sources = [sources]
        elif isinstance(sources, list) and isinstance(sources[0], Source):
            self.sources = sources
        elif isinstance(sources, list):
            self.sources = []
            for ss in sources:
                self.sources.append(Source(ss))
        else:
            raise Exception("Input type not recognized, must be Source, list of Sources, "
                            "or list of coordinates, got %s" % type(sources))

    def __getitem__(self, entry):
        if not isinstance(entry, int):
            raise IndexError("Selection not recognized, must be Int, got %s" % type(entry))
        return self.sources[entry]

    def combiner(self, prop, tolist=True):
        combined = []
        for s in self.sources:
            p = getattr(s, prop)
            if tolist:
                p = p.tolist()
            combined.append(p)
        return combined

    @property
    def coordinates(self):
        """
        List of coordinates combined across sources
        """
        return self.combiner('coordinates')

    @property
    def values(self):
        """
        List of coordinates combined across sources
        """
        return self.combiner('values')

    @property
    def centers(self):
        """
        Array of centers combined across sources
        """
        return asarray(self.combiner('center'))

    @property
    def polygons(self):
        """
        List of polygons combined across sources
        """
        return self.combiner('polygon')

    @property
    def areas(self):
        """
        List of areas combined across sources
        """
        return self.combiner('area', tolist=False)

    @property
    def count(self):
        """
        Number of sources
        """
        return len(self.sources)

    def masks(self, dims=None, binary=True, outline=False, base=None, color=None):
        """
        Composite masks combined across sources as an image.

        Parameters
        ----------
        dims : list or tuple, optional, default = None
            Dimensions of image in which to create masks, must either provide
            these or provide a base image

        binary : boolean, optional, deafult = True
            Whether to incoporate values or only show a binary mask

        outline : boolean, optional, deafult = False
            Whether to only show outlines (derived using binary dilation)

        base : SourceModel or array-like, optional, deafult = None
            Base background image on which to put masks,
            or another set of sources (usually for comparisons).
        """
        from thunder import Colorize
        from matplotlib.cm import get_cmap

        if dims is None and base is None:
            raise Exception("Must provide image dimensions for composite masks "
                            "or provide a base image.")

        if base is not None and isinstance(base, SourceModel):
            outline = True

        if dims is None and base is not None:
            dims = asarray(base).shape

        if isinstance(base, SourceModel):
            base = base.masks(dims, color='silver')

        if isinstance(base, ndarray):
            base = Colorize(cmap='indexed', colors=['white']).transform([base])

        if base is not None and color is None:
            color = 'deeppink'

        if color == 'random':
            combined = zeros(list(dims) + [3])
            ncolors = min(self.count, 20)
            colors = get_cmap('rainbow', ncolors)(range(0, ncolors, 1))[:, 0:3]
            for i, s in enumerate(self.sources):
                combined = maximum(s.mask(dims, binary, outline, colors[i % len(colors)]), combined)
        else:
            combined = zeros(dims)
            for s in self.sources:
                combined = maximum(s.mask(dims, binary, outline), combined)

        if color is not None and color != 'random':
            combined = Colorize(cmap='indexed', colors=[color]).transform([combined])

        if base is not None:
            combined = maximum(base, combined)

        return combined

    def match(self, other, unique=False, minDistance=inf):
        """
        For each source in self, find the index of the closest source in other.

        Uses euclidean distances between centers to determine distances.

        Can select nearest matches with or without enforcing uniqueness;
        if unique is False, will return the closest source in other for
        each source in self, possibly repeating sources multiple times
        if unique is True, will only allow each source in other to be matched
        with a single source in self, as determined by a greedy selection procedure.
        The minDistance parameter can be used to prevent far-away sources from being
        chosen during greedy selection.

        Params
        ------
        other : SourceModel
            The source model to match sources to

        unique : boolean, optional, deafult = True
            Whether to only return unique matches

        minDistance : scalar, optiona, default = inf
            Minimum distance to use when selecting matches
        """
        from scipy.spatial.distance import cdist

        targets = other.centers
        targetInds = range(0, len(targets))
        matches = []
        for s in self.sources:
            update = 1

            # skip if no targets left, otherwise update
            if len(targets) == 0:
                update = 0
            else:
                dists = cdist(targets, s.center[newaxis])
                if dists.min() < minDistance:
                    ind = argmin(dists)
                else:
                    update = 0

            # apply updates, otherwise add a nan
            if update == 1:
                matches.append(targetInds[ind])
                if unique is True:
                    targets = delete(targets, ind, axis=0)
                    targetInds = delete(targetInds, ind)
            else:
                matches.append(NaN)

        return matches

    def distance(self, other, minDistance=inf):
        """
        Compute the distance between each source in self and other.

        First estimates a matching source from other for each source
        in self, then computes the distance between the two sources.
        The matches are unique, using a greedy procedure,
        and minDistance can be used to prevent outliers during matching.

        Parameters
        ----------
        other : SourceModel
            The sources to compute distances to

        minDistance : scalar, optiona, default = inf
            Minimum distance to use when matching indices
        """

        inds = self.match(other, unique=True, minDistance=minDistance)
        d = []
        for jj, ii in enumerate(inds):
            if ii is not NaN:
                d.append(self[jj].distance(other[ii]))
            else:
                d.append(NaN)
        return asarray(d)

    def overlap(self, other, method='support', minDistance=inf):
        """
        Estimate overlap between sources in self and other.

        Will compute the similarity of sources in self that are found
        in other, based on either source pixel overlap or correlation.

        Parameters
        ----------
        other : SourceModel
            The sources to compare to

        method : str, optional, default = "support"
            Method to use when computing overlap between sources.
            Options include 'support' and 'correlation'

        minDistance : scalar, optional, default = inf
            Minimum distance to use when matching indices
        """

        inds = self.match(other, unique=True, minDistance=minDistance)
        d = []
        for jj, ii in enumerate(inds):
            if ii is not NaN:
                d.append(self[jj].overlap(other[ii], method=method))
            else:
                d.append(NaN)
        return asarray(d)

    def similarity(self, other, metric='distance', thresh=5, minDistance=inf):
        """
        Estimate similarity between sources in self and other.

        Will compute the fraction of sources in self that are found
        in other, based on a given distance metric and a threshold.
        The fraction is estimated as the number of sources in self
        found in other, divided by the total number of sources in self.

        Before computing metrics, all sources in self are matched to other,
        and a minimum distance can be set to control matching.

        Parameters
        ----------
        other : SourceModel
            The sources to compare to

        metric : str, optional, default = "distance"
            Metric to use when computing distances,
            options include 'distance' and 'overlap'

        thresh : scalar, optional, default = 5
            The distance below which a source is considered found

        minDistance : scalar, optional, default = inf
            Minimum distance to use when matching indices
        """
        checkParams(metric, ['distance', 'overlap'])

        if metric == 'distance':
            # when evaluating distances,
            # minimum distance should be the threshold
            if minDistance == inf:
                minDistance = thresh
            vals = self.distance(other, minDistance=minDistance)
            vals[isnan(vals)] = inf
        elif metric == 'overlap':
            vals = self.overlap(other, method='support', minDistance=minDistance)
            vals[isnan(vals)] = 0
        else:
            raise Exception("Metric not recognized")

        hits = sum(vals < thresh) / float(len(self.sources))

        return hits

    def transform(self, data, collect=True):
        """
        Extract series from data using a list of sources.

        Currently only supports simple averaging over coordinates.

        Params
        ------
        data : Images or Series object
            The data from which to extract signals

        collect : boolean, optional, default = True
            Whether to collect to local array or keep as a Series
        """
        if not (isinstance(data, Images) or isinstance(data, Series)):
            raise Exception("Input must either be Images or Series (or a subclass)")

        # TODO add support for weighting
        if isinstance(data, Images):
            output = data.meanByRegions(self.coordinates).toSeries()
        else:
            output = data.meanByRegions(self.coordinates)

        if collect:
            return output.collectValuesAsArray()
        else:
            return output

    def clean(self, cleaners=None):
        """
        Apply one or more cleaners to sources, returning filtered sources

        Parameters
        ----------
        cleaners : Cleaner or list of Cleaners, optional, default = None
            Which cleaners to apply, if None, will apply BasicCleaner with defaults
        """
        from thunder.extraction.cleaners import Cleaner, BasicCleaner
        from copy import copy

        if isinstance(cleaners, list):
            for c in cleaners:
                if not isinstance(c, Cleaner):
                    raise Exception("List must only contain Cleaners")
        elif isinstance(cleaners, Cleaner):
            cleaners = [cleaners]
        elif cleaners is None:
            cleaners = [BasicCleaner()]
        else:
            raise Exception("Must provide Cleaner or list of Cleaners, got %s" % type(cleaners))

        newmodel = copy(self)

        for c in cleaners:
            newmodel = c.clean(newmodel)

        return newmodel

    def save(self, f, numpyStorage='auto', include=None, **kwargs):
        """
        Custom save to file with simplified, human-readable output, and selection of lazy attributes.
        """
        import copy
        output = copy.deepcopy(self)
        if isinstance(include, str):
            include = [include]
        if include is not None:
            for prop in include:
                map(lambda s: getattr(s, prop), output.sources)
        output.sources = map(lambda s: s.restore(include).tolist(), output.sources)
        simplify = lambda d: d['sources']['py/homogeneousList']['data']
        super(SourceModel, output).save(f, numpyStorage='ascii', simplify=simplify, **kwargs)

    @classmethod
    def load(cls, f, **kwargs):
        """
        Custom load from file to handle simplified, human-readable output
        """
        unsimplify = lambda d: {'sources': {
            'py/homogeneousList': {'data': d, 'module': 'thunder.extraction.source', 'type': 'Source'}}}
        output = super(SourceModel, cls).load(f, unsimplify=unsimplify)
        output.sources = map(lambda s: s.toarray(), output.sources)
        return output

    @classmethod
    def deserialize(cls, d, **kwargs):
        """
        Custom load from JSON to handle simplified, human-readable output
        """
        unsimplify = lambda d: {'sources': {
            'py/homogeneousList': {'data': d, 'module': 'thunder.extraction.source', 'type': 'Source'}}}
        output = super(SourceModel, cls).deserialize(d, unsimplify=unsimplify)
        output.sources = map(lambda s: s.toarray(), output.sources)
        return output

    def __repr__(self):
        s = self.__class__.__name__
        s += '\n%g sources' % (len(self.sources))
        return s

LAZY_ATTRIBUTES = ["center", "polygon", "bbox", "area"]
