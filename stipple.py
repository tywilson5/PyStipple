import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from scipy.spatial import Delaunay
from UnionFind import *
from linkedlist import *


def read_image(path):
    """
    A wrapper around matplotlib's image loader that deals with
    images that are grayscale or which have an alpha channel

    Parameters
    ----------
    path: string
        Path to file

    Returns
    -------
    ndarray(M, N, 3)
        An RGB color image in the range [0, 1]
    """
    img = plt.imread(path)
    if np.issubdtype(img.dtype, np.integer):
        img = np.array(img, dtype=float)/255
    if len(img.shape) == 3:
        if img.shape[1] > 3:
            # Cut off alpha channel
            img = img[:, :, 0:3]
    if img.size == img.shape[0]*img.shape[1]:
        # Grayscale, convert to rgb
        img = np.concatenate(
            (img[:, :, None], img[:, :, None], img[:, :, None]), axis=2)
    return img


def get_weights(I, thresh, p=1, canny_sigma=0):
    """
    Create pre-pixel weights based on image brightness

    Parameters
    ----------
    I: ndarray(M, N)
        Grayscale image
    thresh: float
        Amount above which to make a point 1
    p: float
        Contrast boost, apply weights^(1/p)

    Returns
    -------
    ndarray(M, N)
        The weights of each pixel, in the range [0, 1]
    """
    weights = np.array(I)
    if np.max(weights) > 1:
        weights /= 255
    weights = np.minimum(weights, thresh)
    weights -= np.min(weights)
    weights /= np.max(weights)
    weights = 1-weights
    weights = weights**(1/p)
    if canny_sigma > 0:
        from skimage import feature
        edges = feature.canny(I, sigma=canny_sigma)
        weights[edges > 0] = 1
    return weights


def stochastic_universal_sample(weights, target_points, jitter=0.1):
    """
    Sample pixels according to a particular density using 
    stochastic universal sampling

    Parameters
    ----------
    ndarray(M, N)
        The weights of each pixel, in the range [0, 1]
    target_points: int
        The number of desired samples
    jitter: float
        Perform a jitter with this standard deviation of a pixel

    Returns
    -------
    ndarray(N, 2)
        Location of point samples
    """
    choices = np.zeros(target_points, dtype=int)
    w = np.zeros(weights.size+1)
    order = np.random.permutation(weights.size)
    w[1::] = weights.flatten()[order]
    w = w/np.sum(w)
    w = np.cumsum(w)
    p = np.random.rand()  # Cumulative probability index, start off random
    idx = 0
    for i in range(target_points):
        while idx < weights.size and not (p >= w[idx] and p < w[idx+1]):
            idx += 1
        idx = idx % weights.size
        choices[i] = order[idx]
        p = (p + 1/target_points) % 1
    X = np.array(list(np.unravel_index(choices, weights.shape)), dtype=float).T
    if jitter > 0:
        X += jitter*np.random.randn(X.shape[0], 2)
    return X


@jit(nopython=True)
def get_centroids(mask, N, weights):
    """
    Return the weighted centroids in a mask
    """
    nums = np.zeros((N, 2))
    denoms = np.zeros(N)
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            idx = int(mask[i, j])
            weight = weights[i, j]
            nums[idx, 0] += weight*i
            nums[idx, 1] += weight*j
            denoms[idx] += weight
    nums = nums[denoms > 0, :]
    denoms = denoms[denoms > 0]
    return nums, denoms


def voronoi_stipple(I, thresh, target_points, p=1, canny_sigma=0, n_iters=10, do_plot=False):
    """
    An implementation of the method of [2]

    [2] Adrian Secord. Weighted Voronoi Stippling

    Parameters
    ----------
    I: ndarray(M, N, 3)
        An RGB/RGBA or grayscale image
    thresh: float
        Amount above which to make a point 1
    p: float
        Contrast boost, apply weights^(1/p)
    canny_sigma: float
        If >0, use a canny edge detector with this standard deviation
    n_iters: int
        Number of iterations
    do_plot: bool
        Whether to plot each iteration

    Returns
    -------
    ndarray(N, 2)
        An array of the stipple pattern, with x coordinates along the first
        column and y coordinates along the second column
    """
    from scipy.ndimage import distance_transform_edt
    import time
    if np.max(I) > 1:
        I = I/255
    if len(I.shape) > 2:
        I = 0.2125*I[:, :, 0] + 0.7154*I[:, :, 1] + 0.0721*I[:, :, 2]
    # Step 1: Get weights and initialize random point distributin
    # via rejection sampling
    weights = get_weights(I, thresh, p, canny_sigma)
    X = stochastic_universal_sample(weights, target_points)
    X = np.array(np.round(X), dtype=int)
    X[X[:, 0] >= weights.shape[0], 0] = weights.shape[0]-1
    X[X[:, 1] >= weights.shape[1], 1] = weights.shape[1]-1

    if do_plot:
        plt.figure(figsize=(10, 10))
    for it in range(n_iters):
        if do_plot:
            plt.clf()
            plt.scatter(X[:, 1], X[:, 0], 4)
            plt.gca().invert_yaxis()
            plt.xlim([0, weights.shape[1]])
            plt.ylim([weights.shape[0], 0])
            plt.savefig("Voronoi{}.png".format(it), facecolor='white')

        mask = np.ones_like(weights)
        X = np.array(np.round(X), dtype=int)
        mask[X[:, 0], X[:, 1]] = 0

        _, inds = distance_transform_edt(mask, return_indices=True)
        ind2num = {}
        for i in range(I.shape[0]):
            for j in range(I.shape[1]):
                coord = (inds[0, i, j], inds[1, i, j])
                if not coord in ind2num:
                    ind2num[coord] = len(ind2num)
        for i in range(I.shape[0]):
            for j in range(I.shape[1]):
                coord = (inds[0, i, j], inds[1, i, j])
                mask[i, j] = ind2num[coord]
        nums, denoms = get_centroids(mask, len(ind2num), weights)
        X = nums/denoms[:, None]
    X[:, 0] = I.shape[0]-X[:, 0]
    return np.fliplr(X)


def density_filter(X, fac, k=1):
    """
    Filter out points below a certain density

    Parameters
    ----------
    X: ndarray(N, 2)
        Point cloud
    fac: float
        Percentile (between 0 and 1) of points to keep, by density
    k: int
        How many neighbors to consider

    Returns
    -------
    ndarray(N)
        Distance of nearest point
    """
    from scipy.spatial import KDTree
    tree = KDTree(X)
    dd, _ = tree.query(X, k=k+1)
    dd = np.mean(dd[:, 1::], axis=1)
    q = np.quantile(dd, fac)
    return X[dd < q, :]


UNTOUCHED = 0
FRONTIER = 1
VISITED = 2


class Vertex:
    def __init__(self, label):
        self.label = label
        self.neighbs = set([])
        self.state = UNTOUCHED
        self.neighbs_dist = {}

    def add_neighbor(self, neighbor):
        self.neighbs.add(neighbor)
        self.neighbs_dist[neighbor.label] = ((self.data['x'] - neighbor.data['x']) ** 2 + (self.data['y'] - neighbor.data['y']) ** 2)
        



class Graph:
    def __init__(self):
        # Key: Vertex to look up
        # Value is the object encapsulating
        # information about that vertex
        self.vertices = {}

    def add_vertex(self, u):
        self.vertices[u] = Vertex(u)

    def add_edge(self, u, v):
        self.vertices[u].neighbs.add(self.vertices[v])
        self.vertices[v].neighbs.add(self.vertices[u])


def delaunay_helper(X):
    """
    Helper function for computing the Delaunay triangulation

    Parameters
    ----------
        X : Stipple pattern of an image

    Returns
    -------
        vertices (list): Vertices of the Delaunay triangulation
        tri : Triangulation of the stipple pattern
    """
    x, y = X[:, 0], X[:, 1]
    vertices = []

    for i in range(X.shape[0]):
        vertex = Vertex(i)
        vertex.data = {'x': x[i], 'y': y[i]}
        vertices.append(vertex)

    tri = Delaunay(X)
    return vertices, tri


def compute_delaunay(X):
    """
    Compute the Delaunay triangulation

    Parameters
    ----------
        X : Stipple pattern of an image

    Returns
    -------
         vertices (list): Vertices of the Delaunay triangulation
         edges (list): Edges of the Delaunay triangulation
    """
    x, y = X[:, 0], X[:, 1]
    vertices, tri = delaunay_helper(X)
    edges = set()

    for triangle in tri.simplices:
        for k in range(3):
            i1, i2 = triangle[k], triangle[(k+1) % 3]
            d = ((x[i1] - x[i2]) ** 2 + (y[i1] - y[i2]) ** 2)
            edges.add((i1, i2, d))

    return vertices, list(edges)


def compute_tour(X):
    """
    Compute the tour of the Delaunay triangulation using DFS

    Parameters
    ----------
        X : Stipple pattern of an image

    Returns
    -------
        tour (list): Tour of the Delaunay triangulation using DFS
    """
    vertices, edges = compute_delaunay(X)
    edges = get_mst_kruskal(vertices, edges)
    tour = []
    frontier = DoublyLinkedList()
    frontier.add_first(vertices[0])

    while len(frontier) > 0:
        vertex = frontier.remove_last()
        vertex.state = VISITED
        tour.append(vertex.label)

        for n in vertex.neighbs:
            if n.state != FRONTIER and n.state != VISITED:
                n.state = FRONTIER
                frontier.add_last(n)
    
    tour = two_opt_heuristic(vertices, tour)

    return tour

def get_mst_kruskal(vertices, edges):
    """
    Compute the minimum spanning tree of the Delaunay triangulation using Kruskal's algorithm

    Parameters
    ----------
        vertices (list): Vertices of the Delaunay triangulation
         edges (list): Edges of the Delaunay triangulation

    Returns
    -------
        new_edges (list): Edges of the minimum spanning tree of the Delaunay triangulation
    """
    edges = sorted(edges, key=dist_of_edge)
    djset = UFFast(len(vertices))
    new_edges = []

    for e in edges:
        (i, j, d) = e
        if not djset.find(i, j):
            
            #if e[3] > 
            djset.union(i, j)
            #print(vertices[i], vertices[j].label)
            vertices[i].add_neighbor(vertices[j])
            vertices[j].add_neighbor(vertices[i])
            new_edges.append(e)

    return new_edges



def two_opt_heuristic(vertices, tour):
    """
    Two-opt heuristic for improving the tour of the Delaunay triangulation

    Parameters
    ----------
        vertices (list): Vertices of the Delaunay triangulation
        tour (list): Tour of the Delaunay triangulation using DFS

    Returns
    -------
        tour (list): Improved tour of the Delaunay triangulation after two-opt heuristic
    """
    improvement = True
    while improvement:
        improvement = False
        for i in range(1, len(tour) - 2):
            for j in range(i + 1, len(tour)-1):
                if calculate_new_distance(tour, i, j, vertices) < calculate_current_distance(tour, i, j, vertices):
                    tour = swap_edges(tour, i, j)
                    improvement = True           # this is alittle janky, but it works for continuous improvement
    return tour


def calculate_new_distance(tour, i, j, vertices):
    """
    Calculate the new distance after swapping edges
    Helper function for two_opt_heuristic

    Parameters
    ----------
        tour (list): Tour of the Delaunay triangulation using DFS
        i (int): Index of the first vertex
        j (int): Index of the second vertex
        vertices (list): Vertices of the Delaunay triangulation

    Returns
    -------
        the new distance after swapping edges
    """
    if vertices[tour[i]].neighbs_dist.get(vertices[tour[j]].label) is None:
        vertices[tour[i]].add_neighbor(vertices[tour[j]])
        vertices[tour[j]].add_neighbor(vertices[tour[i]])
        
    if vertices[tour[j]].neighbs_dist.get(vertices[tour[i]].label) is None:
        vertices[tour[i]].add_neighbor(vertices[tour[j]])
        vertices[tour[j]].add_neighbor(vertices[tour[i]])
        
    if vertices[tour[i + 1]].neighbs_dist.get(vertices[tour[j + 1]].label) is None:
        vertices[tour[i+1]].add_neighbor(vertices[tour[j+1]])
        vertices[tour[j+1]].add_neighbor(vertices[tour[i+1]])
       
    if vertices[tour[j + 1]].neighbs_dist.get(vertices[tour[i + 1]].label) is None:
        vertices[tour[i+1]].add_neighbor(vertices[tour[j+1]])
        vertices[tour[j+1]].add_neighbor(vertices[tour[i+1]])
        
    return vertices[tour[i]].neighbs_dist.get(vertices[tour[j]].label) + vertices[tour[i + 1]].neighbs_dist.get(vertices[tour[j + 1]].label)



def calculate_current_distance(tour, i, j, vertices):
    """
    Calculate the current distance
    Helper function for two_opt_heuristic
    
    Parameters
    ----------
        tour (list): Tour of the Delaunay triangulation using DFS
        i (int): Index of the first vertex
        j (int): Index of the second vertex
        vertices (list): Vertices of the Delaunay triangulation
        
    Returns
    -------
        the current distance    
    """
    if vertices[tour[i]].neighbs_dist.get(vertices[tour[i + 1]].label) is None:
        vertices[tour[i]].add_neighbor(vertices[tour[i + 1]])
        vertices[tour[i + 1]].add_neighbor(vertices[tour[i]])
        
    if vertices[tour[j]].neighbs_dist.get(vertices[tour[j + 1]].label) is None:
        vertices[tour[j]].add_neighbor(vertices[tour[j + 1]])
        vertices[tour[j + 1]].add_neighbor(vertices[tour[j]])
        
    if vertices[tour[i]].neighbs_dist.get(vertices[tour[j]].label) is None:
        vertices[tour[i]].add_neighbor(vertices[tour[j]])
        vertices[tour[j]].add_neighbor(vertices[tour[i]])
        
    if vertices[tour[i + 1]].neighbs_dist.get(vertices[tour[j + 1]].label) is None:
        vertices[tour[i + 1]].add_neighbor(vertices[tour[j + 1]])
        vertices[tour[j + 1]].add_neighbor(vertices[tour[i + 1]])
        
    
    return vertices[tour[i]].neighbs_dist.get(vertices[tour[i + 1]].label) + vertices[tour[j]].neighbs_dist.get(vertices[tour[j + 1]].label)


def swap_edges(tour, i, j):
    """
    Swap the edges of the tour 
    Helper function for two_opt_heuristic
    
    Parameters
    ----------
        tour (list): Tour of the Delaunay triangulation using DFS
        i (int): Index of the first vertex
        j (int): Index of the second vertex
    
    Returns
    -------
        new_tour (list): New tour after swapping edges    
    """
    new_tour = tour[:i+1]                       # from idx 0 to i in original tour
    new_tour.extend(reversed(tour[i+1:j+1]))    # from idx i+1 to j+1 in reversed order
    new_tour.extend(tour[j+1:])                 # from idx j+1 to the end in original tour
    return new_tour





def dist_of_edge(e):
    """
    Get the distance of an edge, Helper function for get_mst_kruskal

    Parameters
    ----------
        e: Edge of the Delaunay triangulation

    Returns
    -------
        Distance of the edge
    """
    return e[2]
