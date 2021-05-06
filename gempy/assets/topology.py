"""
    This file is part of gempy.

    gempy is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    gempy is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with gempy.  If not, see <http://www.gnu.org/licenses/>.


@author: Alexander Schaaf
"""
import numpy as np
from typing import List, Set, Tuple, Dict, Union, Optional
import matplotlib.pyplot as plt


def _get_nunconf(geo_model) -> int:
    return np.count_nonzero(
        geo_model._stack.df.BottomRelation == "Erosion"
    ) - 2  # TODO -2 n other lith series


def _get_nfaults(geo_model) -> int:
    return np.count_nonzero(geo_model._faults.df.isFault)


def _get_fb(geo_model) -> np.ndarray:
    n_unconf = _get_nunconf(geo_model)
    n_faults = _get_nfaults(geo_model)
    return np.round(
        geo_model.solutions.block_matrix[n_unconf:n_faults + n_unconf, 0, :]
    ).astype(int).sum(axis=0).reshape(*geo_model._grid.regular_grid.resolution)


def _get_lb(geo_model) -> np.ndarray:
    return np.round(
        geo_model.solutions.lith_block
    ).astype(int).reshape(*geo_model._grid.regular_grid.resolution)


def compute_topology(
        geo_model,
        cell_number: int = None,
        direction: str = None,
        n_shift: int = 1,
        voxel_threshold: int = 1
):
    """Compute topology of a GemPy model

    Args:
        geo_model (Project): GemPy model project
        cell_number (int, optional): Cell number in chosen direction. Defaults to None.
        direction (str, optional): one of the model's dimensions, x, y, or z. Defaults to None.
        n_shift (int, optional): number of voxels the model is shifted and then substracted for finding interfaces. Defaults to 1.
        voxel_threshold (int, optional): amount of voxels which have to be connected to be considered into the topology calculation. Defaults to 1.

    Returns:
        edges, centroids [numpy array]: edges and centroids of the topology graph
    """
    res = geo_model._grid.regular_grid.resolution
    fb = _get_fb(geo_model)
    lb = _get_lb(geo_model)
    n_lith = len(np.unique(lb))  # ? quicker looking it up in geomodel?

    if cell_number is None or direction is None:
        direction = "None"
    elif direction.capitalize() == "X":
        lb = lb[cell_number, :, :]
        fb = fb[cell_number, :, :]
        res = (1, res[1], res[2])
    elif direction.capitalize() == "Y":
        lb = lb[:, cell_number, :]
        fb = fb[:, cell_number, :]
        res = (res[0], 1, res[2])
    elif direction.capitalize() == "Z":
        lb = lb[:, :, cell_number]
        fb = fb[:, :, cell_number]
        res = (res[0], res[1], 1)

    edges, centroids = _analyze_topology(
        fb.ravel(),
        lb.ravel(),
        n_lith,
        res,
        n_shift,
        voxel_threshold,
        direction
    )

    edges = set((n1, n2) for n1, n2 in edges)
    edges = _filter_reverse_edges(edges)  # ? still necessary? would next line
    # ? not be enough?
    edges = _sort_edge_tuple_nodes(edges)
    return edges, centroids


def _filter_reverse_edges(edges: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    """Filter reversed topology edge tuples to fix doubling of topology edges
    like (1,5) (5,1).
    
    Source:
        https://stackoverflow.com/a/9922322/8040299

    Args:
        edges (Set[Tuple[int, int]]): Set of topologyedge tuples.
    
    Returns:
        Set[Tuple[int, int]]: Filtered set of topology edge tuples
    """

    edges_unique = set()
    for e in edges:
        if not (e in edges_unique or (e[1], e[0]) in edges_unique):
            edges_unique.add(e)
    return edges_unique


def _sort_edge_tuple_nodes(edges: Set[Tuple[int, int]]) -> Set[Tuple[int, int]]:
    """Sort nodes within edge tuples by ascending order.
    
    Args:
        edges (Set[Tuple[int, int]]): Set of edge tuples.
    
    Returns:
        Set[Tuple[int, int]]: Set of sorted edge tuples.
    """
    sorted_edges = set()
    for n1, n2 in edges:
        if n1 > n2:
            sorted_edges.add((int(n2), int(n1)))
        else:
            sorted_edges.add((int(n1), int(n2)))
    return sorted_edges


def _analyze_topology(
        fault_matrix_sum,
        lith_matrix,
        n_lith,
        res,
        n_shift,
        voxel_threshold,
        direction
):
    fault_shift = fault_matrix_sum.min()
    fault_matrix_sum_shift = fault_matrix_sum - fault_shift

    where = np.tile(lith_matrix, (n_lith, 1)) == np.unique(lith_matrix).reshape(
        -1, 1)
    lith_matrix_shift = np.sum(where * np.arange(n_lith).reshape(-1, 1),
                               axis=0) + 1

    topo_matrix = lith_matrix_shift + n_lith * fault_matrix_sum_shift
    topo_matrix_3D = topo_matrix.reshape(*res)

    # x direction
    if direction.capitalize() != "X":
        x_l = topo_matrix_3D[n_shift:, :, :]
        x_r = topo_matrix_3D[:-n_shift, :, :]
        x_edges, x_count = _get_edges(x_l, x_r)
        x_edges = x_edges[:, x_count > voxel_threshold]
    else:
        x_edges = np.array([[], []])

    # y direction
    if direction.capitalize() != "Y":
        y_l = topo_matrix_3D[:, n_shift:, :]
        y_r = topo_matrix_3D[:, :-n_shift, :]
        y_edges, y_count = _get_edges(y_l, y_r)
        y_edges = y_edges[:, y_count > voxel_threshold]
    else:
        y_edges = np.array([[], []])

    # z direction
    if direction.capitalize() != "Z":
        z_l = topo_matrix_3D[:, :, n_shift:]
        z_r = topo_matrix_3D[:, :, :-n_shift]
        z_edges, z_count = _get_edges(z_l, z_r)
        z_edges = z_edges[:, z_count > voxel_threshold]
    else:
        z_edges = np.array([[], []])

    edges = np.unique(
        np.concatenate((x_edges.T, y_edges.T, z_edges.T), axis=0), axis=0
    )

    centroids = _get_centroids(topo_matrix_3D)

    return edges, centroids


def get_lot_node_to_lith_id(
        geo_model,
        centroids: Dict[int, np.ndarray]
) -> Dict[int, int]:
    """Get look-up table to translate topology node id's back into GemPy lith
    id's.
    
    Args:
        geo_model: GemPy geomodel instance with solutions.
        centroids (Dict[int, Array[float, 3]]): Topology node centroids.
    
    Returns:
        Dict[int, int]: Look-up table translating node id -> lith id.
    """
    lb = geo_model.solutions.lith_block.reshape(
        geo_model._grid.regular_grid.resolution
    ).astype(int)

    lot = {}
    for node, pos in centroids.items():
        p = np.round(pos).astype(int)
        lith_id = lb[p[0], p[1], p[2]]
        lot[node] = lith_id
    return lot


def get_lot_lith_to_node_id(
        lot: Dict[int, np.ndarray]
) -> Dict[int, List[int]]:
    """Get look-up table to translate lith id's back into topology node
    id's.
    
    Args:
        lot (Dict[int, Array[float, 3]]): Node to lith id look-up table. Can be
        computed using the function 'get_lot_node_to_lith_id'.
    
    Returns:
        Dict[int, List[int]]: Look-up table.
    """
    lot2 = {}
    for k, v in lot.items():
        if v not in lot2.keys():
            lot2[v] = [k]
        else:
            lot2[v].append(k)
    return lot2


def get_lot_node_to_fault_block(
        geo_model,
        centroids: Dict[int, np.ndarray]
) -> Dict[int, int]:
    """Get a look-up table to access fault block id's for each topology node
    id.
    
    Args:
        geo_model: Geomodel instance.
        centroids (Dict[int, Array[float, 3]]): Geomodel topology centroids.
    
    Returns:
        Dict[int, int]: Look-up table.
    """
    n_lith = len(get_lith_ids(geo_model))
    lot = {}
    for node, _ in centroids.items():
        lot[node] = (node - 0 - (node // n_lith)) // n_lith
    return lot


def get_fault_ids(geo_model) -> List[int]:
    """Get fault id's of all faults in given geomodel.
    
    Args:
        geo_model: Geomodel instance
    
    Returns:
        List[int]: List of fault id's.
    """
    f_series_names = geo_model._faults.df[geo_model._faults.df.isFault].index
    fault_ids = [0]
    for fsn in f_series_names:
        fid = geo_model._surfaces.df[
                geo_model._surfaces.df.series == fsn].id.values[0]
        fault_ids.append(fid)
    return fault_ids


def get_lith_ids(geo_model, basement: bool = True) -> List[int]:
    """ Get lithology id's of all lithologies (except basement) in given
     geomodel.
    
    Args:
        geo_model: Geomodel instance.
    
    Returns:
        List[int]: List of lithology id's.
    """
    fmt_series_names = geo_model._faults.df[~geo_model._faults.df.isFault].index
    lith_ids = []
    for fsn in fmt_series_names:
        if not basement:
            if fsn == "Basement":
                continue
        lids = geo_model._surfaces.df[
            geo_model._surfaces.df.series == fsn].id.values
        for lid in lids:
            lith_ids.append(lid)
    return lith_ids


def get_detailed_labels(
        geo_model,
        edges: Set[Tuple[int, int]],
        centroids: Dict[int, np.ndarray]
) -> Tuple[Set[Tuple[str, str]], Dict[str, np.ndarray]]:
    """Convert given edges and centroids data into more detailed labels with
     pattern 'lithid_faultid'.
    
    Args:
        geo_model: [description]
        edges (Set[Tuple[int, int]]): Set of geomodel topology edges.
        centroids (Dict[int, Array[float, 3]]): Geomodel topology centroids.
    
    Returns:
        Tuple[Set[Tuple[str, str]], Dict[str, Array[float, 3]]]: Re-labeled
            edges and centroids.
    """
    lot_lith = get_lot_node_to_lith_id(geo_model, centroids)
    lot_fault = get_lot_node_to_fault_block(geo_model, centroids)

    centroids_ = {}
    for node, pos in centroids.items():
        n = lot_lith.get(node), lot_fault.get(node)
        n = str(n[0]) + "_" + str(n[1])
        centroids_[n] = pos

    edges_ = set()
    for n1, n2 in edges:
        edges_.add(
            (
                str(lot_lith.get(n1)) + "_" + str(lot_fault.get(n1)),
                str(lot_lith.get(n2)) + "_" + str(lot_fault.get(n2))
            )
        )
    return edges_, centroids_


def _get_edges(
        l: np.ndarray,
        r: np.ndarray
) -> Optional[np.ndarray]:
    """Get edges from given shifted arrays.

    Args:
        l (Array): Topology labels array shifted to one direction.
        r (Array): Topology labels array shifted to the other direction.

    Returns:
        Array: Topology edges.
    """
    shift = np.stack([l.ravel(), r.ravel()])
    i1, i2 = np.nonzero(np.diff(shift, axis=0))
    if len(i2) == 0:  # in case not edges are found (symmetric model along axis)
        return np.array([[], []]), np.array([])
    else:
        return np.unique(shift[:, i2], axis=1, return_counts=True)


def clean_unconformity_topology(
        geo_model,
        unconf_lith_id: int,
        edges: np.ndarray,
        centroids: Dict[int, np.ndarray]
) -> Tuple[Set[Tuple[int, int]], Dict[int, np.ndarray]]:
    """Clean unconformity topology edges and centroids. Needs to be run for
    each unconformity separately.
    
    Args:
        geo_model ([type]): [description]
        unconf_lith_id (int): [description]
        edges (Array[int, ..., 2]): [description]
        centroids (Dict[int, Array[int, ..., 3]]): [description]
    
    Returns:
        Tuple[Set[Tuple[int, int]], Dict[int, Array[int, ..., 3]]]: [description]
    """
    lot = get_lot_node_to_lith_id(geo_model, centroids)
    lot2 = get_lot_lith_to_node_id(lot)
    edges_clean = []

    unconf_centroid = np.mean(
        [centroids[i] for i in lot2[unconf_lith_id]], axis=0
    )
    for node_id in lot2[unconf_lith_id]:
        centroids.pop(node_id)
    centroids[lot2[unconf_lith_id][0]] = unconf_centroid

    for n1, n2 in edges:
        if n1 in lot2[unconf_lith_id]:
            n1 = lot2[unconf_lith_id][0]
        if n2 in lot2[unconf_lith_id]:
            n2 = lot2[unconf_lith_id][0]
        if n1 == n2:
            continue
        edges_clean.append((n1, n2))

    return set(edges_clean), centroids


def jaccard_index(
        edges1: Set[Tuple[int, int]],
        edges2: Set[Tuple[int, int]]
) -> float:
    """Jaccard index.
    
    Args:
        edges1 (Set[Tuple[int, int]]): Set of topology edges.
        edges2 (Set[Tuple[int, int]]): Set of topology edges.
    
    Returns:
        float: Jaccard index.
    """
    intersection_cardinality = len(edges1.intersection(edges2))
    union_cardinality = len(edges1.union(edges2))
    return intersection_cardinality / union_cardinality


def _get_centroids(labels: np.ndarray) -> dict:
    """Get geobody node centroids in array coordinates.
    
    Args:
        labels (Array[int, ..., ..., ...]): Uniquely labeled block.
    
    Returns:
        dict: Geobody node keys yield centroid coordinate tuples in array
            coordinates.
    """
    node_locs = []
    ulabels = np.unique(labels)
    for node in ulabels:
        node_pos = np.argwhere(labels == node)
        node_locs.append(node_pos.mean(axis=0))
    centroids = {n: loc for n, loc in zip(ulabels, node_locs)}
    # for k, v in centroids.items():
    # debug(f"{k}: {v}")
    return centroids


def get_adjacency_matrix(
        geo_model,
        edges: Set[Tuple[int, int]],
        centroids,
) -> np.ndarray:
    """[summary]
    
    Args:
        geo_model ([type]): [description]
        edges (Set[Tuple): [description]
    
    Returns:
        Array[bool, ..., ...]: [description]
    """
    f_ids = get_fault_ids(geo_model)
    lith_ids = get_lith_ids(geo_model)
    n = len([(l, f) for f in f_ids for l in lith_ids])

    M = np.zeros((n, n))
    lith_lot = get_lot_node_to_lith_id(geo_model, centroids)
    fault_lot = get_lot_node_to_fault_block(geo_model, centroids)
    for e1, e2 in edges:
        #     print("nodes:", e1, e2)
        l1, l2 = lith_lot.get(e1), lith_lot.get(e2)
        #     print("lith: ", l1, l2)
        f1, f2 = fault_lot.get(e1), fault_lot.get(e2)
        #     print("fault:", f1, f2)
        lp1 = np.argwhere(lith_ids == l1)[0, 0]
        lp2 = np.argwhere(lith_ids == l2)[0, 0]
        #     print("lpos :", lp1, lp2)
        p1 = lp1 + len(lith_ids) * f1
        p2 = lp2 + len(lith_ids) * f2
        #     print("pos  :", p1, p2)
        M[p1, p2] = 1
        M[p2, p1] = 1

    M = np.flip(np.flip(M, axis=1), axis=0)
    return M.astype(bool)


def _get_adj_matrix_labels(geo_model):
    f_ids = get_fault_ids(geo_model)
    lith_ids = get_lith_ids(geo_model)
    adj_matrix_labels = [(l, f) for f in f_ids for l in lith_ids]
    adj_matrix_lith_labels = [l for f in f_ids for l in lith_ids]
    adj_matrix_fault_labels = [f for f in f_ids for l in lith_ids]
    return adj_matrix_labels, adj_matrix_lith_labels, adj_matrix_fault_labels


def plot_adjacency_matrix(
        geo_model,
        adj_matrix: np.ndarray
):
    """

    Args:
        geo_model:
        adj_matrix: Array[bool, ..., ...]

    Returns:

    """
    f_ids = get_fault_ids(geo_model)
    n_faults = len(f_ids) // 2
    lith_ids = get_lith_ids(geo_model)
    n_liths = len(lith_ids)
    adj_matrix_labels, adj_matrix_lith_labels, adj_matrix_fault_labels = _get_adj_matrix_labels(
        geo_model)
    # ///////////////////////////////////////////////////////
    n = len(adj_matrix_labels)
    fig, ax = plt.subplots(figsize=(n // 2.5, n // 2.5))

    ax.imshow(adj_matrix, cmap="Greys", alpha=1)
    ax.set_xlim(-.5, n_liths * n_faults * 2 - 0.5)
    ax.set_ylim(-.5, n_liths * n_faults * 2 - 0.5)

    ax.set_title("Topology Adjacency Matrix")

    # ///////////////////////////////////////////////////////
    # lith tick labels
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(adj_matrix_lith_labels[::1], rotation=0)
    ax.set_yticklabels(adj_matrix_lith_labels[::1], rotation=0)

    # ///////////////////////////////////////////////////////
    # lith tick labels colors
    colors = list(geo_model._surfaces.colors.colordict.values())
    bboxkwargs = dict(
        edgecolor='none',
    )
    for xticklabel, yticklabel, l in zip(ax.xaxis.get_ticklabels(),
                                         ax.yaxis.get_ticklabels(),
                                         adj_matrix_labels[::1]):
        color = colors[l[0] - 1]

        xticklabel.set_bbox(
            dict(facecolor=color, **bboxkwargs)
        )
        xticklabel.set_color("white")

        yticklabel.set_bbox(
            dict(facecolor=color, **bboxkwargs)
        )
        yticklabel.set_color("white")

    # ///////////////////////////////////////////////////////
    # fault block tick labeling
    newax = fig.add_axes(ax.get_position())
    newax.patch.set_visible(False)

    newax.spines['bottom'].set_position(('outward', 29))
    newax.set_xlim(0, n_faults * 2)

    newax.set_xticks(np.arange(1, n_faults * 2 + 1) - 0.5)
    newax.set_xticklabels(["FB " + str(i + 1) for i in range(n_faults * 2)])

    newax.spines['left'].set_position(('outward', 25))
    newax.set_ylim(0, n_faults * 2)
    newax.set_yticks(np.arange(1, n_faults * 2 + 1) - 0.5)
    newax.set_yticklabels(
        ["FB " + str(i + 1) for i in range(n_faults * 2)][::1])

    # ///////////////////////////////////////////////////////
    # (dotted) lines for fb's
    dlinekwargs = dict(
        color="black",
        linestyle="dashed",
        alpha=0.75,
        linewidth=1
    )
    linekwargs = dict(
        color="black",
        linewidth=1
    )
    for i in range(0, n_faults * 2 + 1):
        pos = i * n_liths - .5

        if i != 0 and i != n_faults * 2:
            ax.axvline(pos, **dlinekwargs)
            ax.axhline(pos, **dlinekwargs)

        # solid spines outside to separate fbs
        line = ax.plot((-3.3, -.51), (pos, pos), **linekwargs)
        line[0].set_clip_on(False)

        line = ax.plot((pos, pos), (-3, -.51), **linekwargs)
        line[0].set_clip_on(False)
    # ///////////////////////////////////////////////////////
    return


def check_adjacency(
        edges: set,
        n1: Union[int, str],
        n2: Union[int, str]
) -> bool:
    """Check if given nodes n1 and n2 are adjacent in given topology
    edge set.
    
    Args:
        edges (set): Topology edges.
        n1 (Union[int, str]): Node 1 label.
        n2 (Union[int, str]): Node 2 label
    
    Returns:
        bool: True if adjacent, otherwise False.
    """
    if (n1, n2) in edges or (n2, n1) in edges:
        return True
    else:
        return False


def get_adjacencies(
        edges: set,
        node: Union[int, str]
) -> set:
    """Get node labels of all adjacent geobodies of geobody with given node
     in given set of edges.
    
    Args:
        edges (set): Topology edges.
        node (Union[int, str]): Node label.
    
    Returns:
        set: All adjacent geobody node labels.
    """
    adjacencies = set()
    for n1, n2 in edges:
        if node == n1:
            adjacencies.add(n2)
        elif node == n2:
            adjacencies.add(n1)
    return adjacencies


def count_unique_topologies(edges: List[Set[Tuple[int, int]]]):
    """Count unique topologie graphs in given list of edge sets.

    Args:
        edges: List of topology edge sets.
            E.g. [{(0,1), (0,2), ...}, {(0,1), (0,2), ...}]

    Returns:
        unique edges
        unique edges count
        unique edges idx
    """
    unique_edges = [edges[0]]
    unique_edges_count = [1]
    unqiue_edges_idx = [0]
    for _, topology in enumerate(edges[1:]):
        skip = False
        for b, utopology in enumerate(unique_edges):
            if utopology == topology:
                unique_edges_count[b] += 1
                unqiue_edges_idx.append(b)
                skip = True
                break
        if skip:
            continue
        unique_edges.append(topology)
        unique_edges_count.append(1)
        unqiue_edges_idx.append(len(unique_edges))

    return unique_edges, np.array(unique_edges_count), np.array(
        unqiue_edges_idx)
