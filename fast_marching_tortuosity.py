import numpy as np
import heapq
from math import sqrt

def _neighbour_offsets(neighborhood, dx=1.0):
    if neighborhood == 6:
        return [((1,0,0), dx), ((-1,0,0), dx), ((0,1,0), dx), ((0,-1,0), dx), ((0,0,1), dx), ((0,0,-1), dx)]
    # 26-neighborhood: all non-zero combos in {-1,0,1}^3
    offs = []
    for di in (-1,0,1):
        for dj in (-1,0,1):
            for dk in (-1,0,1):
                if di==0 and dj==0 and dk==0:
                    continue
                dist = dx * sqrt(di*di + dj*dj + dk*dk)
                offs.append(((di,dj,dk), dist))
    return offs

def compute_tortuosity_fast_marching(pore, extraction, dx=1.0, neighborhood=6):
    """
    Compute geometrical tortuosity using a discrete fast-marching (Dijkstra) on a 3D grid.

    Method:
    - Treat `pore` as a boolean array (True=pore, False=solid).
    - Initialize all pore voxels on the x=0 face as sources (distance 0).
    - Run multi-source Dijkstra over pore voxels (neighbourhood 6 or 26).
    - For pore voxels on the x=NX-1 face (outlet) collect shortest distances.
    - Geometrical tortuosity tau = mean(distance_at_outlet) / L, where L = (NX-1)*dx.

    Returns: (tau, mean_path_length)
    """
    pore = np.asarray(pore, dtype=bool)
    if pore.ndim != 3:
        raise ValueError("pore must be a 3D array")
    Nx, Ny, Nz = pore.shape
    L = (Nx-1) * float(dx)

    # Prepare offsets
    offsets = _neighbour_offsets(neighborhood, dx=dx)

    # Distance array
    inf = float('inf')
    dist = np.full(pore.shape, inf, dtype=float)
    visited = np.zeros(pore.shape, dtype=bool)

    # Multi-source initialization: all pore voxels at x=0
    heap = []
    for j in range(Ny):
        for k in range(Nz):
            if pore[0, j, k]:
                dist[0, j, k] = 0.0
                heapq.heappush(heap, (0.0, 0, j, k))

    # Dijkstra loop
    while heap:
        d, i, j, k = heapq.heappop(heap)
        if visited[i, j, k]:
            continue
        visited[i, j, k] = True
        # relax neighbours
        for (di,dj,dk), cost in offsets:
            ni, nj, nk = i + di, j + dj, k + dk
            if not (0 <= ni < Nx and 0 <= nj < Ny and 0 <= nk < Nz):
                continue
            if not pore[ni, nj, nk]:
                continue
            if visited[ni, nj, nk]:
                continue
            nd = d + cost
            if nd < dist[ni, nj, nk]:
                dist[ni, nj, nk] = nd
                heapq.heappush(heap, (nd, ni, nj, nk))

    # Collect outlet distances (x = Nx-1)
    outlet_mask = pore[Nx-1, extraction[0]:extraction[1], extraction[2]:extraction[3]]
    outlet_dists = dist[Nx-1, extraction[0]:extraction[1], extraction[2]:extraction[3]][outlet_mask]

    if outlet_dists.size == 0:
        raise RuntimeError("No reachable outlet pores found. Check connectivity or BCs.")

    mean_path = float(np.mean(outlet_dists))
    tau = mean_path / L

    return tau

if __name__ == '__main__':
    # Quick self-test on a straight channel: tortuosity should be ~1.0
    nx, ny, nz = 50, 10, 10
    pore = np.zeros((nx, ny, nz), dtype=bool)
    pore[:, :, :] = False
    # straight open channel
    pore[:, 4:6, 4:6] = True
    tau = compute_tortuosity_fast_marching(pore, extraction=[0, ny, 0, nz], dx=1.0, neighborhood=6)
    print(f"tau={tau:.5f} (expected =1.0)")
