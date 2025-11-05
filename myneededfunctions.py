# We import all the necessary functions for the project in this file

import numpy as np
import math
from multiprocessing.sharedctypes import RawArray
import ctypes


_IN = _OUT = _K = _SHAPE = None

def deal_boundary(value, start, end):
    return start if value < start else (end if value > end else value)

def init_worker(input_array, kernel, out_shm, shape):
    """Called once in each process to set up globals without copying the output."""
    global _IN, _OUT, _K, _SHAPE
    _IN = input_array            
    _K  = kernel.astype(np.float32, copy=False)
    _SHAPE = shape
    # Rewrap the shared memory buffer as a NumPy array in this process
    _OUT = np.ctypeslib.as_array(out_shm)
    _OUT = _OUT.reshape(shape)

def process_z_slab(z_range):
    """Compute output for z in [z_start, z_end) sequentially; parallelism across slabs."""
    z_start, z_end = z_range
    Z, Y, X = _SHAPE
    # For each z in the assigned slab
    for z in range(z_start, z_end):
        for y in range(Y):
            for x in range(X):
                acc = 0.0
                # 3x3x3 neighborhood with replicate edges (clamped)
                """ dz, dy & dx refer to relative positions around the center voxel; for
                 example, dz = -1 means the voxel is just before the center voxel in z axis. 
                 Looping through all combinations we visit the entire 3x3x3 neighborhood """
                
                for dz in (-1, 0, 1):
                    zz = deal_boundary(z + dz, 0, Z - 1)
                    kz = dz + 1
                    for dy in (-1, 0, 1):
                        yy = deal_boundary(y + dy, 0, Y - 1)
                        ky = dy + 1
                        # unroll dx loop slightly for clarity/speed
                        for dx in (-1, 0, 1):
                            xx = deal_boundary(x + dx, 0, X - 1)
                            kx = dx + 1
                            acc += float(_IN[zz, yy, xx]) * float(_K[kz, ky, kx])
                _OUT[z, y, x] = acc
    # Return nothing; output is in shared memory

def build_z_tasks(Z, num_workers, chunks_per_worker=3):
    target_chunks = max(1, num_workers * chunks_per_worker)
    chunk = max(1, math.ceil(Z / target_chunks))
    tasks, s = [], 0
    while s < Z:
        e = min(Z, s + chunk)
        tasks.append((s, e))
        s = e
    return tasks

