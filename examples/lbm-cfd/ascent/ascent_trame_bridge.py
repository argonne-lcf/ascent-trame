import sys
sys.path.append(f'../../.venv/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages')
import time
import numpy as np
from multiprocessing.managers import BaseManager
from mpi4py import MPI
import conduit
import ascent.mpi

class QueueManager(BaseManager):
    pass

def main():
    # obtain a mpi4py mpi comm object
    comm = MPI.Comm.f2py(ascent_mpi_comm_id())

    # get task id and number of total tasks
    task_id = comm.Get_rank()
    num_tasks = comm.Get_size()

    # get this MPI task's published blueprint data
    mesh_data = ascent_data().child(0)

    total_x = mesh_data["state/total_size/w"]
    total_y = mesh_data["state/total_size/h"]
    dim_x = mesh_data["coordsets/coords/dims/i"]
    dim_y = mesh_data["coordsets/coords/dims/j"]
    origin_x = mesh_data["coordsets/coords/origin/x"]
    origin_y = mesh_data["coordsets/coords/origin/y"]
    vorticity = mesh_data["fields/vorticity/values"].reshape((dim_y, dim_x))

    data_types = create2dGridDataTypes(comm, task_id, num_tasks, dim_x, dim_y, origin_x, origin_y, total_x, total_y)
    vorticity_all = np.empty((total_y, total_x), np.float64) if task_id == 0 else None
    gatherDataOnRoot(comm, task_id, num_tasks, vorticity, vorticity_all, data_types)

    update_data = None
    if task_id == 0:
        QueueManager.register('get_data_queue')
        QueueManager.register('get_signal_queue')
        mgr = QueueManager(address=('127.0.0.1', 8000), authkey=b'ascent-trame')

        f = open('sim_data.txt', 'w', encoding='utf-8')
        f.write(f'{dir(ascent.mpi)}')
        f.close()
        
        try:
            mgr.connect()

            queue_data = mgr.get_data_queue()
            queue_signal = mgr.get_signal_queue()

            num_barriers = mesh_data["state/num_barriers"]
            barriers = mesh_data["state/barriers"].reshape((num_barriers, 4))

            queue_data.put({'barriers': barriers, 'vorticity': vorticity_all})
            update_data = queue_signal.get()
        except:
            update_data = {}

    # broadcast updates to all ranks, then pass to Ascent callback
    update_data = comm.bcast(update_data, root=0)

    update_node = conduit.Node()
    update_node['task_id'] = task_id
    if 'flow_speed' in update_data:
        update_node['flow_speed'] = update_data['flow_speed']
    if 'barriers' in update_data:
        num_barriers = update_data['barriers'].shape[0]
        update_node['num_barriers'] = num_barriers
        update_node['barriers'].set_external(update_data['barriers'].reshape(num_barriers * 4))
    output = conduit.Node()
    ascent.mpi.execute_callback('steeringCallback', update_node, output)

    freeDataTypes(data_types)

def create2dGridDataTypes(comm, task_id, num_tasks, dim_x, dim_y, origin_x, origin_y, total_x, total_y):
    data_types = {}

    col_type = 'FIRST' if origin_x == 0 else 'LAST' if origin_x + dim_x >= total_x else 'MIDDLE'
    row_type = 'FIRST' if origin_y == 0 else 'LAST' if origin_y + dim_y >= total_y else 'MIDDLE'
    num_x = dim_x if num_tasks == 0 else dim_x - 1 if col_type == 'FIRST' or col_type == 'LAST' else dim_x - 2
    num_y = dim_y if num_tasks == 0 else dim_y - 1 if row_type == 'FIRST' or row_type == 'LAST' else dim_y - 2
    start_x = 0 if col_type == 'FIRST' else 1
    start_y = 0 if row_type == 'FIRST' else 1    

    array = np.array([dim_y, dim_x], dtype=np.int32)
    subsize = np.array([num_y, num_x], dtype=np.int32)
    offset = np.array([start_y, start_x], dtype=np.int32)

    partitions = np.empty((num_tasks, 4), dtype=np.int32) if task_id == 0 else None
    comm.Gather(np.array([num_x, num_y, origin_x, origin_y], np.int32), partitions, root=0)

    data_types['own_scalar'] = MPI.DOUBLE.Create_subarray(array, subsize, offset, order=MPI.ORDER_C)
    data_types['own_scalar'].Commit()
    data_types['other_scalar'] = []
    if task_id == 0:
        array[0] = total_y
        array[1] = total_x
        for i in range(num_tasks): 
            subsize[0] = partitions[i][1]
            subsize[1] = partitions[i][0]
            offset[0] = partitions[i][3]
            offset[1] = partitions[i][2]
            dtype = MPI.DOUBLE.Create_subarray(array, subsize, offset, order=MPI.ORDER_C)
            dtype.Commit()
            data_types['other_scalar'].append(dtype)

    return data_types

def gatherDataOnRoot(comm, task_id, num_tasks, data, recv_data, data_types):
    if task_id == 0:
        comm.Sendrecv((data, 1, data_types['own_scalar']), 0, 0, (recv_data, 1, data_types['other_scalar'][0]), 0, 0, status=None)
        for i in range(1, num_tasks):
            comm.Recv((recv_data, 1, data_types['other_scalar'][i]), i, 0, status=None)
    else:
        comm.Send((data, 1, data_types['own_scalar']), 0, 0)
            

def createDivergingColorMap(size):
    # create color map - diverging: blue-white-red
    cmap = np.empty((size, 3), dtype=np.uint8)

    for n in range(size):
        if n < (size // 2):  # blue to white
            t = n / (size // 2)
            cmap[n][0] = int(t * 255.0)
            cmap[n][1] = int(t * 255.0)
            cmap[n][2] = 255
        else:                # white to red
            t = (n - (size // 2)) / (size // 2)
            cmap[n][0] = 255;
            cmap[n][1] = int((1.0 - t) * 255.0)
            cmap[n][2] = int((1.0 - t) * 255.0)

    return cmap 

def scalarArrayToRgb(scalar, s_min, s_max, cmap):
    h, w = scalar.shape
    cmap_size, channels = cmap.shape
    pixels = np.empty((h, w, 3), dtype=np.uint8)
    for j in range(h - 1, -1, -1):
        px_row = h - j - 1
        for i in range(w):
            scalar_val = scalar[j][i]
            if scalar_val < s_min:
                scalar_val = s_min
            elif scalar_val > s_max:
                scalar_val = s_max
            val = int((cmap_size - 1) * ((scalar_val - s_min) / (s_max - s_min)))
            pixels[px_row][i][0] = cmap[val][0]
            pixels[px_row][i][1] = cmap[val][1]
            pixels[px_row][i][2] = cmap[val][2]
    return pixels

def freeDataTypes(data_types):
    data_types['own_scalar'].Free()
    for dtype in data_types['other_scalar']:
        dtype.Free()

main()
