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

    #f = open(f'sim_data_{task_id:02d}.txt', 'w', encoding='utf-8')
    #f.write(f'RANK {task_id}\n')

    update_data = None
    if task_id == 0:
        # get rank 0's published blueprint data
        mesh_data = ascent_data().child(0)

        topology_name = mesh_data['fields/vorticity_whole/topology']
        #f.write(f'{topology_name}\n')
        coordset_name = mesh_data[f'topologies/{topology_name}/coordset']
        #f.write(f'{coordset_name}\n')
        dim_x = mesh_data[f'coordsets/{coordset_name}/dims/i'] - 1 # 1 fewer element than vertex
        dim_y = mesh_data[f'coordsets/{coordset_name}/dims/j'] - 1 # 1 fewer element than vertex
        #f.write(f'{dim_x}x{dim_y}\n')
        vorticity = mesh_data['fields/vorticity_whole/values'].reshape((dim_y, dim_x))

        QueueManager.register('get_data_queue')
        QueueManager.register('get_signal_queue')
        mgr = QueueManager(address=('127.0.0.1', 8000), authkey=b'ascent-trame')

        try:
            mgr.connect()

            queue_data = mgr.get_data_queue()
            queue_signal = mgr.get_signal_queue()

            num_barriers = mesh_data["state/num_barriers"]
            barriers = mesh_data["state/barriers"].reshape((num_barriers, 4))

            queue_data.put({'barriers': barriers, 'vorticity': vorticity})
            update_data = queue_signal.get()
        except:
            update_data = {}

    #f.close()

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

main()

