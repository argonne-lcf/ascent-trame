import numpy as np
from mpi4py import MPI
import conduit
from ascenttrame.producer import AscentProducer

def main():
    # create Ascent-Trame bridge data producer
    comm =  MPI.Comm.f2py(ascent_mpi_comm_id())
    bridge = AscentProducer(comm)

    # only need rank 0 to connect and pass data to/from Trame application
    port = 8000 if bridge.getTaskId() == 0 else -1
    if bridge.connectToTrameQueues(port):
        # get simulation data published to Ascent
        mesh_data = ascent_data().child(0)
        
        # trigger callback to redistribute data (gather on rank 0)
        output = conduit.Node()
        bridge.triggerSteeringCallback('repartitionCallback', mesh_data, output)

        if bridge.getTaskId() == 0:
            num_barriers = mesh_data['state/num_barriers']
            barriers = mesh_data['state/barriers'].reshape((num_barriers, 4))
            topology_name = output['fields/vorticity/topology']
            coordset_name = output[f'topologies/{topology_name}/coordset']
            dim_x = output[f'coordsets/{coordset_name}/dims/i'] - 1 # 1 fewer element than vertex
            dim_y = output[f'coordsets/{coordset_name}/dims/j'] - 1 # 1 fewer element than vertex
            vorticity = output['fields/vorticity/values'].reshape((dim_y, dim_x))

            # pass simulation data to Trame application
            bridge.sendDataToTrame({'barriers': barriers, 'vorticity': vorticity})

        # get steering updates from Trame application
        update_data = bridge.getSteeringDataFromTrame()
                
        # trigger callback in simulation with steering updates
        update_node = conduit.Node()
        update_node['task_id'] = bridge.getTaskId()
        if 'flow_speed' in update_data:
            update_node['flow_speed'] = update_data['flow_speed']
        if 'barriers' in update_data:
            num_barriers = update_data['barriers'].shape[0]
            update_node['num_barriers'] = num_barriers
            update_node['barriers'].set_external(update_data['barriers'].ravel())
        output = conduit.Node()
        bridge.triggerSteeringCallback('steeringCallback', update_node, output)
            

main()

