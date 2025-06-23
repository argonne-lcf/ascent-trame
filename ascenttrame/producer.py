import numpy as np
from multiprocessing.managers import BaseManager
from mpi4py import MPI
import conduit
import ascent.mpi

class QueueManager(BaseManager):
    pass

class AscentProducer:
    def __init__(self, comm):
        # obtain a mpi4py mpi comm object
        self._comm = comm
        
        # get task id and number of total tasks
        self._task_id = self._comm.Get_rank()
        self._num_tasks = self._comm.Get_size()

        # initialize Trame queues
        self._queue_manager = None
        self._queue_data = None
        self._queue_signal = None

    def getTaskId(self):
        return self._task_id

    def getNumberOfTasks(self):
        return self._num_tasks

    def connectToConsumerQueues(self, port_number):
        interactive = np.array([1], np.uint8)

        if port_number >= 0:
            # attempt to connect to Trame queue manager
            QueueManager.register('get_data_queue')
            QueueManager.register('get_signal_queue')
            self._queue_manager = QueueManager(address=('127.0.0.1', port_number), authkey=b'ascent-trame')

            try:
                self._queue_manager.connect()
            except:
                self._queue_manager = None
                interactive[0] = 0

        # check if all tasks with corresponding trame tasks successfully connected
        global_interactive = np.empty(1, np.uint8)
        self._comm.Allreduce((interactive, 1, MPI.UNSIGNED_CHAR), (global_interactive, 1, MPI.UNSIGNED_CHAR), op=MPI.MIN)

        if global_interactive[0] == 1 and self._queue_manager is not None:
            # get access to Trame's queues
            self._queue_data = self._queue_manager.get_data_queue()
            if self._task_id == 0:
                self._queue_signal = self._queue_manager.get_signal_queue()

        return bool(global_interactive[0])

    def sendSimulationDataToConsumer(self, data):
        if self._queue_data is not None:
            self._queue_data.put(data)

    def getSteeringDataFromConsumer(self):
        update_data = {}
        if self._queue_signal is not None:
            update_data = self._queue_signal.get()
        update_data = self._comm.bcast(update_data, root=0)
        return update_data

    def triggerSteeringCallback(self, callback_name, in_node, out_node):
        ascent.mpi.execute_callback(callback_name, in_node, out_node)

