from multiprocessing import Process, Queue
from multiprocessing.managers import BaseManager

class QueueManager(BaseManager):
    pass

class AscentConsumer:
    def __init__(self, port):
        # create queues for Ascent data
        self._queue_data = Queue()
        self._queue_signal = Queue()

        # create queues for Python app state and updates
        self._state_queue = Queue()
        self._update_queue = Queue()

        # start Queue Manager in new thread
        self._queue_mgr_thread = Process(target=AscentConsumer._runQueueManager, args=(port, self._queue_data, self._queue_signal))
        self._queue_mgr_thread.daemon = True
        self._queue_mgr_thread.start()

        # start bridge to Python app in new thread
        ascent_bridge_thread = Process(target=AscentConsumer._runAscentBridge, args=(self._queue_data, self._queue_signal, self._state_queue, self._update_queue))
        ascent_bridge_thread.daemon = True
        ascent_bridge_thread.start()

    def getStateQueue(self):
        return self._state_queue

    def getUpdateQueue(self):
        return self._update_queue

    @staticmethod
    def _runQueueManager(port, queue_data, queue_signal):
        # register queues with Queue Manager
        QueueManager.register('get_data_queue', callable=lambda:queue_data)
        QueueManager.register('get_signal_queue', callable=lambda:queue_signal)

        # create Queue Manager
        mgr = QueueManager(address=('127.0.0.1', port), authkey=b'ascent-trame')

        # start Queue Manager server
        server = mgr.get_server()
        server.serve_forever()

    @staticmethod
    def _runAscentBridge(queue_data, queue_signal, state_queue, update_queue):
        while True:
            sim_data = queue_data.get()
            state_queue.put(sim_data)
            updates = update_queue.get()
            queue_signal.put(updates)

