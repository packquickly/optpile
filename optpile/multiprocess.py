import math
import os
from collections.abc import Callable
from multiprocessing import Process, Queue
from typing import Optional

import dill


#
# This is a helper module for the `run` and `full_history_run` methods in
# `opt_tester` allowing for parallelism.
#
# Python's `multiprocessing` relies on pickle for object serialisation, which is a
# huge pain, and means it can't accept functions not defined at the top-level.
# `run` and `full_history_run` both
#
# A fork `multprocess` does exist which is supposed to address this problem, but I
# (packquickly) consistently get import errors trying to use it (and don't like
# introducing another dependency.)
#
# This module implements a simple threadpool (processpool?) manually using dill.
#


class CustomProcess(Process):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._target = dill.dumps(self._target)

    def run(self):
        if self._target:
            self._target = dill.loads(self._target)
            self._target(*self._args, **self._kwargs)


# A context manager which simplifies mapping a function over a set
# of inputs.
class ProcessPool:
    def __init__(self, n_cpus: Optional[int] = None):
        if n_cpus is None:
            max_processes = os.cpu_count()
            if max_processes is None:
                print(
                    "WARNING: no cpus found on machine defaulting to a single process. "
                    "Try passing `n_cpus` manually if this is an error."
                )
            else:
                self.max_processes = max_processes
        else:
            self.max_processes = n_cpus
        self._processes = []

    def map(self, fn: Callable, input_list: list):
        # Used for message passing between processes.
        chunk_size = math.ceil(len(input_list) / self.max_processes)
        n_chunks = len(input_list) // chunk_size
        identified_sequence = [(id, value) for id, value in enumerate(input_list)]
        chunked_sequence = []
        for i in range(0, n_chunks - 1):
            chunked_sequence.append(
                identified_sequence[i * chunk_size : (i + 1) * chunk_size]
            )
        chunked_sequence.append(identified_sequence[(n_chunks - 1) * chunk_size :])
        q = Queue()

        def id_managed_fn(id_value, q):
            ids, value = zip(*id_value)
            output = map(fn, value)
            q.put([(id, output) for (id, output) in zip(ids, output)])

        def start_new_process(id_value):
            p = CustomProcess(target=id_managed_fn, args=(id_value, q))
            p.start()
            return p

        for _ in range(n_chunks):
            self._processes.append(start_new_process(chunked_sequence.pop(0)))

        results = []
        for p in self._processes:
            p.join()
            results.append(q.get())

        # This dark magic comprehension just flattens a list of lists
        results = [item for sublist in results for item in sublist]

        # Empty the list of processes.
        self._processes = []
        # Sort the results based on the key, and then remove all the keys
        return [x for x in map(lambda x: x[1], sorted(results, key=lambda x: x[0]))]