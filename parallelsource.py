from __future__ import print_function

import numpy as np
#import torch
#from torch.multiprocessing import Process, Event, Queue
from multiprocessing import Process, Event, Queue

from neuralnilm.data.source import Source, Sequence

import logging
logger = logging.getLogger(__name__)

#torch.multiprocessing.set_sharing_strategy('file_system')

#class TorchSequence():
#    def __init__(self, seq):
#        self.__converted = False
#        self.input = torch.from_numpy(seq.input).share_memory_()
#        self.target = torch.from_numpy(seq.target).share_memory_()
#        self.all_appliances = None if seq.all_appliances.empty else seq.all_appliances
#        self.metadata = seq.metadata
#        self.weights = seq.weights
#
#    def to_sequence(self):
#        if self.__converted is False:
#            self.input = self.input.numpy()
#            self.target = self.target.numpy()
#            if self.all_appliances is None:
#                self.all_appliances = Sequence.empty_df
#            self.__converted = True
#        return self

class TrainingSequence():
    def __init__(self, seq):
        self.input = seq.input
        self.target = seq.target
        self.weights = seq.weights

    def to_sequence(self):
        self.all_appliances = Sequence.empty_df
        self.metadata = {}
        return self

class MultiprocessActivationsSource():
    def __init__(self, source, num_processes=4, num_seq_per_batch=64, master_seed=42, **get_sequence_kwargs):
        self._stop = Event()
        self._is_started = False
        self._queue = Queue(maxsize=32)
        self._source = source
        self._get_sequence_kwargs = get_sequence_kwargs
        self._master_seed = master_seed
        self._num_processes = num_processes
        self._num_seq_per_batch = num_seq_per_batch
        self._processes = []

    @property
    def num_batches_for_validation(self):
        return self._source.num_batches_for_validation

    def run(self, rng_seed):
        self._source.rng_seed = rng_seed
        self._source.rng = np.random.RandomState(rng_seed)

        def compile_batch():
            batch = []
            for i in range(self._num_seq_per_batch):
                seq = self._source._get_sequence(fold='train', **self._get_sequence_kwargs)
                batch.append(TrainingSequence(seq))
            return batch

        batch = compile_batch()
        while not self._stop.is_set():
            try:
                self._queue.put(batch)
            except AssertionError:
                # queue is closed
                break
            batch = compile_batch()

    def get_batch(self, *args, **kwargs):
        return Source.get_batch(self, *args, **kwargs)

    def get_sequence(self, fold='train', timeout=30, **get_sequence_kwargs):
        if fold == 'train':
            if self._is_started:
                while True:
                    batch = self._queue.get(timeout=timeout)
                    for seq in batch:
                        yield seq.to_sequence()
            else:
                raise RuntimeError("Process is not running!")
        else:
            return self._source.get_sequence(fold=fold, **get_sequence_kwargs)

    def start(self):
        if self._is_started == False:
            rng = np.random.RandomState(self._master_seed)
            MAX_SEED = 2**32-1
            self._is_started = True
            for i in range(self._num_processes):
                seed = rng.randint(MAX_SEED)
                p = Process(target=self.run, args=(seed,), name='neuralnilm-source-process-{}'.format(i))
                self._processes.append(p)
                p.start()

    def stop(self):
        self._stop.set()
        self._queue.close()
        for p in self._processes:
            p.terminate()
        for p in self._processes:
            p.join()

    def report(self):
        return self._source.report()