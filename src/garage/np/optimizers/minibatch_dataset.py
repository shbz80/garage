import numpy as np

# todo many changes made here
# class BatchDataset:
#     def __init__(self, inputs, batch_size, extra_inputs=None):
#         self._inputs = []
#         self._extra_inputs = []
#         for i in inputs:
#             if not isinstance(i, list):
#                 self._inputs.append(i)
#             else:
#                 self._extra_inputs.append(i)
#                 if extra_inputs is not None:
#                     self._extra_inputs.append(extra_inputs)
#
#         self._batch_size = batch_size
#         if batch_size is not None:
#             self._ids = np.arange(self._inputs[0].shape[0])
#             self.update()
#
#     @property
#     def number_batches(self):
#         if self._batch_size is None:
#             return 1
#         return int(np.ceil(self._inputs[0].shape[0] * 1.0 / self._batch_size))
#
#     def iterate(self, update=True):
#         if self._batch_size is None:
#             yield list(self._inputs) + list(self._extra_inputs)
#         else:
#             for itr in range(self.number_batches):
#                 batch_start = itr * self._batch_size
#                 batch_end = (itr + 1) * self._batch_size
#                 batch_ids = self._ids[batch_start:batch_end]
#                 batch = [d[batch_ids] for d in self._inputs]
#                 extra_batch = [[d[id] for id in batch_ids] for d in self._extra_inputs]
#                 # yield (list(batch) + list(extra_batch))
#                 yield list(batch) + list(extra_batch)
#             if update:
#                 self.update()
#
#     def update(self):
#         np.random.shuffle(self._ids)

class BatchDataset:
    def __init__(self, inputs, batch_size, extra_inputs=None):
        self._inputs = []
        self._extra_inputs = []
        for i in inputs:
            if not isinstance(i, list):
                self._inputs.append(i)
            else:
                self._extra_inputs.append(i)
                if extra_inputs is not None:
                    self._extra_inputs.append(extra_inputs)
        # self._inputs = [i for i in inputs]
        # if extra_inputs is None:
        #     extra_inputs = []
        # self._extra_inputs = extra_inputs
        self._batch_size = batch_size
        if batch_size is not None:
            self._ids = np.arange(self._inputs[0].shape[0])
            self.update()

    @property
    def number_batches(self):
        if self._batch_size is None:
            return 1
        return int(np.ceil(self._inputs[0].shape[0] * 1.0 / self._batch_size))

    def iterate(self, update=True):
        if self._batch_size is None:
            yield list(self._inputs) + list(self._extra_inputs)
        else:
            for itr in range(self.number_batches):
                batch_start = itr * self._batch_size
                batch_end = (itr + 1) * self._batch_size
                batch_ids = self._ids[batch_start:batch_end]
                batch = [d[batch_ids] for d in self._inputs]
                extra_batch = [[d[id] for id in batch_ids] for d in self._extra_inputs]
                # yield list(batch) + list(self._extra_inputs)
                yield list(batch) + list(extra_batch)
            if update:
                self.update()

    def update(self):
        np.random.shuffle(self._ids)
