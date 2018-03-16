# -*- coding: utf-8 -*-
__all__ = [
    "Devices",
]

from collections import Iterable

# Local imports
from device import Device


class Devices(Iterable):

    def __init__(self, device_ids=0, n_streams=0):
        """
        Iterable of CUDA device objects. Using this allows for access 
        to multiple GPUs from a single CPU host thread.

        Parameters
        ----------
        dev_ids : int or list of ints, optional
            CUDA device ID(s).

        nstreams : int or list of ints, optional
            Number of CUDA streams per device.

        Attributes
        ----------
        devices : list
            CUDA enabled devices.
        """
        try:
            device_ids = list(device_ids)
        except device_ids:
            device_ids = [device_ids]

        try:
            n_streams = list(n_streams)               #Each device has a different number of streams
        except TypeError:
            n_streams = len(device_ids) * [n_streams] #Each device has the same number of streams

        # Create the Device container, and then immediately pop it so the next can be created
        self.__devices = [Device(_device_id, _nstreams) for _device_id, _nstreams in zip(device_ids, n_streams)]


    def sync(self):
        """
        Block the host thread until all devices have completed 
        all of their tasks.
        """
        for d in self.__devices:
            d.sync()


    def __len__(self):
        return len(self.__devices)


    def __iter__(self):
        for d in self.__devices:
            yield d


    def __getitem__(self, key):
        return self.__devices[key]


    def __enter__(self):
        return self


    def __exit__(self, *args, **kwargs):
        for d in self.__devices:
            d.__exit__()