"""General use functions.
"""
import time
import argparse
import numpy as np
from scipy.optimize import nnls

################################################################################

class Battery:
    def __init__(self, initial_level=1.0, max_level=1.0, min_level=0.0, discharge_rate=0.01):
        self.level = initial_level
        self.max_level = max_level
        self.min_level = min_level
        self.discharge_rate = discharge_rate

    def discharge(self, amount):
        self.level -= amount
        if self.level < self.min_level:
            self.level = self.min_level

    def charge(self, amount):
        self.level += amount
        if self.level > self.max_level:
            self.level = self.max_level

    def get_level(self):
        return self.level

    def set_level(self, level):
        if level < self.min_level:
            self.level = self.min_level
        elif level > self.max_level:
            self.level = self.max_level
        else:
            self.level = level


def sync(i, start_time, timestep):
    """Syncs the stepped simulation with the wall-clock.

    Function `sync` calls time.sleep() to pause a for-loop
    running faster than the expected timestep.

    Parameters
    ----------
    i : int
        Current simulation iteration.
    start_time : timestamp
        Timestamp of the simulation start.
    timestep : float
        Desired, wall-clock step of the simulation's rendering.

    """
    if timestep > .04 or i%(int(1/(24*timestep))) == 0:
        elapsed = time.time() - start_time
        if elapsed < (i*timestep):
            time.sleep(timestep*i - elapsed)

################################################################################

def str2bool(val):
    """Converts a string into a boolean.

    Parameters
    ----------
    val : str | bool
        Input value (possibly string) to interpret as boolean.

    Returns
    -------
    bool
        Interpretation of `val` as True or False.

    """
    if isinstance(val, bool):
        return val
    elif val.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif val.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("[ERROR] in str2bool(), a Boolean value is expected")
