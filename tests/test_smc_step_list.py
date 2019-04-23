from smcpy.smc.smc_step import SMCStep
from smcpy.particles.particle import Particle
from ..hdf5.hdf5_storage import HDF5Storage
import pytest
import numpy as np


@pytest.fixture
def particle_list():
    particle = Particle({'a': 1, 'b': 2}, 0.2, -0.2)
    return 5 * [particle]


@pytest.fixture
def step_tester():
    return SMCStep()


@pytest.fixture
def filled_step(step_tester, particle_list):
    step_tester.set_particles(particle_list)
    return step_tester


@pytest.fixture
def h5file():
    return HDF5Storage('temp.hdf5', 'w')
