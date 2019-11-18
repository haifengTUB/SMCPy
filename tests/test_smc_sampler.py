from smcpy.smc.smc_sampler import SMCSampler
from smcpy.mcmc.mcmc_sampler import MCMCSampler
import h5py
import os
import pytest


def test_setup_communicator():
    comm, size, my_rank = SMCSampler.setup_communicator()
    assert size == 1 and my_rank == 0


def test_setup_mcmc_sampler(model):
    data = model.evaluate({'a': 0., 'b': 0.})
    param_priors = {'a': ['Uniform', 0., 1.], 'b': ['Uniform', 0., 1.]}
    mcmc = SMCSampler.setup_mcmc_sampler(data, model, param_priors)
    assert isinstance(mcmc, MCMCSampler)


def test_custom_temperature_schedule(sampler):
    num_particles = 5
    restart_time_step = 2
    num_mcmc_steps = 1
    noise_stddev = 0.5
    temperature_schedule = [0., 0.2, 1.0]

    sampler.sample(num_particles, num_mcmc_steps, temperature_schedule,
                   measurement_std_dev=noise_stddev)
    assert sampler.temp_schedule == temperature_schedule


def test_autosaver(sampler):
    num_particles = 5
    num_mcmc_steps = 1
    noise_stddev = 0.5
    temperature_schedule = [0., 0.5, 1.]
    sampler.sample(num_particles, num_mcmc_steps, temperature_schedule,
                   measurement_std_dev=noise_stddev,
                   ess_threshold=num_particles * 0.5,
                   autosave_file='autosaver.hdf5')
    with h5py.File('autosaver.hdf5', 'r') as hdf:
        group1 = hdf.get("steps")
        group1_items = list(group1.items())
        assert len(group1_items) == len(temperature_schedule) - 1


def test_load_step_list(sampler):
    step_list = sampler.load_step_list('autosaver.hdf5')
    assert len(step_list) == 2


def test_trim_step_list(sampler):
    restart_time_step = 3
    step_list = sampler.load_step_list('autosaver.hdf5')
    trimmed_list = sampler.trim_step_list(step_list, restart_time_step)
    assert len(trimmed_list) == restart_time_step - 1


def test_restart_sampling(sampler):
    num_particles = 5
    restart_time_step = 2
    num_mcmc_steps = 1
    noise_stddev = 0.5
    temperature_schedule = [0., 0.5, 1.]
    sampler.sample(num_particles, num_mcmc_steps, temperature_schedule,
                   measurement_std_dev=noise_stddev,
                   restart_time_step=restart_time_step,
                   hdf5_to_load='autosaver.hdf5',
                   autosave_file='restart.hdf5')
    with h5py.File('restart.hdf5', 'r') as hdf:
        group1 = hdf.get("steps")
        group1_items = list(group1.items())
        assert len(group1_items) == restart_time_step - 1
    os.remove('autosaver.hdf5')
    os.remove('restart.hdf5')


def test_custom_temperature_schedule_sets_num_time_steps(sampler):
    num_particles = 5
    num_mcmc_steps = 1
    noise_stddev = 0.5
    temperature_schedule = [0., 0.2, 1.0]

    sampler.sample(num_particles, num_mcmc_steps=num_mcmc_steps,
                   measurement_std_dev=noise_stddev,
                   temperature_schedule=temperature_schedule)
    assert sampler.num_time_steps == 3
