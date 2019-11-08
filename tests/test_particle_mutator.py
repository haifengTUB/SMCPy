import pytest
from smcpy.mcmc.mcmc_sampler import MCMCSampler
from smcpy.smc.smc_step import SMCStep
from smcpy.particles.particle_mutator import ParticleMutator
from smcpy.particles.particle import Particle


class StubParticle(Particle):

    def __init__(self, std_dev_sample=None):
        self.params = {'a': 1, 'b': 2, 'c': 3}
        self.log_weight = -2.
        if std_dev_sample is not None:
            self.params['std_dev'] = std_dev_sample


@pytest.fixture
def measurement_std_dev():
    return 1


@pytest.fixture
def stub_step_with_std_dev_samples():
    return [StubParticle() for _ in range(10)]


@pytest.fixture
def stub_step_fixed_std_dev():
    return [StubParticle(measurement_std_dev) for _ in range(10)]


@pytest.mark.parametrize('mutated_params,mutation_ratio',
                         [({'a': 2, 'b': 3, 'c': 3}, 1),
                          ({'a': 1, 'b': 2, 'c': 3}, 0)])
def test_mutation_count_1_and_0(mocker, stub_step_with_std_dev_samples,
                                mutated_params, mutation_ratio):
    mcmc = MCMCSampler(data=[None], model=None, params=[None])
    mocker.patch.object(mcmc, 'sample')
    mocker.patch.object(mcmc, 'get_log_likelihood', return_value=-2.)
    mocker.patch.object(mcmc, 'generate_pymc_model')
    mocker.patch.object(mcmc, 'get_state', return_value=mutated_params)

    step = SMCStep()
    step.set_particles(stub_step_with_std_dev_samples)
    pm = ParticleMutator(step, mcmc, num_mcmc_steps=2)
    step = pm.mutate_particles(measurement_std_dev=1)

    assert pm.mutation_ratio == mutation_ratio
