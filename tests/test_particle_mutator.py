import pytest
from smcpy.mcmc.mcmc_sampler import MCMCSampler
from smcpy.smc.smc_step import SMCStep
from smcpy.particles.particle_mutator import ParticleMutator
from smcpy.particles.particle import Particle


class StubParticle(Particle):

    def __init__(self, params, std_dev_sample=None):
        self.params = params
        self.log_weight = -2.
        if std_dev_sample is not None:
            self.params['std_dev'] = std_dev_sample


@pytest.fixture
def mcmc(mocker):
    mcmc = MCMCSampler(data=[None], model=None, params=[None])
    mocker.patch.object(mcmc, 'sample')
    mocker.patch.object(mcmc, 'get_log_likelihood', return_value=-2.)
    mocker.patch.object(mcmc, 'generate_pymc_model')
    return mcmc

@pytest.fixture
def particle_list_1():
    return [StubParticle({'a': 1, 'b': 2, 'c': 3}) for _ in range(10)]

@pytest.fixture
def particle_list_mixed():
    list1 = [StubParticle({'a': 1, 'b': 2, 'c': 3}) for _ in range(5)]
    list2 = [StubParticle({'a': 2, 'b': 3, 'c': 3}) for _ in range(5)]
    return list1 + list2


@pytest.mark.parametrize('mutated_params,mutation_ratio,list_choice',
                         [({'a': 2, 'b': 3, 'c': 3}, 1, 0),
                          ({'a': 1, 'b': 2, 'c': 3}, 0, 0),
                          ({'a': 2, 'b': 3, 'c': 3}, 0.5, 1)])
def test_mutation_count(mocker, mcmc, mutated_params, mutation_ratio,
                        list_choice, particle_list_1, particle_list_mixed):
    mocker.patch.object(mcmc, 'get_state', return_value=mutated_params)

    step = SMCStep()
    init_list = [particle_list_1, particle_list_mixed][list_choice]
    step.set_particles(init_list)
    pm = ParticleMutator(step, mcmc, measurement_std_dev=1)
    new_step = pm.mutate_particles(num_mcmc_steps=2)

    assert pm.mutation_ratio == mutation_ratio
