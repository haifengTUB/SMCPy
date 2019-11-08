'''
Notices:
Copyright 2018 United States Government as represented by the Administrator of
the National Aeronautics and Space Administration. No copyright is claimed in
the United States under Title 17, U.S. Code. All Other Rights Reserved.

Disclaimers
No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF
ANY KIND, EITHER EXPRessED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED
TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNess FOR A PARTICULAR PURPOSE, OR
FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR
FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE
SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN
ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS,
RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS
RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY
DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF
PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."

Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE
UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY
PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY
LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE,
INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S
USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLess THE
UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY
PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR
ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS
AGREEMENT.
'''

from ..mcmc.mcmc_sampler import MCMCSampler
from ..smc.smc_step import SMCStep
from ..hdf5.hdf5_storage import HDF5Storage
from ..utils.properties import Properties
from ..utils.progress_bar import set_bar
from ..utils.single_rank_comm import SingleRankComm
from ..particles.particle_initializer import ParticleInitializer
from ..particles.particle_updater import ParticleUpdater
from ..particles.particle_mutator import ParticleMutator
from tqdm import tqdm
import numpy as np
import imp


class SMCSampler(Properties):
    '''
    Class for performing parallel Sequential Monte Carlo sampling.
    '''

    def __init__(self, data, model, param_priors):
        self._comm, self._size, self._rank = self.setup_communicator()
        self._mcmc = self.setup_mcmc_sampler(data, model, param_priors)
        self._step_list = []
        super(SMCSampler, self).__init__()

    @staticmethod
    def setup_communicator():
        """
        Detects whether multiple processors are available and sets
        self.number_CPUs and self.cpu_rank accordingly.
        """
        try:
            imp.find_module('mpi4py')

            from mpi4py import MPI
            comm = MPI.COMM_WORLD.Clone()

            size = comm.size
            rank = comm.rank
            comm = comm

        except ImportError:

            size = 1
            rank = 0
            comm = SingleRankComm()

        return comm, size, rank

    @staticmethod
    def setup_mcmc_sampler(data, model, param_priors):
        mcmc = MCMCSampler(data=data, model=model, params=param_priors,
                           storage_backend='ram')
        return mcmc

    def sample(self, num_particles, num_time_steps, num_mcmc_steps,
               measurement_std_dev=None, ess_threshold=None,
               proposal_center=None, proposal_scales=None, restart_time_step=1,
               hdf5_to_load=None, autosave_file=None):
        '''
        Driver method that performs Sequential Monte Carlo sampling.

        :param num_particles: number of particles to use during sampling
        :type num_particles: int
        :param num_time_steps: number of time steps in temperature schedule that
            is used to transition between prior and posterior distributions.
        :type num_time_steps: int
        :param num_mcmc_steps: number of mcmc steps to take during mutation
        :param num_mcmc_steps: int
        :param measurement_std_dev: standard deviation of the measurement error;
            if unknown, set to None and it will be estimated along with other
            model parameters.
        :type measurement_std_dev: float or None
        :param ess_threshold: threshold equivalent sample size; triggers
            resampling when ess > ess_threshold
        :type ess_threshold: float or int
        :param proposal_center: initial parameter dictionary, which is used to
            define the initial proposal distribution when generating particles;
            default is None, and initial proposal distribution = prior.
        :type proposal_center: dict
        :param proposal_scales: defines the scale of the initial proposal
            distribution, which is centered at proposal_center, the initial
            parameters; i.e. prop ~ MultivarN(q1, (I*proposal_center*scales)^2).
            Proposal scales should be passed as a dictionary with keys and
            values corresponding to parameter names and their associated scales,
            respectively. The default is None, which sets initial proposal
            distribution = prior.
        :type proposal_scales: dict
        :param restart_time_step: time step at which to restart sampling;
            default is zero, meaning the sampling process starts at the prior
            distribution; note that restart_time_step < num_time_steps. The
            step at restart_time is retained, and the sampling begins at the
            next step (t=restart_time_step+1).
        :type restart_time_step: int
        :param hdf5_to_load: file path of a step list
        :type hdf5_to_load: string
        :param autosave_file: file name of autosave file
        :type autosave_file: string

        :Returns: A list of SMCStep class instances that contains all particles
            and their past generations at every time step.
        '''

        self.autosaver = autosave_file
        self.num_time_steps = num_time_steps
        self.restart_time_step = restart_time_step
        self.temp_schedule = np.linspace(0., 1., num_time_steps)
        start_time_step = 1
        if self.restart_time_step == 1:
            initializer = ParticleInitializer(self._mcmc, self.temp_schedule,
                                              self._comm)
            initializer.set_proposal_distribution(proposal_center,
                                                  proposal_scales)
            particles = initializer.initialize_particles(measurement_std_dev,
                                                         num_particles)
            self.step = self._initialize_step(particles)
            self._add_step_to_step_list(self.step)
            self._autosave_step(1)

        else:
            start_time_step = restart_time_step
            step_list = self.load_step_list(hdf5_to_load)
            self.step_list = self.trim_step_list(step_list,
                                                 self.restart_time_step,
                                                 self._comm)
            self.step = self.step_list[-1].copy()
            self._autosave_step_list()

        updater = ParticleUpdater(self.step, ess_threshold, self._comm)

        p_bar = tqdm(range(num_time_steps)[start_time_step + 1:])
        last_ess = num_particles
        for t in p_bar:
            temperature_step = self.temp_schedule[t] - self.temp_schedule[t - 1]
            self.step = updater.update_log_weights(temperature_step)
            self.step = updater.resample_if_needed()
            mutator = ParticleMutator(self.step, self._mcmc, num_mcmc_steps,
                                      self._comm)
            self.step = mutator.mutate_particles(measurement_std_dev,
                                                 self.temp_schedule[t])
            self._autosave_step(t)
            self._add_step_to_step_list(self.step)
            
            if self._rank == 0:
                set_bar(p_bar, t, last_ess, updater._ess,
                        mutator.mutation_ratio, updater._resample_status)
                last_ess = updater._ess

        self._close_autosaver()
        return self._step_list

    @staticmethod
    def load_step_list(h5_file, mpi_comm=SingleRankComm()):
        '''
        Loads and returns a step list stored using the HDF5Storage
        class.

        :param hdf5_to_load: file path of a step_list saved using the
            self.save_step_list() methods.
        :type hdf5_to_load: string

        :Returns: A list of SMCStep class instances that contains all particles
            at each time step.
        '''

        rank = mpi_comm.Get_rank()
        if rank == 0:
            hdf5 = HDF5Storage(h5_file, mode='r')
            step_list = hdf5.read_step_list()
            hdf5.close()
            print 'Step list loaded from %s.' % h5_file
        else:
            step_list = None
        return step_list

    @staticmethod
    def trim_step_list(step_list, restart_time_step, mpi_comm=SingleRankComm()):
        rank = mpi_comm.Get_rank()
        if rank == 0:
            to_keep = range(0, restart_time_step - 1)
            trimmed_steps = [step_list[i] for i in to_keep]
            step_list = trimmed_steps
        return step_list

    def save_step_list(self, h5_file):
        '''
        Saves self.step to an hdf5 file using the HDF5Storage class.
        :param h5_file: file path at which to save step list
        :type h5_file: string
        '''

        if self._rank == 0:
            hdf5 = HDF5Storage(h5_file, mode='w')
            hdf5.write_step_list(self.step_list)
            hdf5.close()
        return None

    def _initialize_step(self, particles):
        particles = self._comm.gather(particles, root=0)
        if self._rank == 0:
            step = SMCStep()
            step.set_particles(np.concatenate(particles))
            step.normalize_step_log_weights()
        else:
            step = None
        return step

    def _add_step_to_step_list(self, step):
        if self._rank == 0:
            self._step_list.append(step.copy())
        return None

    def _autosave_step(self, step_index):
        if self._rank == 0 and self._autosaver is not None:
            self.autosaver.write_step(self.step, step_index)
        return None

    def _close_autosaver(self):
        if self._rank == 0 and self._autosaver is not None:
            self.autosaver.close()
        return None

    def _autosave_step_list(self):
        if self._rank == 0 and self._autosaver is not None:
            self.autosaver.write_step_list(self.step_list)
        return None
