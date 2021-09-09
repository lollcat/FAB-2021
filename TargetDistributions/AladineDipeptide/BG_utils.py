import torch
from boltzgen import openmm_interface as omi

class PriorDistribution:
    def __init__(self):
        raise NotImplementedError

    def log_prob(self, z):
        """
        :param z: value or batch of latent variable
        :return: log probability of the distribution for z
        """
        raise NotImplementedError

class Boltzmann(PriorDistribution):
    """
    Boltzmann distribution using OpenMM to get energy and forces
    """
    def __init__(self, sim_context, temperature, energy_cut, energy_max):
        """
        Constructor
        :param sim_context: Context of the simulation object used for energy
        and force calculation
        :param temperature: Temperature of System
        """
        # Save input parameters
        self.sim_context = sim_context
        self.temperature = temperature
        self.energy_cut = torch.tensor(energy_cut)
        self.energy_max = torch.tensor(energy_max)

        # Set up functions
        self.openmm_energy = omi.OpenMMEnergyInterface.apply
        self.regularize_energy = omi.regularize_energy

        self.norm_energy = lambda pos: self.regularize_energy(
            self.openmm_energy(pos, self.sim_context, temperature)[:, 0],
            self.energy_cut, self.energy_max)

    def log_prob(self, z):
        # torch.autograd.grad(torch.sum(-self.norm_energy(z)), z)
        return -self.norm_energy(z)