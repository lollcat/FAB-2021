from openmmtools.testsystems import AlanineDipeptideVacuum
from bgtorch.utils.train import linlogcut
from bgtorch.distribution.energy import Energy
import mdtraj as md
import torch
from simtk import openmm as mm
from simtk import unit
from simtk.openmm import app
from openmmtools import testsystems
from simtk.openmm.app import StateDataReporter
import mdtraj
import boltzgen as bg

temperature = 1000
system = AlanineDipeptideVacuum(constraints=None)

sim = app.Simulation(system.topology, system.system,
                        mm.LangevinIntegrator(temperature * unit.kelvin,
                                            1. / unit.picosecond,
                                            1. * unit.femtosecond),
                        mm.Platform.getPlatformByName('CPU'))


if __name__ == '__main__':
    #target_dist = bg.distributions.Boltzmann(
    from TargetDistributions.AladineDipeptide.BG_utils import Boltzmann
    target_dist = Boltzmann(
        sim.context, temperature,
        energy_cut=1e7,
        energy_max=1.e20)
    def target_log_prob(x):
        batch_size = x.shape[0]
        x = x.reshape(batch_size, 22, 3)
        return target_dist.log_prob(x)
    print(target_log_prob(torch.randn(10, 66)))

    from TargetDistributions.AladineDipeptide.gradient_descent import grad_descent_search

    #grad_descent_search(target_log_prob=target_log_prob, n_points=10)
    grad_descent_search(target_log_prob=lambda x: target_dist.log_prob(x), shape=(22, 3), n_points=10)