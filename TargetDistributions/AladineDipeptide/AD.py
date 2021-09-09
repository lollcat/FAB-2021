import torch
import torch.nn as nn
import numpy as np
import sys
import mdtraj as md

#sys.path.insert(0, '../../normalizing-flows')
import normflow as nf

from simtk import openmm as mm
from simtk import unit
from simtk.openmm import app
from openmmtools import testsystems
from simtk.openmm.app import StateDataReporter
import mdtraj

sys.path.insert(0, '../')
import boltzgen as bg

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--processID', type=int, default=0)
    args = parser.parse_args()

    # log mass 0 stepsize 0.05
    ndim = 66
    z_matrix = [
        (0, [1, 4, 6]),
        (1, [4, 6, 8]),
        (2, [1, 4, 0]),
        (3, [1, 4, 0]),
        (4, [6, 8, 14]),
        (5, [4, 6, 8]),
        (7, [6, 8, 4]),
        (11, [10, 8, 6]),
        (12, [10, 8, 11]),
        (13, [10, 8, 11]),
        (15, [14, 8, 16]),
        (16, [14, 8, 6]),
        (17, [16, 14, 15]),
        (18, [16, 14, 8]),
        (19, [18, 16, 14]),
        (20, [18, 16, 19]),
        (21, [18, 16, 19])
    ]
    cart_indices = [6, 8, 9, 10, 14]
    temperature = 1000

    system = testsystems.AlanineDipeptideVacuum(constraints=None)
    sim = app.Simulation(system.topology, system.system,
                         mm.LangevinIntegrator(temperature * unit.kelvin,
                                               1. / unit.picosecond,
                                               1. * unit.femtosecond),
                         mm.Platform.getPlatformByName('CPU'))

    # Load the training data
    #training_data_traj = mdtraj.load('saved_data/aldp_vacuum_without_const.h5')
    ala2_pdb = md.load('TargetDistributions/AladineDipeptide/data/alanine-dipeptide.pdb').topology
    training_data_traj = md.load('TargetDistributions/AladineDipeptide/data/ala2_1000K_train.xtc', top=ala2_pdb)
    training_data_traj.center_coordinates()
    ind = training_data_traj.top.select("backbone")
    training_data_traj.superpose(training_data_traj, 0, atom_indices=ind,
                                 ref_atom_indices=ind)
    # Gather the training data into a pytorch Tensor with the right shape
    training_data = training_data_traj.xyz
    n_atoms = training_data.shape[1]
    n_dim = n_atoms * 3
    training_data_npy = training_data.reshape(-1, n_dim)
    training_data = torch.from_numpy(
        training_data_npy.astype("float64"))

    coord_transform = bg.flows.CoordinateTransform(training_data,
                                                   n_dim, z_matrix, cart_indices)
    target_dist = bg.distributions.TransformedBoltzmannParallel(
        system, temperature,
        energy_cut=5.e+2,
        energy_max=1.e+20,
        transform=coord_transform,
        n_threads=3)
    print(f"sample log_probs from training: {target_dist.log_prob(coord_transform.inverse(training_data[0:20])[0])} \n")
    print(f"random log_probs {target_dist.log_prob(torch.randn(5, 60).double()*0.2)} \n ")
    target_dist.dim = 60

    def performance_metrics(*args, **kwargs):
        return {}, {} # dummy function
    target_dist.performance_metrics = performance_metrics
    return target_dist

if __name__ == '__main__':
    target_dist = main()
    torch.set_default_dtype(torch.float64)
    # test gradient descent
    from TargetDistributions.AladineDipeptide.gradient_descent import grad_descent_search
    model = grad_descent_search(target_log_prob=lambda x: target_dist.log_prob(x), shape=(60,), n_points=20,
                                epochs=500, per_print=10)
    # test HMC
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    from ImportanceSampling.SamplingAlgorithms.HamiltonianMonteCarlo import HMC
    hmc = HMC(n_distributions=3, dim=target_dist.dim)

    p_accept = []
    epsilons = []
    sampler_samples = torch.randn((1000, 66)) * 0.2
    print(sampler_samples[0][0:5])
    x_HMC = torch.clone(sampler_samples)
    for _ in tqdm(range(10)):
        x_HMC = hmc.run(x_HMC, lambda x: target_dist.log_prob(x), 0)
        interesting_info = hmc.interesting_info()
        p_accept.append(interesting_info['dist1_p_accept_0'])
        epsilons.append(interesting_info[f"epsilons_dist0_loop0"])

    plt.plot(p_accept)
    plt.title("p_accept")
    plt.show()
    plt.plot(epsilons)
    plt.title("common epsilons")
    plt.show()