from openmmtools.testsystems import AlanineDipeptideVacuum
from bgtorch.utils.train import linlogcut
from bgtorch.distribution.energy import Energy
import mdtraj as md
import torch

ala2 = AlanineDipeptideVacuum(constraints=None)

temperature = 1000
n_atoms = ala2.mdtraj_topology.n_atoms

class RegularizedEnergy(Energy):
    def __init__(self, energy_model, high_energy, max_energy):
        super().__init__(energy_model.dim)
        self.energy_model = energy_model
        self.high_energy = high_energy
        self.max_energy = max_energy

    def _energy(self, x, temperature=None):
        U = self.energy_model.energy(x)
        U_reg = linlogcut(U, high_val=self.high_energy, max_val=self.max_energy)
        return U_reg


ala2_pdb = md.load('TargetDistributions/AladineDipeptide/data/alanine-dipeptide.pdb').topology
training_data_traj = md.load('TargetDistributions/AladineDipeptide/data/ala2_1000K_train.xtc', top=ala2_pdb)

training_data = training_data_traj.xyz.reshape((training_data_traj.xyz.shape[0],
                                                3*training_data_traj.xyz.shape[1]))
training_data = torch.tensor(training_data)

from bgtorch.distribution.energy.openmm import OpenMMEnergy, OpenMMEnergyBridge
from simtk import openmm, unit
INTEGRATOR_ARGS = (temperature*unit.kelvin,
                   1.0/unit.picoseconds,
                   1.0*unit.femtoseconds)
ala2_omm_energy_bridge = OpenMMEnergyBridge(ala2.system, unit.nanometers,
                                            openmm_integrator=openmm.LangevinIntegrator,
                                            openmm_integrator_args=INTEGRATOR_ARGS)

ala2_omm_energy = OpenMMEnergy(3 * n_atoms, ala2_omm_energy_bridge)


# Reguarlize molecular energy
ala2_energy_reg = RegularizedEnergy(ala2_omm_energy, high_energy=500, max_energy=int(1e8)) #10000)







if __name__ == '__main__':
    #from TargetDistributions.AladineDipeptide.gradient_descent import grad_descent_search
    #grad_descent_search(ala2_energy_reg)
    """
    from ImportanceSampling.SamplingAlgorithms.HamiltonianMonteCarlo import HMC
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    dim = ala2_energy_reg.dim
    #target_log_prob = lambda x: -torch.squeeze(ala2_energy_reg.energy(x))
    target_log_prob = lambda x: -torch.squeeze(ala2_omm_energy.energy(x))

    hmc = HMC(n_distributions=3, dim=dim)

    p_accept = []
    epsilons = []
    sampler_samples = torch.randn((1000, 66)) * 0.2
    print(sampler_samples[0][0:5])
    x_HMC = torch.clone(sampler_samples)
    for _ in tqdm(range(10)):
        x_HMC = hmc.run(x_HMC, target_log_prob, 0)
        interesting_info = hmc.interesting_info()
        p_accept.append(interesting_info['dist1_p_accept_0'])
        epsilons.append(interesting_info[f"epsilons_dist0_loop0"])

    plt.plot(p_accept)
    plt.title("p_accept")
    plt.show()
    plt.plot(epsilons)
    plt.title("common epsilons")
    plt.show()
    """



