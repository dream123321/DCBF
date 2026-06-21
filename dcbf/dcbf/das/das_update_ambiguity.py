import os

import numpy as np
from typing import Optional, Union, List
from ase import Atoms
from pymlip.core import MTPCalactor,PyConfiguration
from ase.calculators.calculator import Calculator
from ase.data import atomic_numbers as at_nu
from ase.io import iread
import os
import glob
import yaml
import pickle
from ..path_names import SUS2_MODEL_DIR

class MTPCalculator(Calculator):
    """
    MTP calculator based on ase Calculator
    """

    implemented_properties = ["energy",  "forces", "energies" ,"stress"]

    def __init__(self, potential: str = "p.mtp", mtpcalc: MTPCalactor = None, unique_numbers: List = None ,compute_stress: bool = True, stress_weight: float = 1.0,print_EK: bool = True, **kwargs):
        """

        Args:
            potential (str): xxx.mtp
            compute_stress (bool): whether to calculate the stress
            stress_weight (float): the stress weight.
            **kwargs:
        """
        super().__init__(**kwargs)
        self.potential = potential   ##  xxx.mtp
        self.compute_stress = compute_stress
        self.print_EK = print_EK
        self.stress_weight = stress_weight
        self.mtpcalc = MTPCalactor(self.potential)
        self.unique_numbers=unique_numbers

    def calculate(
        self,
        atoms: Optional[Atoms] = None,
        properties: Optional[list] = None,
        system_changes: Optional[list] = None,
        unique_numbers: Optional[list] = None
    ):
        """
        Args:
            atoms (ase.Atoms): ase Atoms object
            properties (list): list of properties to calculate
            system_changes (list): monitor which properties of atoms were
                changed for new calculation. If not, the previous calculation
                results will be loaded.
        Returns:

        """
        properties = properties or ["energy"]
        system_changes = system_changes or all_changes
        super().calculate(atoms=atoms, properties=properties, system_changes=system_changes)

        # graph = self.potential.graph_converter(atoms)
        # graph_list = graph.as_tf().as_list()
        # results = self.potential.get_efs_tensor(graph_list, include_stresses=self.compute_stress)
        cfg=PyConfiguration.from_ase_atoms(atoms,unique_numbers=self.unique_numbers)
       # print(cfg.types)
        V=atoms.cell.volume
        self.mtpcalc.calc(cfg)
        self.results['energy'] = np.array(cfg.energy)
        self.results['forces'] = cfg.force
        self.results['energies'] = np.array(cfg.site_energys)
        #print(f"PE:{np.array(cfg.energy):.4f} (ev)")

        if self.compute_stress:
            self.results['stress'] = -np.array([cfg.stresses[0,0],cfg.stresses[1,1],cfg.stresses[2,2],cfg.stresses[1,2],cfg.stresses[0,2],cfg.stresses[0,1]])* self.stress_weight/V

def get_force_ambiguity(force_set):
    force_set_sub_mean = force_set - np.mean(force_set, axis=0)
    force_set_sub_mean_norm = np.sum(force_set_sub_mean ** 2, axis=2)
    ee_list = np.sqrt(np.mean(force_set_sub_mean_norm, axis=0))
    # print(f"max/min/mean: {np.max(ee_list):.3f} {np.min(ee_list):.3f} {np.mean(ee_list):.3f}")
    # return np.max(ee_list)
    return ee_list


def calc_ensemble_ambiguity(atoms_list, model_fns, unique_numbers):
    calc_set = [MTPCalculator(potential=i, unique_numbers=unique_numbers) for i in model_fns]
    ee_list = []
    for i, atoms in enumerate(atoms_list):
        forces_set = []
        for calc in calc_set:
            atoms.calc = calc
            forces_set.append(atoms.get_forces())
        forces_set = np.array(forces_set, dtype=float)
        ee = get_force_ambiguity(forces_set)
        atomic_numbers = atoms.numbers
        ee_list_single = np.zeros(len(unique_numbers))
        for ii, uu in enumerate(unique_numbers):
            filter_ele_index = atomic_numbers == uu
            # if len(filter_ele_index) ==0, element uu not in this conf
            tt = sum(filter_ele_index)
            if tt == 0:
                filter_ele_index = []
            ee_list_single[ii] = np.max(ee[filter_ele_index]) if len(filter_ele_index) > 0 else 0.0
        ee_list.append(ee_list_single)

    #  ee_list NxM, N=# of confs in training set, M=# of unique elements
    ee_list = np.array(ee_list)
    return ee_list


def record_yaml(yaml_file,af_adaptive,select_stru_num):
    with open(yaml_file, 'rb') as file:
        data = yaml.safe_load(file)
    data['af_adaptive'] = af_adaptive
    data['select_stru_num'] = select_stru_num
    with open(yaml_file, 'w') as file:
        yaml.safe_dump(data, file, default_flow_style=False)

def af_limit_update(pwd,yaml_file):
    af_limit_list = os.path.join(os.path.dirname(pwd),'af_limit_list.pkl')
    if not os.path.exists(af_limit_list):
        with open(yaml_file, 'rb') as file:
            data = yaml.safe_load(file)
        temp = data['af_limit']
        return temp
    else:
        with open(af_limit_list, 'rb') as file:
            data = pickle.load(file)
        true_count = 0
        for item in data:
            if item:
                true_count += 1
        with open(yaml_file, 'rb') as file:
            data = yaml.safe_load(file)
        temp = data['af_limit']
        data['af_limit'] = temp * (true_count + 1)
        with open(yaml_file, 'w') as file:
            yaml.safe_dump(data, file, default_flow_style=False)
        return temp * (true_count + 1)

def af_limit_record(pwd,label):
    af_limit_list = os.path.join(os.path.dirname(pwd), 'af_limit_list.pkl')
    if not os.path.exists(af_limit_list):
        empty_list = []
        with open(af_limit_list, 'wb') as file:
            pickle.dump(empty_list, file)
    else:
        with open(af_limit_list, 'rb') as file:
            data_list = pickle.load(file)
        data_list.append(label)
        with open(af_limit_list, 'wb') as file:
            pickle.dump(data_list, file)

class das_update_ambiguity():
    def __init__(
        self,
        ele,
        sort_ele,
        logger,
        af_default=0.010,
        af_limit=0.200,
        af_failed=0.500,
        over_fitting_factor=1.1,
        threshold_limit_factor=1.1,
        bond_hierarchy=False,
        #**kwargs,
    ):

        if sort_ele:
            self.unique_numbers = sorted([at_nu[i] for i in ele])
        else:
            self.unique_numbers = [at_nu[i] for i in ele]
        self.logger = logger
        self.low_factor = over_fitting_factor
        self.bond_hierachy = bond_hierarchy
        self.threshold_limit_factor = threshold_limit_factor
        if self.bond_hierachy:
            if isinstance(af_default, list):
                if len(af_default) != self.unique_numbers:
                    self.logger.error("The setting of af_default is invalid")
                    sys.exit()
                self.af_default = af_default
            else:
                self.af_default = [af_default] * len(self.unique_numbers)

            if isinstance(af_limit, list):
                if len(af_limit) != self.unique_numbers:
                    self.logger.error("The setting of af_limit is invalid")
                self.af_limit = af_limit
            else:
                self.af_limit = [af_limit] * len(self.unique_numbers)

            if isinstance(af_failed, list):
                if len(af_failed) != self.unique_numbers:
                    self.logger.error("The setting of af_failed is invalid")
                self.af_failed = af_failed
            else:
                self.af_failed = [af_failed] * len(self.unique_numbers)
        else:
            self.af_default = af_default
            self.af_limit = af_limit
            self.af_failed = af_failed


    def run(self,select_conf,model_fns,gen_num):
        prev_select_conf = select_conf
        prev_model_fns = model_fns

        label = False
        if prev_select_conf is None:
            af_adaptive = self.af_default
        else:
            #prev_select_dataset = AtomicDataset.load(prev_select_conf)
            prev_select_dataset = list(iread(prev_select_conf))
            # if len(prev_select_dataset) < 1:
            #     af_adaptive = self.af_default
            #elif self.driver.status_record.iter_i == 0:
            #    af_adaptive = self.af_default
            if gen_num == 0:
                af_adaptive = self.af_default
            else:
                #pre_select_atoms = prev_select_dataset.ase_atoms
                # train_ee: (# of confs x # of elements)
                #train_ee = calc_ensemble_ambiguity(pre_select_atoms, prev_model_fns, self.unique_numbers)
                train_ee = calc_ensemble_ambiguity(prev_select_dataset, prev_model_fns, self.unique_numbers)
                af_adaptive = np.max(train_ee, axis=0) * self.low_factor
                if self.bond_hierachy:
                    # auto increase af_limit if need
                    for ii in range(len(self.unique_numbers)):
                        if af_adaptive[ii] > self.af_limit[ii]:
                            self.af_limit[ii] *= self.threshold_limit_factor
                            self.logger.info(
                                f"{af_adaptive[ii]} > {self.af_limit[ii] / self.threshold_limit_factor}, "
                                f"increase af_limit[{ii}] to {self.af_limit[ii]}"
                            )
                    for ii in range(len(self.unique_numbers)):
                        if af_adaptive[ii] > self.af_limit[ii]:
                            af_adaptive[ii] = self.af_limit[ii]
                else:
                    af_max = np.max(af_adaptive)
                    af_min = np.min(af_adaptive)
                    if af_max > af_min * 2:
                        self.logger.warning(
                            "The ambiguity of different elements is too large,"
                            "please consider setting bond_hierachy=True."
                        )
                    af_adaptive = np.max(af_adaptive)
                    if af_adaptive > self.af_limit:
                        self.af_limit *= self.threshold_limit_factor
                        self.logger.info(
                            f"{af_adaptive} > {self.af_limit / self.threshold_limit_factor}, increase "
                            f"af_limit to {self.af_limit}"
                        )
                    if af_adaptive > self.af_limit:
                        af_adaptive = self.af_limit
                        label = True
        self.logger.info(f"The ambiguity threshold: {af_adaptive}")
        return af_adaptive,label

if __name__ == '__main__':
    pwd = os.getcwd()
    mtp = os.path.join(pwd, SUS2_MODEL_DIR)
    xyz = 'scf_filter.xyz'
    model_fns = [os.path.join(mtp,a) for a in glob.glob(os.path.join(mtp,'current*'))]
    ele = list(set(list(iread(xyz))[0].get_chemical_symbols()))
    sort_ele = True
    gen_num = 0
    logger = ''
    af_default = 0.010
    af_limit = 0.200
    af_failed = 0.500
    over_fitting_factor = 1.1
    das_update_ambiguity(ele=ele,sort_ele=sort_ele,af_default=af_default,af_limit=af_limit,af_failed=af_failed,over_fitting_factor=over_fitting_factor,logger=logger).run(xyz,model_fns,gen_num)
