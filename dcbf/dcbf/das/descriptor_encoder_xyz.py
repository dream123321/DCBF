import os
from dscribe.descriptors import ACSF,SOAP,MBTR
from ase import Atoms
from ase.io import iread,write,read
import matplotlib.pyplot as plt
from maml.sampling.direct import DIRECTSampler, BirchClustering, SelectKFromClusters, M3GNetStructure
import time
import matplotlib.ticker as mtick
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')


def _patch_dscribe_system_init_for_new_ase():
    from dscribe.core.system import System as DscribeSystem

    if getattr(DscribeSystem, "_dcbf_keyword_init_patch", False):
        return

    def _new_init(
        self,
        symbols=None,
        positions=None,
        numbers=None,
        tags=None,
        momenta=None,
        masses=None,
        magmoms=None,
        charges=None,
        scaled_positions=None,
        cell=None,
        pbc=None,
        celldisp=None,
        constraint=None,
        calculator=None,
        info=None,
        wyckoff_positions=None,
        equivalent_atoms=None,
    ):
        Atoms.__init__(
            self,
            symbols=symbols,
            positions=positions,
            numbers=numbers,
            tags=tags,
            momenta=momenta,
            masses=masses,
            magmoms=magmoms,
            charges=charges,
            scaled_positions=scaled_positions,
            cell=cell,
            pbc=pbc,
            celldisp=celldisp,
            constraint=constraint,
            calculator=calculator,
            info=info,
        )
        self.wyckoff_positions = wyckoff_positions
        self.equivalent_atoms = equivalent_atoms
        self._cell_inverse = None
        self._displacement_tensor = None
        self._distance_matrix = None
        self._inverse_distance_matrix = None

    DscribeSystem.__init__ = _new_init
    DscribeSystem._dcbf_keyword_init_patch = True


def _sanitize_atoms_for_dscribe(atoms_list):
    clean = []
    for atoms in atoms_list:
        clean.append(
            Atoms(
                symbols=atoms.get_chemical_symbols(),
                positions=atoms.get_positions(),
                cell=atoms.get_cell(),
                pbc=atoms.get_pbc(),
            )
        )
    return clean


def _resolve_dscribe_n_jobs():
    raw = os.environ.get("DCBF_MBTR_N_JOBS", "1").strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid DCBF_MBTR_N_JOBS value: {raw!r}") from exc
    if value == 0:
        raise ValueError("DCBF_MBTR_N_JOBS cannot be 0")
    return value


class sample_plot():
    def plot_PCAfeature_coverage(self,all_features, selected_indexes, method="Effectively"):
        #fig, ax = plt.subplots(figsize=(6, 6))
        selected_features = all_features[selected_indexes]
        if all_features.shape[1] > 1:
            plt.plot(all_features[:, 0], all_features[:, 1], "*", alpha=0.5, label=f"All {len(all_features):,} structures")
            plt.plot(
                selected_features[:, 0],
                selected_features[:, 1],
                "*",
                alpha=0.5,
                label=f"{method} sampled {len(selected_features):,}",
            )
            legend = plt.legend(frameon=False, fontsize=14, loc="upper left", bbox_to_anchor=(-0.02, 1.02), reverse=True)
            for lh in legend.legend_handles:
                lh.set_alpha(1)
            plt.ylabel("PC 2", size=20)
            plt.xlabel("PC 1", size=20)
            plt.savefig('cover_fig.png',dpi=300)
            #plt.show()

    def calculate_feature_coverage_score(self,all_features, selected_indexes, n_bins=100):
        selected_features = all_features[selected_indexes]
        n_all = np.count_nonzero(
            np.histogram(all_features, bins=np.linspace(min(all_features), max(all_features), n_bins))[0]
        )
        n_select = np.count_nonzero(
            np.histogram(selected_features, bins=np.linspace(min(all_features), max(all_features), n_bins))[0]
        )
        return n_select / n_all

    def calculate_all_FCS(self,all_features, selected_indexes, b_bins=100):
        select_scores = [
            self.calculate_feature_coverage_score(all_features[:, i], selected_indexes, n_bins=b_bins)
            for i in range(all_features.shape[1])
        ]
        return select_scores

    def sample_plot(self,vectors,f_in,n=1000,threshold_init=0.15,k=10,b_bins=100,sample_out=None,remaining_out=None):

        DIRECT_sampler = DIRECTSampler(
            structure_encoder=None,
            clustering=BirchClustering(n=n, threshold_init=threshold_init),
            select_k_from_clusters=SelectKFromClusters(k=k),
        )
        #DIRECT_selection = DIRECT_sampler.fit_transform(vectors)
        #print(DIRECT_selection["PCAfeatures"].shape)
        t1 = DIRECT_sampler.named_steps['StandardScaler'].fit_transform(vectors)
        t2 = DIRECT_sampler.named_steps['PrincipalComponentAnalysis'].fit_transform(t1)
        temp = DIRECT_sampler.named_steps['BirchClustering']
        t3 = temp.fit_transform(t2)
        DIRECT_selection = DIRECT_sampler.named_steps['SelectKFromClusters'].fit_transform(t3)
        print(
            f"DIRECT selected {len(DIRECT_selection['selected_indexes'])} structures from {len(DIRECT_selection['PCAfeatures'])} structures in data.")
        explained_variance = DIRECT_sampler.pca.pca.explained_variance_ratio_
        DIRECT_selection["PCAfeatures_unweighted"] = DIRECT_selection["PCAfeatures"] / explained_variance[:len(DIRECT_selection["PCAfeatures"][0])]

        # plt.plot(
        #     range(1, 31),
        #     explained_variance[:30] * 100,
        #     "o-",
        # )
        # plt.xlabel("i$^{\mathrm{th}}$ PC", size=20)
        # plt.ylabel("Explained variance", size=20)
        # ax = plt.gca()
        # ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        #plt.show()
        all_features = DIRECT_selection["PCAfeatures_unweighted"]
        selected_indexes = DIRECT_selection["selected_indexes"]
        self.plot_PCAfeature_coverage(all_features, selected_indexes)
        all_features = DIRECT_selection["PCAfeatures_unweighted"]
        scores_MPF_DIRECT = self.calculate_all_FCS(all_features, DIRECT_selection["selected_indexes"], b_bins=b_bins)
        # scores_MPF_MS = self.calculate_all_FCS(all_features, manual_selection_index, b_bins=100)

        x = np.arange(len(scores_MPF_DIRECT))
        x_ticks = [f"PC {n + 1}" for n in range(len(x))]

        plt.figure(figsize=(15, 4))
        plt.bar(
            x + 0.45,
            scores_MPF_DIRECT,
            width=0.3,
            label=f"DIRECT, $\overline{{\mathrm{{Coverage\ score}}}}$ = {np.mean(scores_MPF_DIRECT):.3f}",
        )
        # plt.bar(
        #     x + 0.3,
        #     scores_MPF_MS,
        #     width=0.3,
        #     label=f"Manual, $\overline{{\mathrm{{Coverage\ score}}}}$ = {np.mean(scores_MPF_MS):.3f}",
        # )
        plt.xticks(x + 0.45, x_ticks, size=16)
        plt.yticks(np.linspace(0, 1.0, 6), size=16)
        plt.ylabel("Coverage score", size=20)
        plt.legend(shadow=True, loc="upper right", fontsize=16)
        #plt.show()

        atoms_list = list(iread(f_in))

        '''采样的数据存下来'''
        new_list = []
        for i in selected_indexes:
            new_list.append(atoms_list[i])
        rename_sample = str(len(selected_indexes)) + '_' + sample_out
        #write(rename_sample,new_list,format='extxyz')


        '''剩余的数据存下来'''
        remain_indexes = []
        remaining_data = []
        #print(all_features)
        #print(selected_indexes)
        for i in range(len(all_features)):
            if i not in selected_indexes:
                remaining_data.append(atoms_list[i])
                remain_indexes.append(i)
        rename_remaining = str(len(remain_indexes)) + '_' + remaining_out
        #write(rename_remaining, remaining_data, format='extxyz')

        return np.mean(scores_MPF_DIRECT),selected_indexes,remain_indexes,new_list,remaining_data,t3['labels'],temp.threshold_init


class encoder():
    def __init__(self, f_in, save_path):
        self.f_in = f_in
        self.save_path = save_path

    def save(self, data, filename):
        with open(filename, 'wb') as file:
            pickle.dump(data, file)

    def MBTR(self):
        atoms_list = list(iread(self.f_in))
        clean_atoms_list = _sanitize_atoms_for_dscribe(atoms_list)
        species = set(atoms_list[0].get_chemical_symbols())
        _patch_dscribe_system_init_for_new_ase()
        n_jobs = _resolve_dscribe_n_jobs()

        desc = MBTR(
            species=species,
            periodic=True,
            geometry={"function": "inverse_distance"},
            grid={"min": 0, "max": 1.0, "sigma": 0.02, "n": 50},
            weighting={"function": "exp", "r_cut": 7, "threshold": 1e-3},
            sparse=False,
            normalization="l2",
        )

        vectors = desc.create(clean_atoms_list, n_jobs=n_jobs)
        self.save(vectors, self.save_path)
        print(f'MBTR.shape = {vectors.shape}')
        return vectors
















