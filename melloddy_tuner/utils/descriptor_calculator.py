import json
from math import nan
from melloddy_tuner.utils.helper import int_to_sha256
from typing import Dict, Tuple
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.AllChem import GetHashedMorganFingerprint, GetMorganFingerprint
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

import numpy as np
import copy
import logging


class DescriptorCalculator(object):
    """
    Wrapper class to calculate molecular descriptors (fingerprints).


    """

    def __init__(
        self,
        radius: int,
        hashed: bool,
        fold_size: int,
        binarized: bool,
        secret: str,
        verbosity: int = 0,
        desc_type: str = "ecfp",
    ):
        self.desc_type = desc_type
        self.radius = radius
        self.hashed = hashed
        self.size = fold_size
        self.secret = secret
        self.binarize = binarized
        self.verbosity = verbosity
        if self.desc_type.lower() == "ecfp":
            self.permuted_ix = DescriptorCalculator.set_permutation(
                size=self.size, key=self.secret
            )
        self.rdkit_all_descriptors = [x[0] for x in Descriptors._descList]
        self.rdkit_custom_descriptors = ['MaxEStateIndex', 'MinEStateIndex', 'MaxAbsEStateIndex', 'MinAbsEStateIndex',
        'qed', 'MolWt', 'HeavyAtomMolWt', 'ExactMolWt', 'NumValenceElectrons',
        'NumRadicalElectrons', 'MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge',
        'MinAbsPartialCharge', 'FpDensityMorgan1', 'FpDensityMorgan2', 'FpDensityMorgan3',
        'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v',
        'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Ipc',
        'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12',
        'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3',
        'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7',
        'PEOE_VSA8', 'PEOE_VSA9', 'SMR_VSA1', 'SMR_VSA10',
        'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5',
        'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9',
        'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11', 'SlogP_VSA12',
        'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5',
        'SlogP_VSA6', 'SlogP_VSA7', 'SlogP_VSA8', 'SlogP_VSA9',
        'TPSA', 'EState_VSA1', 'EState_VSA10', 'EState_VSA11',
        'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5',
        'EState_VSA6', 'EState_VSA7', 'EState_VSA8', 'EState_VSA9',
        'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3',
        'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7',
        'VSA_EState8', 'VSA_EState9', 'FractionCSP3', 'HeavyAtomCount',
        'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles', 'NumAliphaticRings',
        'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors',
        'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds', 'NumSaturatedCarbocycles',
        'NumSaturatedHeterocycles', 'NumSaturatedRings', 'RingCount', 'MolLogP',
        'MolMR']
        


    # functions enabling pickle
    def __getstate__(self):
        return (
            self.radius,
            self.hashed,
            self.size,
            self.binarize,
            self.secret,
            self.verbosity,
            self.desc_type
        )

    def __setstate__(self, state):
        self.__init__(*state)

    @classmethod
    def from_param_dict(cls, secret, method_param_dict, verbosity=0):
        """Function to create and initialize a SaccolFoldAssign Calculator

        Args:
            secret: secret key used (for fold hashing)
            verbosity (int): controlls verbosity
            par_dict(dict): dictionary of method parameters
        """
        return cls(secret=secret, **method_param_dict, verbosity=verbosity)

    def set_permutation(size: int, key: str):
        original_ix = np.arange(size)
        hashed_ix = np.array(
            [
                int.from_bytes(int_to_sha256(j, key), "big") % 2 ** 63
                for j in original_ix
            ]
        )
        permuted_ix = hashed_ix.argsort().argsort()
        return permuted_ix

    def make_scrambled_lists(self, fp_feat_arr: np.array) -> list:
        """Pseudo-random scrambling with secret.

        Args:
            fp_list (list): fingerprint list
            secret (str): secret key
            bitsize (int): bitsize (shape)

        Returns:
            list: scrambled list
        """
        original_ix = np.arange(self.size)

        scrambled = []
        if (np.sort(self.permuted_ix) == original_ix).all():
            for x in fp_feat_arr:
                scrambled.append(int(self.permuted_ix[x]))
        else:
            print("Check index permutation failed.")
        return np.array(scrambled)

    def get_fp(self, smiles):
        mol_fp = {}

        mol = MolFromSmiles(smiles)  # Read SMILES and convert it to RDKit mol object.

        if self.desc_type.lower() == "ecfp":
            if self.hashed is True:
                try:
                    mol_fp = GetHashedMorganFingerprint(
                        mol, self.radius, self.size
                    ).GetNonzeroElements()
                except (ValueError, AttributeError) as e:
                    return None, False, str(e)
                return mol_fp
            else:
                try:
                    mol_fp = GetMorganFingerprint(
                        mol, self.radius, self.size
                    ).GetNonzeroElements()
                except (ValueError, AttributeError) as e:
                    return None, False, str(e)
                return mol_fp
        elif self.desc_type.lower() == "rdkit_all":
            calc = MolecularDescriptorCalculator(self.rdkit_all_descriptors)
            return calc.CalcDescriptors(mol)
        elif self.desc_type.lower() == "rdkit_custom":
            calc = MolecularDescriptorCalculator(self.rdkit_custom_descriptors)
            return calc.CalcDescriptors(mol)

    def scramble_fp(self, mol_fp):
        fp_feat = np.array(list(mol_fp.keys()))
        fp_val = np.array(list(mol_fp.values()))
        fp_feat_scrambled = DescriptorCalculator.make_scrambled_lists(self, fp_feat)
        if self.binarize:
            fp_val.fill(int(1))

        return fp_feat_scrambled.tolist(), fp_val.tolist()

    @staticmethod
    def fp_to_json(fp_feat, fp_val):
        fp_feat_json = json.dumps(fp_feat)
        fp_val_json = json.dumps(fp_val)
        return fp_feat_json, fp_val_json

    def calculate_single(self, smiles: str) -> Tuple:
        """
        Calculation of Morgan fingerprints (ECFP equivalent) with a given radius

        Args:
            smi (str): SMILES string

        Returns:
            Tuple(np.array(list), np.array(list)): NumPy arrays of fingerprint feature list, fingerprint value list
        """

        try:
            mol_fp = self.get_fp(smiles)
        except ValueError as err:
            return None, None, False, str(err)
        
        if self.desc_type.lower() == "ecfp":
            try:
                fp_feat_scrambled, fp_val_binarized = self.scramble_fp(mol_fp)
                fp_feat, fp_val = self.fp_to_json(fp_feat_scrambled, fp_val_binarized)
            except ValueError as err:
                return None, None, False, str(err)

            return fp_feat, fp_val, True, None
        
        else:
            return mol_fp, mol_fp, True, None
        
