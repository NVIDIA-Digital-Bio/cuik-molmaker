# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from rdkit import Chem
from rdkit.Chem import Descriptors
import argparse
import pandas as pd
import numpy as np
import time
from cuik_molmaker.utils.descriptor_normalization import get_normalization_functions
import json
import pkg_resources
from cuik_molmaker.utils.descriptor_normalization import DESCRIPTASTORUS_DESC_LIST



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to the file containing SMILES strings")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output file")
    parser.add_argument("--normalization_params", type=str, choices=("best", "fast", "descriptastorus"), help="Path to the JSON file containing normalization parameters")
    parser.add_argument("--smiles_column", type=str, required=True, help="Name of the column containing SMILES strings")
    parser.add_argument("--rdkit2D_descriptors_nan_sub_value", type=float, help="Value to substitute for NaN values in rdkit2D descriptors")
    return parser.parse_args()  


class MoleculeFeaturizer:
    def __init__(self, molecular_descriptor_type: str, **kwargs):

        if molecular_descriptor_type == "rdkit2D":
            self.molecular_descriptor_type = "rdkit2D"

            # Set up rdkit2D descriptors params
            all_desc_fn_dict = {desc[0]: desc[1] for desc in Descriptors._descList}
            if "rdkit2D_descriptor_list" in kwargs:
                self.rdkit2D_descriptor_list = [(desc,all_desc_fn_dict[desc]) for desc in kwargs["rdkit2D_descriptor_list"]]
            else:
                # Use all rdkit2D descriptors
                self.rdkit2D_descriptor_list = Descriptors._descList

            if "rdkit2D_normalization_type" in kwargs:
                self.rdkit2D_normalization_type = kwargs["rdkit2D_normalization_type"]
                assert self.rdkit2D_normalization_type in ("best", "fast", "descriptastorus", None)
            else:
                self.rdkit2D_normalization_type = None

            if self.rdkit2D_normalization_type is not None:
                with open(pkg_resources.resource_filename('cuik_molmaker', f'data/{self.rdkit2D_normalization_type}_normalization_params.json')) as f:
                    norm_params = json.load(f)
                rdkit2D_desc_name_list = [d[0] for d in self.rdkit2D_descriptor_list]
                self.rdkit2D_normalization_fn_dict = get_normalization_functions(rdkit2D_desc_name_list, norm_params)
    
        else:
            raise ValueError(f"Invalid molecular descriptor type: {molecular_descriptor_type}. Only 'rdkit2D' is supported.")
        
        if "rdkit2D_descriptors_nan_sub_value" in kwargs:
            self.rdkit2D_descriptors_nan_sub_value = kwargs["rdkit2D_descriptors_nan_sub_value"]
        else:
            self.rdkit2D_descriptors_nan_sub_value = None
        
        

    def _compute_rdkit2D_descriptors(self, smi: str) -> np.ndarray:

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return [np.nan] * len(self.rdkit2D_descriptor_list)
        return [func[1](mol) for func in self.rdkit2D_descriptor_list]
    
    def compute_rdkit2D_descriptors(self, smi_list: list[str]) -> np.ndarray:

        desc = [self._compute_rdkit2D_descriptors(smi) for smi in smi_list]
        desc = np.array(desc)
        if self.rdkit2D_normalization_type is not None:
            desc = self._normalize_rdkit2D_descriptors(desc)
        
        # Set nans to a value
        if self.rdkit2D_descriptors_nan_sub_value is not None:
            desc[np.isnan(desc)] = self.rdkit2D_descriptors_nan_sub_value
        
        return desc

    def featurize(self, smi_list: list[str]) -> np.ndarray:
        return self.compute_rdkit2D_descriptors(smi_list)

    def _normalize_rdkit2D_descriptors(self, desc: np.ndarray) -> np.ndarray:

        norm_desc = np.zeros_like(desc, dtype=np.float64)
        for i, desc_name in enumerate(self.rdkit2D_descriptor_list):
            norm_fn = self.rdkit2D_normalization_fn_dict[desc_name[0]]
            norm_desc[:, i] = norm_fn(desc[:, i])

        return norm_desc

if __name__ == "__main__":
    args = parse_args()
    smi_list = pd.read_csv(args.input_file)[args.smiles_column].to_list()

    desc_name_fn_list = Descriptors._descList
    if args.normalization_params == "descriptastorus":
        print(f"Using {len(DESCRIPTASTORUS_DESC_LIST)} descriptors from Descriptastorus package as descriptastorus normalization parameters is specified.")
        desc_name_list = DESCRIPTASTORUS_DESC_LIST
    else:
        desc_name_list = [d[0] for d in desc_name_fn_list]
    desc_start_time = time.time()
    featurizer = MoleculeFeaturizer(molecular_descriptor_type="rdkit2D", rdkit2D_descriptor_list=desc_name_list, rdkit2D_normalization_type=args.normalization_params)
    desc = featurizer.featurize(smi_list)
    desc_end_time = time.time()
    print(f"Descriptor computation time: {(desc_end_time - desc_start_time):.4f}s")

    output_df = pd.DataFrame(desc, columns=desc_name_list)

    output_df.to_csv(args.output_file, index=False)