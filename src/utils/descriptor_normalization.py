# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. # noqa: E501
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict, List

import numpy as np
import scipy.stats as stats

# This is a list of RDKit2D descriptors that are available in Descriptastorus package
DESCRIPTASTORUS_DESC_LIST = [
    "BalabanJ",
    "BertzCT",
    "Chi0",
    "Chi0n",
    "Chi0v",
    "Chi1",
    "Chi1n",
    "Chi1v",
    "Chi2n",
    "Chi2v",
    "Chi3n",
    "Chi3v",
    "Chi4n",
    "Chi4v",
    "EState_VSA1",
    "EState_VSA10",
    "EState_VSA11",
    "EState_VSA2",
    "EState_VSA3",
    "EState_VSA4",
    "EState_VSA5",
    "EState_VSA6",
    "EState_VSA7",
    "EState_VSA8",
    "EState_VSA9",
    "ExactMolWt",
    "FpDensityMorgan1",
    "FpDensityMorgan2",
    "FpDensityMorgan3",
    "FractionCSP3",
    "HallKierAlpha",
    "HeavyAtomCount",
    "HeavyAtomMolWt",
    "Ipc",
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "LabuteASA",
    "MaxAbsEStateIndex",
    "MaxAbsPartialCharge",
    "MaxEStateIndex",
    "MaxPartialCharge",
    "MinAbsEStateIndex",
    "MinAbsPartialCharge",
    "MinEStateIndex",
    "MinPartialCharge",
    "MolLogP",
    "MolMR",
    "MolWt",
    "NHOHCount",
    "NOCount",
    "NumAliphaticCarbocycles",
    "NumAliphaticHeterocycles",
    "NumAliphaticRings",
    "NumAromaticCarbocycles",
    "NumAromaticHeterocycles",
    "NumAromaticRings",
    "NumHAcceptors",
    "NumHDonors",
    "NumHeteroatoms",
    "NumRadicalElectrons",
    "NumRotatableBonds",
    "NumSaturatedCarbocycles",
    "NumSaturatedHeterocycles",
    "NumSaturatedRings",
    "NumValenceElectrons",
    "PEOE_VSA1",
    "PEOE_VSA10",
    "PEOE_VSA11",
    "PEOE_VSA12",
    "PEOE_VSA13",
    "PEOE_VSA14",
    "PEOE_VSA2",
    "PEOE_VSA3",
    "PEOE_VSA4",
    "PEOE_VSA5",
    "PEOE_VSA6",
    "PEOE_VSA7",
    "PEOE_VSA8",
    "PEOE_VSA9",
    "RingCount",
    "SMR_VSA1",
    "SMR_VSA10",
    "SMR_VSA2",
    "SMR_VSA3",
    "SMR_VSA4",
    "SMR_VSA5",
    "SMR_VSA6",
    "SMR_VSA7",
    "SMR_VSA8",
    "SMR_VSA9",
    "SlogP_VSA1",
    "SlogP_VSA10",
    "SlogP_VSA11",
    "SlogP_VSA12",
    "SlogP_VSA2",
    "SlogP_VSA3",
    "SlogP_VSA4",
    "SlogP_VSA5",
    "SlogP_VSA6",
    "SlogP_VSA7",
    "SlogP_VSA8",
    "SlogP_VSA9",
    "TPSA",
    "VSA_EState1",
    "VSA_EState10",
    "VSA_EState2",
    "VSA_EState3",
    "VSA_EState4",
    "VSA_EState5",
    "VSA_EState6",
    "VSA_EState7",
    "VSA_EState8",
    "VSA_EState9",
    "fr_Al_COO",
    "fr_Al_OH",
    "fr_Al_OH_noTert",
    "fr_ArN",
    "fr_Ar_COO",
    "fr_Ar_N",
    "fr_Ar_NH",
    "fr_Ar_OH",
    "fr_COO",
    "fr_COO2",
    "fr_C_O",
    "fr_C_O_noCOO",
    "fr_C_S",
    "fr_HOCCN",
    "fr_Imine",
    "fr_NH0",
    "fr_NH1",
    "fr_NH2",
    "fr_N_O",
    "fr_Ndealkylation1",
    "fr_Ndealkylation2",
    "fr_Nhpyrrole",
    "fr_SH",
    "fr_aldehyde",
    "fr_alkyl_carbamate",
    "fr_alkyl_halide",
    "fr_allylic_oxid",
    "fr_amide",
    "fr_amidine",
    "fr_aniline",
    "fr_aryl_methyl",
    "fr_azide",
    "fr_azo",
    "fr_barbitur",
    "fr_benzene",
    "fr_benzodiazepine",
    "fr_bicyclic",
    "fr_diazo",
    "fr_dihydropyridine",
    "fr_epoxide",
    "fr_ester",
    "fr_ether",
    "fr_furan",
    "fr_guanido",
    "fr_halogen",
    "fr_hdrzine",
    "fr_hdrzone",
    "fr_imidazole",
    "fr_imide",
    "fr_isocyan",
    "fr_isothiocyan",
    "fr_ketone",
    "fr_ketone_Topliss",
    "fr_lactam",
    "fr_lactone",
    "fr_methoxy",
    "fr_morpholine",
    "fr_nitrile",
    "fr_nitro",
    "fr_nitro_arom",
    "fr_nitro_arom_nonortho",
    "fr_nitroso",
    "fr_oxazole",
    "fr_oxime",
    "fr_para_hydroxylation",
    "fr_phenol",
    "fr_phenol_noOrthoHbond",
    "fr_phos_acid",
    "fr_phos_ester",
    "fr_piperdine",
    "fr_piperzine",
    "fr_priamide",
    "fr_prisulfonamd",
    "fr_pyridine",
    "fr_quatN",
    "fr_sulfide",
    "fr_sulfonamd",
    "fr_sulfone",
    "fr_term_acetylene",
    "fr_tetrazole",
    "fr_thiazole",
    "fr_thiocyan",
    "fr_thiophene",
    "fr_unbrch_alkane",
    "fr_urea",
    "qed",
]


def _get_continuous_cdf_fn(dist_name, params, data_stats):
    dist_func = getattr(stats, dist_name)

    data_min = data_stats["data_min"]
    data_max = data_stats["data_max"]

    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    def cdf(
        v, dist=dist_func, arg=arg, loc=loc, scale=scale, minV=data_min, maxV=data_max
    ):
        v = dist.cdf(np.clip(v, minV, maxV), loc=loc, scale=scale, *arg)
        return np.clip(v, 0.0, 1.0)

    return cdf


def _get_discrete_cdf_fn(params, data_stats):
    arr = np.array(params)
    data_min = int(data_stats["data_min"])
    data_max = int(data_stats["data_max"])

    # arr contains the cdf values for each integer value in the range
    # [data_min, data_max]
    def cdf(v, arr=arr, minV=data_min, maxV=data_max):
        v = v.astype(np.int32)
        return arr[np.clip(v, minV, maxV) - minV]

    return cdf


def identity_fn(v: Any) -> Any:
    """Identity function that just returns the input value.

    Args:
        v: The input value of any type.

    Returns:
        The same value as the input, unchanged.
    """
    return v


def get_normalization_functions(
    desc_list: List[str], param_dict: Dict[str, Dict[str, Any]]
) -> Dict[str, Callable]:
    """Creates a dictionary of normalization functions for molecular descriptors.

    This function generates normalization functions for each descriptor in the provided
    list based on the normalization parameters specified in param_dict. It supports
    three types of normalization: no normalization (identity function), continuous
    normalization using statistical distributions, and discrete normalization using
    pre-computed CDF values.

    Args:
        desc_list: List of descriptor names to create normalization functions for.
        param_dict: Dictionary containing normalization parameters for each descriptor.
            Each descriptor should have a nested dictionary with the following keys:
            - "normalization_type": One of "no_norm", "continuous", or "discrete"
            - "normalization_name": (for continuous) Name of scipy.stats distribution
            - "params": Distribution parameters or pre-computed CDF values
            - "data_stats": Dictionary with "data_min" and "data_max" values

    Returns:
        Dictionary mapping descriptor names to their normalization functions.
        Each function takes a value and returns the normalized value.

    Raises:
        ValueError: If an unknown normalization_type is encountered.

    """

    norm_func_dict = {}
    for desc_name in desc_list:
        if desc_name not in param_dict:
            print(f"{desc_name} not in param_dict. Skipping")
            continue

        desc_param_dict = param_dict[desc_name]
        norm_type = desc_param_dict["normalization_type"]
        if norm_type == "no_norm":
            norm_func_dict[desc_name] = identity_fn
            continue
        if norm_type == "continuous":
            dist_name = desc_param_dict["normalization_name"]
            params = desc_param_dict["params"]
            data_stats = desc_param_dict["data_stats"]
            norm_func_dict[desc_name] = _get_continuous_cdf_fn(
                dist_name, params, data_stats
            )
        elif norm_type == "discrete":
            params = desc_param_dict["params"]
            data_stats = desc_param_dict["data_stats"]
            norm_func_dict[desc_name] = _get_discrete_cdf_fn(params, data_stats)
        else:
            raise ValueError("Unknown norm_type: {norm_type}")
    return norm_func_dict
