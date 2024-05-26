from typing import Literal
import pytest
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import Descriptors

from rfpred.functions import (
    canonicalise_smiles,
    get_solvents,
    get_maccs,
    get_solvent_features,
    get_rdkit_descriptors,
    process_input
)
 
@pytest.mark.parametrize("smiles, expected", [
    ("C1=CC=CC=C1", "c1ccccc1"),  # Benzene
    ("C(C(=O)O)N", "NCC(=O)O"),  # Alanine
    ("O=C(O)CC(=O)O", "O=C(O)CC(=O)O"),  # Succinic acid
    ("CC(C)CC1=CC=CC=C1", "CC(C)Cc1ccccc1"),  # Isopropylbenzene
    ("N[C@@H](C)C(=O)O", "C[C@H](N)C(=O)O"),  # L-alanine
    ("N[C@H](C)C(=O)O", "C[C@@H](N)C(=O)O")  # D-alanine
])
def test_canonical_smiles(smiles: Literal['C1=CC=CC=C1'] | Literal['C(C(=O)O)N'] | Literal['O=C(O)CC(=O)O'] | Literal['CC(C)CC1=CC=CC=C1'] | Literal['N[C@@H](C)C(=O)O'] | Literal['N[C@H](C)C(=O)O'], expected: Literal['c1ccccc1'] | Literal['NCC(=O)O'] | Literal['O=C(O)CC(=O)O'] | Literal['CC(C)Cc1ccccc1'] | Literal['C[C@H](N)C(=O)O'] | Literal['C[C@@H](N)C(=O)O']):
    assert canonicalise_smiles(smiles) == expected


def test_empty_smiles():

    empty_smiles = ""
    assert canonicalise_smiles(empty_smiles) == ""


def test_invalide_smiles():

    invalid_smiles = "invalid_smiles"
    
    with pytest.raises(ValueError) as e:
        canonicalise_smiles(invalid_smiles)
    assert str(e.value).startswith(f"Invalid SMILES string: {invalid_smiles}")


def test_get_solvents():

    test = {
            'productSmiles': ["['CCO']", "['CC(=O)O']", "['CCN(CC)CC']", "['CCCCCCCC']", "['CCCCCCC']", "['CCCCCC']", "['CCCCC']", "['CCCC']", "['CCC']", "['CC']", "['C']", "['C1CCCCC1']", "['CC(C)C(=O)O']"],
            'Rf': ['0.5', '0.77', '0.1', '1.3', '0.6', '0.8', '0.9', '0.4', '0.7', '0.3', '0.2', None, '0.33'],
            'Solvent_A': ['ethyl acetate', None, 'hexane', 'hexane', 'Methanol', 'Hexane', 'Ethyl acetate', 'Ethyl acetate', 'hexane', 'Ethyl acetate', 'hexane', 'DCM', None],
            'Solvent_B': ['hexane', 'hexane', 'ethyl acetate', None, 'Ethanol', 'Ethyl acetate', 'Tetrahydrofuran', 'hexane', 'DCM', 'hexane', 'DCM', 'MeOH', None],
            'Percent_A': ['50', None, '20', '20', '70', '10', '100', '40', '66.666666', '60', '50', '80', '20'],
            'Percent_B': ['50', '30', None, '80', '30', '25', None, '80', '33.333333', '40', '45', '20', '80'],
            'Additive_C': [None, None, None, None, 'TEA', None, None, None, None, 'TEA', None, None, None],
            'Percent_C': [None, None, None, None, '5', None, None, None, None, None, '5', None, None]
        }
    df_test = pd.DataFrame(test)

    expected = ['DCM', 'Ethanol', 'Ethyl acetate', 'ethyl acetate', 'Hexane', 'hexane', 'MeOH', 'Methanol', 'Tetrahydrofuran']

    result = get_solvents(df_test)

    assert sorted(result) == sorted(expected)


def test_get_maccs():
    # Test with a known molecule
    smiles = 'CCO'  # Ethanol
    expected_maccs = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smiles))
    expected_array = np.array([int(x) for x in list(expected_maccs.ToBitString())])
    result = get_maccs(smiles)
    np.testing.assert_array_equal(result, expected_array)

    # Test with another known molecule
    smiles = 'CCCC'  # Butane
    expected_maccs = MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smiles))
    expected_array = np.array([int(x) for x in list(expected_maccs.ToBitString())])
    result = get_maccs(smiles)
    np.testing.assert_array_equal(result, expected_array)


def test_get_solvent_features():
    solvents = ['water', 'ethanol', 'methanol', 'acetone']

    # Test with known solvents and percentages
    solvent_A = 'water'
    solvent_B = 'ethanol'
    percent_A = 60.0
    expected_feature = np.array([60.0, 40.0, 0.0, 0.0])
    result = get_solvent_features(solvent_A, solvent_B, percent_A, solvents)
    np.testing.assert_array_equal(result, expected_feature)

    # Test with a different combination
    solvent_A = 'methanol'
    solvent_B = 'acetone'
    percent_A = 30.0
    expected_feature = np.array([0.0, 0.0, 30.0, 70.0])
    result = get_solvent_features(solvent_A, solvent_B, percent_A, solvents)
    np.testing.assert_array_equal(result, expected_feature)

    # Test with solvent_A not in the list
    solvent_A = 'hexane'
    solvent_B = 'ethanol'
    percent_A = 50.0
    expected_feature = np.array([0.0, 50.0, 0.0, 0.0])
    result = get_solvent_features(solvent_A, solvent_B, percent_A, solvents)
    np.testing.assert_array_equal(result, expected_feature)

    # Test with solvent_B not in the list
    solvent_A = 'water'
    solvent_B = 'hexane'
    percent_A = 80.0
    expected_feature = np.array([80.0, 0.0, 0.0, 0.0])
    result = get_solvent_features(solvent_A, solvent_B, percent_A, solvents)
    np.testing.assert_array_equal(result, expected_feature)

    # Test with both solvents not in the list
    solvent_A = 'hexane'
    solvent_B = 'benzene'
    percent_A = 20.0
    expected_feature = np.array([0.0, 0.0, 0.0, 0.0])
    result = get_solvent_features(solvent_A, solvent_B, percent_A, solvents)
    np.testing.assert_array_equal(result, expected_feature)

    # Test with percent_A as 0
    solvent_A = 'ethanol'
    solvent_B = 'acetone'
    percent_A = 0.0
    expected_feature = np.array([0.0, 0.0, 0.0, 100.0])
    result = get_solvent_features(solvent_A, solvent_B, percent_A, solvents)
    np.testing.assert_array_equal(result, expected_feature)

    # Test with percent_A as 100
    solvent_A = 'water'
    solvent_B = 'methanol'
    percent_A = 100.0
    expected_feature = np.array([100.0, 0.0, 0.0, 0.0])
    result = get_solvent_features(solvent_A, solvent_B, percent_A, solvents)
    np.testing.assert_array_equal(result, expected_feature)

def test_get_rdkit_descriptors():
    # Testfall 1: Wasser (H2O)
    smiles = 'O'
    expected = np.array([
        Descriptors.MolWt(Chem.MolFromSmiles(smiles)),
        Descriptors.MolLogP(Chem.MolFromSmiles(smiles)),
        Descriptors.NumHDonors(Chem.MolFromSmiles(smiles)),
        Descriptors.NumHAcceptors(Chem.MolFromSmiles(smiles))
    ])
    result = get_rdkit_descriptors(smiles)
    np.testing.assert_array_almost_equal(result, expected, decimal=6)

    # Testfall 2: Benzol (C6H6)
    smiles = 'c1ccccc1'
    expected = np.array([
        Descriptors.MolWt(Chem.MolFromSmiles(smiles)),
        Descriptors.MolLogP(Chem.MolFromSmiles(smiles)),
        Descriptors.NumHDonors(Chem.MolFromSmiles(smiles)),
        Descriptors.NumHAcceptors(Chem.MolFromSmiles(smiles))
    ])
    result = get_rdkit_descriptors(smiles)
    np.testing.assert_array_almost_equal(result, expected, decimal=6)

    # Testfall 3: Ethanol (C2H6O)
    smiles = 'CCO'
    expected = np.array([
        Descriptors.MolWt(Chem.MolFromSmiles(smiles)),
        Descriptors.MolLogP(Chem.MolFromSmiles(smiles)),
        Descriptors.NumHDonors(Chem.MolFromSmiles(smiles)),
        Descriptors.NumHAcceptors(Chem.MolFromSmiles(smiles))
    ])
    result = get_rdkit_descriptors(smiles)
    np.testing.assert_array_almost_equal(result, expected, decimal=6)

def test_process_input():
# Beispiel-Eingaben
    smiles = 'CCO'  # Ethanol
    solvent_A = 'DCM'
    solvent_B = 'MeOH'
    percent_A = 60.0

    # Erwartete Ausgabe
    solvents = ['DCM', 'MeOH', 'MeCN', 'Toluene', 'Hexane', 'Chloroform', 'Acetone', 'EtOH', 'diethyl ether', 'heptane', 'petroleum ether (2-methylpentane)', 'triethylamine', 'EtOAc', 'THF']
    maccs = get_maccs(smiles)
    solvent_features = get_solvent_features(solvent_A, solvent_B, percent_A, solvents)
    rdkit_descriptors = get_rdkit_descriptors(smiles)
    expected = np.concatenate([maccs, rdkit_descriptors, solvent_features])

    # Tatsächliche Ausgabe der Funktion
    result = process_input(smiles, solvent_A, solvent_B, percent_A)

    # Testen, ob das Ergebnis mit der erwarteten Ausgabe übereinstimmt
    np.testing.assert_array_almost_equal(result, expected, decimal=6)

