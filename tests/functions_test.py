from typing import Literal
import pytest
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import Descriptors

from rfpred.functions import (
    extract_rows_with_rf,
    canonicalise_smiles,
    get_solvents,
    convert_solvents,
    clean_smiles,
    are_enantiomers,
    get_maccs,
    get_solvent_features,
    get_rdkit_descriptors,
    process_input
)


def test_extract_rows_with_rf():
    """
    Test the extract_rows_with_rf function to ensure it correctly extracts rows
    containing 'Rf' values from the specified column in a DataFrame.

    The function should match 'Rf' in various formats and contexts, including
    different cases and notations. It should return only the rows that contain
    'Rf' values.
    """

    # Test texts with Rf and without Rf values
    test = {'paragraphText': [
        'The value of the compound is', 
        'The compound has an RF value of 0.5', 
        'The measured Rf:0.5 of the product...', 
        'Dont show this f:0.5 value',
        'An RF(0.5) was measured',
        'Dont show this R: 0.5 value',
        'The Rf value of the compound is Rf=0.4',
        'The valueRf of the cpmound is 0.5',
        'The measured Rf : 0.5 of the product...',
        'Sometimes theRfvalue is hidden in the text',
        'And somtimes the Rf~0.6 is noted with a ~.'
    ]}
    
    df_test = pd.DataFrame(test)

    # Expected output of the function
    expected = {'paragraphText': [ 
        'The compound has an RF value of 0.5', 
        'The measured Rf:0.5 of the product...', 
        'An RF(0.5) was measured',
        'The Rf value of the compound is Rf=0.4',
        'The valueRf of the cpmound is 0.5',
        'The measured Rf : 0.5 of the product...',
        'Sometimes theRfvalue is hidden in the text',
        'And somtimes the Rf~0.6 is noted with a ~.'

    ]}

    df_expected = pd.DataFrame(expected)

    #running the function
    result = extract_rows_with_rf(df_test, 'paragraphText')

    #comparing the result of the function with the expected output
    pd.testing.assert_frame_equal(result, df_expected)

 
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


def test_convert_solvents():

    test = {
        'Solvent_A': ['ethyl acetate', 'Hexane', 'methylene chloride', 'mecn', 'n-hex', 'acetone', 'ethanol', 'n-heptane', 'et3n', 'none', '20%Etahnol', 'none', 'EA/HE'],
        'Solvent_B': ['hexane', 'EA', 'acetonitril', 'ch3oh', 'toluene', 'trichloromethane', 'et2o', 'pe', 'tetrahydrofuran', 'EA', 'et2o', '(TEA)', 'THF']
    }

    df_test = pd.DataFrame(test)

    expected = {
        'Solvent_A': ['EtOAc', 'Hexane', 'DCM', 'MeCN', 'Hexane', 'Acetone', 'EtOH', 'heptane', 'triethylamine', 'none'],
        'Solvent_B': ['Hexane', 'EtOAc', 'MeCN', 'MeOH', 'Toluene', 'Chloroform', 'diethyl ether', 'petroleum ether (2-methylpentane)', 'THF', 'EtOAc' ],
        'Solvent_A_Smiles': ['O=C(OCC)C', 'CCCCCC', 'ClCCl', 'CC#N', 'CCCCCC', 'CC(=O)C', 'CCO', 'CCCCCCC', 'CCN(CC)CC', None],
        'Solvent_B_Smiles': ['CCCCCC', 'O=C(OCC)C', 'CC#N', 'CO', 'Cc1ccccc1', 'ClC(Cl)Cl', 'CCOCC', 'CCCC(C)C', 'C1CCOC1', 'O=C(OCC)C']
    }

    df_expected = pd.DataFrame(expected)

    sorted_out = {
        'Solvent_A': ['20%Etahnol', 'none', 'EA/HE'],
        'Solvent_B': ['et2o', '(TEA)', 'THF']
    }   

    df_sorted_out_expected = pd.DataFrame(sorted_out)

    result, df_sorted_out = convert_solvents(df_test, 'Solvent_A', 'Solvent_B')

    pd.testing.assert_frame_equal(df_expected, result)

    pd.testing.assert_frame_equal(df_sorted_out_expected, df_sorted_out)


def clean_smiles():
    
    test = {
        'Smiles': ['  COO', 'CCC  ', '"COC"', 'CC', ' COO"', '" CCCC', '   C   ']
    }

    df_test = pd.DataFrame(test)

    expected = {
        'Smiles': ['COO', 'CCC', 'COC', 'CC', 'COO', 'CCCC', 'C']
    }

    df_expected = pd.DataFrame(expected)

    result = clean_smiles(df_test, 'Smiles')

    pd.testing.assert_frame_equal(df_expected, result)

def test_are_enantiomers():

    # Test with known enantiomers
    smiles_A = 'N[C@@H](C)C(=O)O'  # L-alanine
    smiles_B = 'N[C@H](C)C(=O)O'  # D-alanine
    assert are_enantiomers(smiles_A, smiles_B) == (True, 1)

    # Test with known non-enantiomers
    smiles_A = 'CCO'  # Ethanol
    smiles_B = 'CCCC'  # Butane
    assert are_enantiomers(smiles_A, smiles_B) == (False, None)

    # Test with the same molecule
    smiles_A = 'CCO'  # Ethanol
    smiles_B = 'CCO'  # Ethanol
    assert are_enantiomers(smiles_A, smiles_B) == (False, None)

    # Test with invalid SMILES
    smiles_A = 'invalid_smiles'
    smiles_B = 'CCO'  # Ethanol
    assert are_enantiomers(smiles_A, smiles_B) == (False, None)

    smiles_A = 'CCO'  # Ethanol
    smiles_B = 'invalid_smiles'
    assert are_enantiomers(smiles_A, smiles_B) == (False, None)

    smiles_A = 'invalid_smiles'
    smiles_B = 'invalid_smiles'
    assert are_enantiomers(smiles_A, smiles_B) == (False, None)

    smiles_A = 'CC(O)[C@@H](N)C(=O)O'
    smiles_B = 'CC(O)[C@H](N)C(=O)O'
    assert are_enantiomers(smiles_A, smiles_B) == (True, 3)


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
    # Testcase 1: Wasser (H2O)
    smiles = 'O'
    expected = np.array([
        Descriptors.MolWt(Chem.MolFromSmiles(smiles)),
        Descriptors.MolLogP(Chem.MolFromSmiles(smiles)),
        Descriptors.NumHDonors(Chem.MolFromSmiles(smiles)),
        Descriptors.NumHAcceptors(Chem.MolFromSmiles(smiles))
    ])
    result = get_rdkit_descriptors(smiles)
    np.testing.assert_array_almost_equal(result, expected, decimal=6)

    # Testcase 2: Benzene (C6H6)
    smiles = 'c1ccccc1'
    expected = np.array([
        Descriptors.MolWt(Chem.MolFromSmiles(smiles)),
        Descriptors.MolLogP(Chem.MolFromSmiles(smiles)),
        Descriptors.NumHDonors(Chem.MolFromSmiles(smiles)),
        Descriptors.NumHAcceptors(Chem.MolFromSmiles(smiles))
    ])
    result = get_rdkit_descriptors(smiles)
    np.testing.assert_array_almost_equal(result, expected, decimal=6)

    # Testcase 3: Ethanol (C2H6O)
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
# Example input
    smiles = 'CCO'  # Ethanol
    solvent_A = 'DCM'
    solvent_B = 'MeOH'
    percent_A = 60.0

    # Expected output
    solvents = ['DCM', 'MeOH', 'MeCN', 'Toluene', 'Hexane', 'Chloroform', 'Acetone', 'EtOH', 'diethyl ether', 'heptane', 'petroleum ether (2-methylpentane)', 'triethylamine', 'EtOAc', 'THF']
    maccs = get_maccs(smiles)
    solvent_features = get_solvent_features(solvent_A, solvent_B, percent_A, solvents)
    rdkit_descriptors = get_rdkit_descriptors(smiles)
    expected = np.concatenate([maccs, rdkit_descriptors, solvent_features])

    # output of the function
    result = process_input(smiles, solvent_A, solvent_B, percent_A)

    # Test if the output is as expected
    np.testing.assert_array_almost_equal(result, expected, decimal=6)

