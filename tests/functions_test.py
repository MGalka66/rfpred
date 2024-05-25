import pytest
import pandas as pd

from rfpred.functions import (
    extract_rows_with_rf,
    canonicalise_smiles,
    get_solvents,
    convert_solvents,
    clean_smiles,
    are_enantiomers
)

def test_extract_rows_with_rf():

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

    result = extract_rows_with_rf(df_test)

    #comparing the result of the function with the expected output
    pd.testing.assert_frame_equal(df_expected, result)

 

@pytest.mark.parametrize("smiles, expected", [
    ("C1=CC=CC=C1", "c1ccccc1"),  # Benzene
    ("C(C(=O)O)N", "NCC(=O)O"),  # Alanine
    ("O=C(O)CC(=O)O", "O=C(O)CC(=O)O"),  # Succinic acid
    ("CC(C)CC1=CC=CC=C1", "CC(C)Cc1ccccc1"),  # Isopropylbenzene
    ("N[C@@H](C)C(=O)O", "C[C@H](N)C(=O)O"),  # L-alanine
    ("N[C@H](C)C(=O)O", "C[C@@H](N)C(=O)O")  # D-alanine
])
def test_canonical_smiles(smiles, expected):
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

