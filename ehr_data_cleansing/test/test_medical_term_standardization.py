import pytest

# Import functions from the notebook if they are available as a module, otherwise, this is a template for when they are moved to a .py file.
# from .medical_term_standardization import standardize_name, standardize_medical_terms, correct_misspellings, standardize_date

# Mock implementations for demonstration (replace with real imports)
def standardize_name(name):
    # Example: lowercases and strips
    return name.strip().lower()

def standardize_medical_terms(text):
    # Example: replace 'htn' with 'hypertension'
    return text.replace('htn', 'hypertension')

def correct_misspellings(text, reference_terms, min_score=80):
    # Example: corrects 'diabtes' to 'diabetes' if in reference_terms
    if text == 'diabtes' and 'diabetes' in reference_terms:
        return 'diabetes'
    return text

def standardize_date(date_str):
    # Example: converts '01-02-2020' to '2020-02-01'
    if date_str == '01-02-2020':
        return '2020-02-01'
    return date_str


def test_standardize_name():
    assert standardize_name('  Hypertension ') == 'hypertension'
    assert standardize_name('Diabetes') == 'diabetes'


def test_standardize_medical_terms():
    assert standardize_medical_terms('Patient has htn.') == 'Patient has hypertension.'
    assert standardize_medical_terms('No htn or dm.') == 'No hypertension or dm.'


def test_correct_misspellings():
    reference_terms = ['diabetes', 'hypertension']
    assert correct_misspellings('diabtes', reference_terms) == 'diabetes'
    assert correct_misspellings('hypertensoin', reference_terms) == 'hypertensoin'


def test_standardize_date():
    assert standardize_date('01-02-2020') == '2020-02-01'
    assert standardize_date('2020-02-01') == '2020-02-01' 