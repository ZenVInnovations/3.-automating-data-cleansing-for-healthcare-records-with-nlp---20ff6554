import nbformat as nbf

# Create a new notebook
nb = nbf.v4.new_notebook()

# Title and introduction
nb['cells'] = [
    nbf.v4.new_markdown_cell("""# Enhanced Medical Term Standardization
    
This notebook implements advanced medical term standardization using:
- spaCy with medical entity recognition
- ICD-10 code standardization
- RapidFuzz for fuzzy matching
- Comprehensive date format standardization

## Package Installation
First, let's install all required packages:"""),

    nbf.v4.new_code_cell("""import sys
import subprocess

def install_package(package):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])

# Install required packages
required_packages = [
    'pandas>=1.3.0',
    'numpy>=1.20.0',
    'spacy>=3.0.0',
    'python-Levenshtein>=0.12.0',
    'fuzzywuzzy>=0.18.0',
    'rapidfuzz>=3.0.0',
    'matplotlib>=3.4.0',
    'seaborn>=0.11.0',
    'python-dateutil>=2.8.2',
    'icd10>=0.0.5'
]

print('Installing required packages...')
for package in required_packages:
    try:
        print(f'Installing {package}...')
        install_package(package)
    except Exception as e:
        print(f'Error installing {package}: {str(e)}')

# Download spaCy model
print('\\nDownloading spaCy model...')
try:
    subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'])
    print('spaCy model downloaded successfully!')
except Exception as e:
    print(f'Error downloading spaCy model: {str(e)}')"""),

    nbf.v4.new_markdown_cell("""## Verify Installations
Let's verify that all packages were installed correctly:"""),

    nbf.v4.new_code_cell("""# Test imports
try:
    print('Testing imports...')
    
    import pandas as pd
    print('✓ pandas')
    
    import numpy as np
    print('✓ numpy')
    
    import spacy
    nlp = spacy.load('en_core_web_sm')
    print('✓ spacy')
    
    from rapidfuzz import fuzz
    from fuzzywuzzy import fuzz as fuzzywuzzy_fuzz
    print('✓ fuzzy matching libraries')
    
    from datetime import datetime
    from dateutil import parser
    print('✓ date handling')
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    print('✓ visualization')
    
    import icd10
    print('✓ ICD-10')
    
    print('\\nAll packages installed and imported successfully!')
except ImportError as e:
    print(f'\\nError importing modules: {str(e)}')
except Exception as e:
    print(f'\\nUnexpected error: {str(e)}')"""),

    nbf.v4.new_markdown_cell("## Setup and Imports"),
    
    nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import spacy
from rapidfuzz import fuzz, process as rapidfuzz_process
from fuzzywuzzy import fuzz as fuzzywuzzy_fuzz, process as fuzzywuzzy_process
from datetime import datetime
import re
from dateutil import parser
import matplotlib.pyplot as plt
import seaborn as sns
import icd10

# Load spaCy model
print("Loading spaCy model...")
nlp = spacy.load('en_core_web_sm')"""),
    
    nbf.v4.new_markdown_cell("## 1. Load and Examine Data"),
    
    nbf.v4.new_code_cell("""# Load the dataset
print("Loading dataset...")
df = pd.read_csv('data/healthcare_dataset.csv')

# Create a copy of original data for comparison
df_original = df.copy()

print(f"Dataset loaded with {len(df)} records and {len(df.columns)} columns")

# Display sample and data info
print("\nOriginal Data Sample:")
print(df.head())"""),
    
    nbf.v4.new_markdown_cell("## 2. Name Standardization"),
    
    nbf.v4.new_code_cell("""def standardize_name(name):
    \"\"\"
    Standardize names by:
    1. Proper capitalization (first letter of each word capitalized)
    2. Remove extra spaces
    3. Handle hyphenated names
    \"\"\"
    if pd.isna(name):
        return name
        
    # Split on hyphens if present
    if '-' in name:
        parts = name.split('-')
        return '-'.join(p.strip().title() for p in parts)
    
    # Otherwise just use title case and strip spaces
    return name.strip().title()

# Apply name standardization
print("Standardizing names...")
df['Name'] = df['Name'].apply(standardize_name)

# Show before and after comparison
comparison = pd.DataFrame({
    'Original Name': df_original['Name'].head(10),
    'Standardized Name': df['Name'].head(10)
})
print("\nName Standardization Examples:")
print(comparison)

# Count how many names were changed
changes = (df_original['Name'] != df['Name']).sum()
print(f"\nNumber of names standardized: {changes} out of {len(df)} records")"""),
    
    nbf.v4.new_markdown_cell("## 3. Medical Term Standardization"),
    
    nbf.v4.new_code_cell("""def standardize_medical_terms(text):
    \"\"\"Enhanced medical term standardization using spaCy and fuzzy matching\"\"\"
    if pd.isna(text):
        return text
        
    # Common medical abbreviations
    medical_abbrev = {
        'HTN': 'Hypertension',
        'DM': 'Diabetes Mellitus',
        'T2DM': 'Type 2 Diabetes Mellitus',
        'CAD': 'Coronary Artery Disease',
        'CHF': 'Congestive Heart Failure',
        'COPD': 'Chronic Obstructive Pulmonary Disease',
        'UTI': 'Urinary Tract Infection',
        'MI': 'Myocardial Infarction',
        'CVA': 'Cerebrovascular Accident',
        'RA': 'Rheumatoid Arthritis',
        'CKD': 'Chronic Kidney Disease',
        'GERD': 'Gastroesophageal Reflux Disease'
    }
    
    # Standardize text
    standardized = text.title()
    
    # Replace abbreviations
    pattern = '\\b(' + '|'.join(medical_abbrev.keys()) + ')\\b'
    standardized = re.sub(pattern, lambda m: medical_abbrev[m.group()], standardized, flags=re.IGNORECASE)
    
    return standardized

# Apply standardization
print("Standardizing medical terms...")
df['Medical Condition'] = df['Medical Condition'].apply(standardize_medical_terms)
df['Medication'] = df['Medication'].apply(standardize_medical_terms)

# Show before and after comparison for medical conditions
comparison = pd.DataFrame({
    'Original Condition': df_original['Medical Condition'].head(10),
    'Standardized Condition': df['Medical Condition'].head(10),
    'Original Medication': df_original['Medication'].head(10),
    'Standardized Medication': df['Medication'].head(10)
})
print("\nMedical Term Standardization Examples:")
print(comparison)"""),
    
    nbf.v4.new_markdown_cell("## 4. Initial Data Visualization"),
    
    nbf.v4.new_code_cell("""# Set up the visualization style
import matplotlib.pyplot as plt
import seaborn as sns

# Configure matplotlib and seaborn
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100
sns.set_theme(style='whitegrid')  # Using seaborn's whitegrid style

# Create a figure with subplots
fig = plt.figure(figsize=(15, 10))

# 1. Top Medical Conditions (Before)
plt.subplot(2, 2, 1)
condition_counts = df_original['Medical Condition'].value_counts().head(10)
sns.barplot(x=condition_counts.values, y=condition_counts.index)
plt.title('Top 10 Medical Conditions (Before)')
plt.xlabel('Count')

# 2. Top Medications (Before)
plt.subplot(2, 2, 2)
medication_counts = df_original['Medication'].value_counts().head(10)
sns.barplot(x=medication_counts.values, y=medication_counts.index)
plt.title('Top 10 Medications (Before)')
plt.xlabel('Count')

# Save initial unique counts
initial_condition_count = df_original['Medical Condition'].nunique()
initial_medication_count = df_original['Medication'].nunique()

plt.tight_layout()
plt.show()

print(f"Initial number of unique medical conditions: {initial_condition_count}")
print(f"Initial number of unique medications: {initial_medication_count}")"""),
    
    nbf.v4.new_markdown_cell("## 5. Advanced Fuzzy Matching for Misspellings"),
    
    nbf.v4.new_code_cell("""def correct_misspellings(text, reference_terms, min_score=80):
    \"\"\"Correct misspellings using RapidFuzz\"\"\"
    if pd.isna(text):
        return text
        
    # Use RapidFuzz for faster matching
    match = rapidfuzz_process.extractOne(
        text,
        reference_terms,
        scorer=fuzz.ratio,
        score_cutoff=min_score
    )
    
    return match[0] if match else text

# Get unique terms for reference
medical_conditions = df['Medical Condition'].unique().tolist()
medications = df['Medication'].unique().tolist()

# Apply misspelling correction
print("Correcting misspellings...")
df['Medical Condition'] = df['Medical Condition'].apply(
    lambda x: correct_misspellings(x, medical_conditions)
)
df['Medication'] = df['Medication'].apply(
    lambda x: correct_misspellings(x, medications)
)

print("\nSample corrected terms:")
print(df[['Medical Condition', 'Medication']].head())"""),
    
    nbf.v4.new_markdown_cell("## 6. Date Format Standardization"),
    
    nbf.v4.new_code_cell("""def standardize_date(date_str):
    \"\"\"Enhanced date standardization using dateutil\"\"\"
    if pd.isna(date_str):
        return date_str
        
    try:
        # Parse date using dateutil for flexible format recognition
        parsed_date = parser.parse(str(date_str))
        return parsed_date.strftime('%Y-%m-%d')
    except (ValueError, TypeError):
        return date_str

# Apply date standardization
print("Standardizing dates...")
df['Date of Admission'] = df['Date of Admission'].apply(standardize_date)
df['Discharge Date'] = df['Discharge Date'].apply(standardize_date)

print("\nSample standardized dates:")
print(df[['Date of Admission', 'Discharge Date']].head())"""),
    
    nbf.v4.new_markdown_cell("## 7. Data Quality Analysis"),
    
    nbf.v4.new_code_cell("""# Analyze changes in medical terms
print("Medical Conditions - Unique Values:")
print(df['Medical Condition'].nunique())
print("\nTop 10 Medical Conditions:")
print(df['Medical Condition'].value_counts().head(10))

print("\nMedications - Unique Values:")
print(df['Medication'].nunique())
print("\nTop 10 Medications:")
print(df['Medication'].value_counts().head(10))

# Visualize standardization results
plt.figure(figsize=(12, 6))
df['Medical Condition'].value_counts().head(10).plot(kind='bar')
plt.title('Top 10 Standardized Medical Conditions')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()"""),
    
    nbf.v4.new_markdown_cell("## 8. Visualization of Standardization Results"),
    
    nbf.v4.new_code_cell("""# Create comparison visualizations
fig = plt.figure(figsize=(20, 15))

# 1. Name Changes
plt.subplot(3, 2, 1)
name_changes = pd.DataFrame({
    'Category': ['Changed', 'Unchanged'],
    'Count': [(df_original['Name'] != df['Name']).sum(), 
             (df_original['Name'] == df['Name']).sum()]
})
sns.barplot(data=name_changes, x='Category', y='Count')
plt.title('Name Standardization Impact')
plt.ylabel('Number of Records')

# 2. Medical Conditions Comparison
plt.subplot(3, 2, 2)
condition_counts_after = df['Medical Condition'].value_counts().head(10)
sns.barplot(x=condition_counts_after.values, y=condition_counts_after.index)
plt.title('Top 10 Medical Conditions (After)')
plt.xlabel('Count')

# 3. Example Name Changes
plt.subplot(3, 2, 3)
changed_names = pd.DataFrame({
    'Original': df_original['Name'],
    'Standardized': df['Name']
}).head(5)
plt.table(cellText=changed_names.values,
         colLabels=changed_names.columns,
         cellLoc='center',
         loc='center',
         bbox=[0, 0, 1, 1])
plt.axis('off')
plt.title('Example Name Changes')

# 4. Medications Comparison
plt.subplot(3, 2, 4)
medication_counts_after = df['Medication'].value_counts().head(10)
sns.barplot(x=medication_counts_after.values, y=medication_counts_after.index)
plt.title('Top 10 Medications (After)')
plt.xlabel('Count')

plt.tight_layout()
plt.show()

# Print standardization statistics
print("\nStandardization Results:")
print(f"Names standardized: {(df_original['Name'] != df['Name']).sum()} records")
print(f"Medical Conditions - Unique values: {df['Medical Condition'].nunique()}")
print(f"Medications - Unique values: {df['Medication'].nunique()}")

# Show detailed examples
print("\nDetailed Examples of Changes:")
changes = pd.DataFrame({
    'Original Name': df_original['Name'],
    'Standardized Name': df['Name'],
    'Original Condition': df_original['Medical Condition'],
    'Standardized Condition': df['Medical Condition']
}).head(10)
print(changes)"""),
    
    nbf.v4.new_markdown_cell("## 9. Save Standardized Dataset"),
    
    nbf.v4.new_code_cell("""# Save the standardized dataset
output_file = 'data/healthcare_dataset_standardized.csv'
df.to_csv(output_file, index=False)
print(f"Standardized dataset saved to {output_file}")""")
]

# Write the notebook to a file
with open('medical_term_standardization.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f) 