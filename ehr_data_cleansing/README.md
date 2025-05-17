# EHR Data Cleansing with NLP and Machine Learning

This project demonstrates automated data cleansing for Electronic Health Records (EHRs) using Natural Language Processing (NLP) and machine learning techniques.

## Features

- Standardizing medical terminology (ICD-10 codes)
- Correcting typographical errors in diagnoses
- Standardizing medication names, dosages, and frequencies
- Date format standardization
- Entity extraction from medical text
- Clustering similar diagnoses using machine learning

## Setup Instructions

1. **Install Dependencies**:

   ```
   pip install -r requirements.txt
   ```

2. **Install SpaCy Language Model**:

   ```
   python -m spacy download en_core_web_md
   ```

3. **Jupyter Notebook**:
   Run the `ehr_data_cleansing.ipynb` notebook:
   ```
   jupyter notebook ehr_data_cleansing.ipynb
   ```

## Usage

The notebook processes EHR data in CSV format with the following fields:

- patient_id
- diagnosis
- diagnosis_code (ICD-10)
- medication
- date_of_visit

The cleansing pipeline:

1. Loads sample or real EHR data
2. Applies standardization processes
3. Extracts medical entities
4. Clusters similar diagnoses
5. Outputs a cleansed dataset

## Customization

To use with your own data:

1. Replace the sample data creation cell with code to load your own data
2. Update the standard terminology dictionaries with domain-specific terms
3. Adjust matching thresholds as needed for your data quality

## Requirements

See `requirements.txt` for a complete list of dependencies.
