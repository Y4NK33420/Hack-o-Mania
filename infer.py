import pefile
import math

def calculate_entropy(data):
    if not data:
        return 0.0
    entropy = 0
    for x in range(256):
        p_x = float(data.count(x))/len(data)
        if p_x > 0:
            entropy += - p_x * math.log(p_x, 2)
    return entropy

def get_resource_data(pe, resource_entry):
    data = b''
    for entry in resource_entry.entries:  # Changed this line
        if hasattr(entry, 'directory'):
            data += get_resource_data(pe, entry.directory)
        else:
            data += pe.get_data(entry.data.struct.OffsetToData, entry.data.struct.Size)
    return data

def extract_pe_features(file_path):
    pe_features = {}



    pe = pefile.PE(file_path)
    # Basic PE header information
    pe_features['Machine'] = pe.FILE_HEADER.Machine
    pe_features['SizeOfOptionalHeader'] = pe.FILE_HEADER.SizeOfOptionalHeader
    pe_features['Characteristics'] = pe.FILE_HEADER.Characteristics
    # Optional header information
    pe_features['AddressOfEntryPoint'] = pe.OPTIONAL_HEADER.AddressOfEntryPoint
    pe_features['ImageBase'] = pe.OPTIONAL_HEADER.ImageBase
    pe_features['Subsystem'] = pe.OPTIONAL_HEADER.Subsystem
    # Section information
    pe_features['SectionsNb'] = len(pe.sections)
    entropy_list = [section.get_entropy() for section in pe.sections]
    pe_features['SectionsMeanEntropy'] = sum(entropy_list) / len(entropy_list)
    # Imports information
    pe_features['ImportsNbDLL'] = len(pe.DIRECTORY_ENTRY_IMPORT)
    imports_count = sum(len(entry.imports) for entry in pe.DIRECTORY_ENTRY_IMPORT)
    pe_features['ImportsNb'] = imports_count
    pe_features['ImportsNbOrdinal'] = len(pe.DIRECTORY_ENTRY_IMPORT)
    # Exports information
    if hasattr(pe, 'DIRECTORY_ENTRY_EXPORT'):
        pe_features['ExportNb'] = len(pe.DIRECTORY_ENTRY_EXPORT.symbols)
    else:
        pe_features['ExportNb'] = 0
# Resources information
    pe_features['ResourcesNb'] = len(pe.DIRECTORY_ENTRY_RESOURCE.entries)

    pe_features['LoadConfigurationSize'] = pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_LOAD_CONFIG']].Size
    pe_features['VersionInformationSize'] = pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_RESOURCE']].Size
    return pe_features.values()

# Example usage:
# file_path = r"D:\Downloads\SpotifySetup.exe"
# pe_features = extract_pe_features(file_path)
# print('length is ',len(pe_features))
# print(pe_features)

import joblib
import pandas as pd

# Load the trained model
model = joblib.load(r'/Users/harshmodi/Desktop/HackoMania/LandingPage/malware.joblib')

# Function to preprocess data for inference
def preprocess_data(pe_features):
    # Convert dictionary to DataFrame with a single row
    data = pd.DataFrame([pe_features])
    # Handle any missing values
    data.fillna(0, inplace=True)  # Assuming missing values are filled with 0
    return data


def main(file_path):
    features = extract_pe_features(file_path)
    data = preprocess_data(features)
    prediction = model.predict_proba(data)[0]
    return prediction[1]*100
# Extract features from PE file
# pe_features = extract_pe_features(file_path)

# # Preprocess the extracted features
# data = preprocess_data(pe_features)

# # Perform inference using the loaded model
# prediction = model.predict_proba(data)[0]

# # The prediction will be a list of two values, where the first value is the probability of the file being benign and the second value is the probability of the file being malware
# print('The predicted probability of the file being malware is:', prediction[1]*100, '%')
print(main(r'/Users/harshmodi/myenv/lib/python3.11/site-packages/setuptools/cli-arm64.exe'))