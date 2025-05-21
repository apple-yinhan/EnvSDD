import sys
import config
import pandas as pd
from tqdm import tqdm
import json
import os

metadata = pd.read_csv(config.envsdd_test_meta_path)
os.makedirs(f"{config.metadata_json_file}/tta/test", exist_ok=True)
os.makedirs(f"{config.metadata_json_file}/ata/test", exist_ok=True)

# # Output metadata jsons
## tta deepfake
tta_test01_path = f"{config.metadata_json_file}/tta/test/test01.json"
tta_test01_mono_path = f"{config.metadata_json_file}/tta/test/test01_mono.json"
tta_test01_poly_path = f"{config.metadata_json_file}/tta/test/test01_poly.json"

tta_test02_path = f"{config.metadata_json_file}/tta/test/test02.json"
tta_test02_mono_path = f"{config.metadata_json_file}/tta/test/test02_mono.json"
tta_test02_poly_path = f"{config.metadata_json_file}/tta/test/test02_poly.json"

tta_test03_path = f"{config.metadata_json_file}/tta/test/test03.json"
tta_test03_mono_path = f"{config.metadata_json_file}/tta/test/test03_mono.json"
tta_test03_poly_path = f"{config.metadata_json_file}/tta/test/test03_poly.json"

tta_test04_path = f"{config.metadata_json_file}/tta/test/test04.json"
tta_test04_mono_path = f"{config.metadata_json_file}/tta/test/test04_mono.json"
tta_test04_poly_path = f"{config.metadata_json_file}/tta/test/test04_poly.json"

## ata deepfake
ata_test01_path = f"{config.metadata_json_file}/ata/test/test01.json"
ata_test01_mono_path = f"{config.metadata_json_file}/ata/test/test01_mono.json"
ata_test01_poly_path = f"{config.metadata_json_file}/ata/test/test01_poly.json"

ata_test02_path = f"{config.metadata_json_file}/ata/test/test02.json"
ata_test02_mono_path = f"{config.metadata_json_file}/ata/test/test02_mono.json"
ata_test02_poly_path = f"{config.metadata_json_file}/ata/test/test02_poly.json"

ata_test03_path = f"{config.metadata_json_file}/ata/test/test03.json"
ata_test03_mono_path = f"{config.metadata_json_file}/ata/test/test03_mono.json"
ata_test03_poly_path = f"{config.metadata_json_file}/ata/test/test03_poly.json"

ata_test04_path = f"{config.metadata_json_file}/ata/test/test04.json"
ata_test04_mono_path = f"{config.metadata_json_file}/ata/test/test04_mono.json"
ata_test04_poly_path = f"{config.metadata_json_file}/ata/test/test04_poly.json"

# # begin generate
tta_test01_data, tta_test01_mono_data, tta_test01_poly_data = [], [], []
tta_test02_data, tta_test02_mono_data, tta_test02_poly_data = [], [], []
tta_test03_data, tta_test03_mono_data, tta_test03_poly_data = [], [], []
tta_test04_data, tta_test04_mono_data, tta_test04_poly_data = [], [], []

ata_test01_data, ata_test01_mono_data, ata_test01_poly_data = [], [], []
ata_test02_data, ata_test02_mono_data, ata_test02_poly_data = [], [], []
ata_test03_data, ata_test03_mono_data, ata_test03_poly_data = [], [], []
ata_test04_data, ata_test04_mono_data, ata_test04_poly_data = [], [], []

nums = metadata.shape[0]
for i in tqdm(range(nums)):
    wavename = metadata.iloc[i,0]
    dataset = metadata.iloc[i,2]
    faketype = metadata.iloc[i,3]
    generator = metadata.iloc[i,4]
    
    if faketype == 'real':
        real_info = {
                'file_path': f'{config.envsdd_test_folder}/{wavename}',
                'label': 'real',
                'attack_type': 'real',
                'generative_model': 'real',
            }
        if 'TUT' in dataset or 'Urban' in dataset: # 1,3,4,5
            tta_test01_data.append(real_info)
            ata_test01_data.append(real_info)
            tta_test02_data.append(real_info)
            ata_test02_data.append(real_info)
        if 'Urban' in dataset: # 1
            tta_test01_mono_data.append(real_info)
            ata_test01_mono_data.append(real_info)
            tta_test02_mono_data.append(real_info)
            ata_test02_mono_data.append(real_info)
        if 'TUT' in dataset: # 3,4,5
            tta_test01_poly_data.append(real_info)
            ata_test01_poly_data.append(real_info)
            tta_test02_poly_data.append(real_info)
            ata_test02_poly_data.append(real_info)
        if 'DCASE' in dataset or 'Clotho' in dataset: # 2,6
            tta_test03_data.append(real_info)
            ata_test03_data.append(real_info)
            tta_test04_data.append(real_info)
            ata_test04_data.append(real_info)
        if 'DCASE' in dataset: # 2
            tta_test03_mono_data.append(real_info)
            ata_test03_mono_data.append(real_info)
            tta_test04_mono_data.append(real_info)
            ata_test04_mono_data.append(real_info)
        if 'Clotho' in dataset: # 6
            tta_test03_poly_data.append(real_info)
            ata_test03_poly_data.append(real_info)
            tta_test04_poly_data.append(real_info)
            ata_test04_poly_data.append(real_info)

    if faketype == 'ata':
        if 'TUT' in dataset or 'Urban' in dataset: # 1,3,4,5
            if generator == 'audioldm1':
                ata_test01_data.append({
                'file_path': f'{config.envsdd_test_folder}/{wavename}',
                'label': 'fake',
                'attack_type': 'ata',
                'generative_model': 'audioldm1',
            })
            elif generator == 'audioldm2':
                ata_test02_data.append({
                'file_path': f'{config.envsdd_test_folder}/{wavename}',
                'label': 'fake',
                'attack_type': 'ata',
                'generative_model': 'audioldm2',
            })
        if 'Urban' in dataset: # 1
            if generator == 'audioldm1':
                ata_test01_mono_data.append({
                'file_path': f'{config.envsdd_test_folder}/{wavename}',
                'label': 'fake',
                'attack_type': 'ata',
                'generative_model': 'audioldm1',
            })
            elif generator == 'audioldm2':
                ata_test02_mono_data.append({
                'file_path': f'{config.envsdd_test_folder}/{wavename}',
                'label': 'fake',
                'attack_type': 'ata',
                'generative_model': 'audioldm2',
            })
        if 'TUT' in dataset: # 3,4,5
            if generator == 'audioldm1':
                ata_test01_poly_data.append({
                'file_path': f'{config.envsdd_test_folder}/{wavename}',
                'label': 'fake',
                'attack_type': 'ata',
                'generative_model': 'audioldm1',
            })
            elif generator == 'audioldm2':
                ata_test02_poly_data.append({
                'file_path': f'{config.envsdd_test_folder}/{wavename}',
                'label': 'fake',
                'attack_type': 'ata',
                'generative_model': 'audioldm2',
            })
        if 'DCASE' in dataset or 'Clotho' in dataset: # 2,6
            if generator == 'audioldm1':
                ata_test03_data.append({
                'file_path': f'{config.envsdd_test_folder}/{wavename}',
                'label': 'fake',
                'attack_type': 'ata',
                'generative_model': 'audioldm1',
            })
            elif generator == 'audioldm2':
                ata_test04_data.append({
                'file_path': f'{config.envsdd_test_folder}/{wavename}',
                'label': 'fake',
                'attack_type': 'ata',
                'generative_model': 'audioldm2',
            })
        if 'DCASE' in dataset: # 2
            if generator == 'audioldm1':
                ata_test03_mono_data.append({
                'file_path': f'{config.envsdd_test_folder}/{wavename}',
                'label': 'fake',
                'attack_type': 'ata',
                'generative_model': 'audioldm1',
            })
            elif generator == 'audioldm2':
                ata_test04_mono_data.append({
                'file_path': f'{config.envsdd_test_folder}/{wavename}',
                'label': 'fake',
                'attack_type': 'ata',
                'generative_model': 'audioldm2',
            })
        if 'Clotho' in dataset: # 6
            if generator == 'audioldm1':
                ata_test03_poly_data.append({
                'file_path': f'{config.envsdd_test_folder}/{wavename}',
                'label': 'fake',
                'attack_type': 'ata',
                'generative_model': 'audioldm1',
            })
            elif generator == 'audioldm2':
                ata_test04_poly_data.append({
                'file_path': f'{config.envsdd_test_folder}/{wavename}',
                'label': 'fake',
                'attack_type': 'ata',
                'generative_model': 'audioldm2',
            })
    
    if faketype == 'tta':
        if 'TUT' in dataset or 'Urban' in dataset: # 1,3,4,5
            if generator == 'audioldm1' or generator == 'audioldm2' or generator == 'audiogen':
                tta_test01_data.append({
                'file_path': f'{config.envsdd_test_folder}/{wavename}',
                'label': 'fake',
                'attack_type': 'tta',
                'generative_model': generator,
            })
            elif generator == 'audiolcm' or generator == 'tangoflux':
                tta_test02_data.append({
                'file_path': f'{config.envsdd_test_folder}/{wavename}',
                'label': 'fake',
                'attack_type': 'tta',
                'generative_model': generator,
            })
        if 'Urban' in dataset: # 1
            if generator == 'audioldm1' or generator == 'audioldm2' or generator == 'audiogen':
                tta_test01_mono_data.append({
                'file_path': f'{config.envsdd_test_folder}/{wavename}',
                'label': 'fake',
                'attack_type': 'tta',
                'generative_model': generator,
            })
            elif generator == 'audiolcm' or generator == 'tangoflux':
                tta_test02_mono_data.append({
                'file_path': f'{config.envsdd_test_folder}/{wavename}',
                'label': 'fake',
                'attack_type': 'tta',
                'generative_model': generator,
            })
        if 'TUT' in dataset: # 3,4,5
            if generator == 'audioldm1' or generator == 'audioldm2' or generator == 'audiogen':
                tta_test01_poly_data.append({
                'file_path': f'{config.envsdd_test_folder}/{wavename}',
                'label': 'fake',
                'attack_type': 'tta',
                'generative_model': generator,
            })
            elif generator == 'audiolcm' or generator == 'tangoflux':
                tta_test02_poly_data.append({
                'file_path': f'{config.envsdd_test_folder}/{wavename}',
                'label': 'fake',
                'attack_type': 'tta',
                'generative_model': generator,
            })
        if 'DCASE' in dataset or 'Clotho' in dataset: # 2,6
            if generator == 'audioldm1' or generator == 'audioldm2' or generator == 'audiogen':
                tta_test03_data.append({
                'file_path': f'{config.envsdd_test_folder}/{wavename}',
                'label': 'fake',
                'attack_type': 'tta',
                'generative_model': generator,
            })
            elif generator == 'audiolcm' or generator == 'tangoflux':
                tta_test04_data.append({
                'file_path': f'{config.envsdd_test_folder}/{wavename}',
                'label': 'fake',
                'attack_type': 'tta',
                'generative_model': generator,
            })
        if 'DCASE' in dataset: # 2
            if generator == 'audioldm1' or generator == 'audioldm2' or generator == 'audiogen':
                tta_test03_mono_data.append({
                'file_path': f'{config.envsdd_test_folder}/{wavename}',
                'label': 'fake',
                'attack_type': 'tta',
                'generative_model': generator,
            })
            elif generator == 'audiolcm' or generator == 'tangoflux':
                tta_test04_mono_data.append({
                'file_path': f'{config.envsdd_test_folder}/{wavename}',
                'label': 'fake',
                'attack_type': 'tta',
                'generative_model': generator,
            })
        if 'Clotho' in dataset: # 6
            if generator == 'audioldm1' or generator == 'audioldm2' or generator == 'audiogen':
                tta_test03_poly_data.append({
                'file_path': f'{config.envsdd_test_folder}/{wavename}',
                'label': 'fake',
                'attack_type': 'tta',
                'generative_model': generator,
            })
            elif generator == 'audiolcm' or generator == 'tangoflux':
                tta_test04_poly_data.append({
                'file_path': f'{config.envsdd_test_folder}/{wavename}',
                'label': 'fake',
                'attack_type': 'tta',
                'generative_model': generator,
            })
    
# Save metadata to JSON files
## tta
with open(tta_test01_path, "w") as f:
    json.dump(tta_test01_data, f, indent=4)
with open(tta_test02_path, "w") as f:
    json.dump(tta_test02_data, f, indent=4)
with open(tta_test03_path, "w") as f:
    json.dump(tta_test03_data, f, indent=4)
with open(tta_test04_path, "w") as f:
    json.dump(tta_test04_data, f, indent=4)

with open(tta_test01_mono_path, "w") as f:
    json.dump(tta_test01_mono_data, f, indent=4)
with open(tta_test02_mono_path, "w") as f:
    json.dump(tta_test02_mono_data, f, indent=4)
with open(tta_test03_mono_path, "w") as f:
    json.dump(tta_test03_mono_data, f, indent=4)
with open(tta_test04_mono_path, "w") as f:
    json.dump(tta_test04_mono_data, f, indent=4)

with open(tta_test01_poly_path, "w") as f:
    json.dump(tta_test01_poly_data, f, indent=4)
with open(tta_test02_poly_path, "w") as f:
    json.dump(tta_test02_poly_data, f, indent=4)
with open(tta_test03_poly_path, "w") as f:
    json.dump(tta_test03_poly_data, f, indent=4)
with open(tta_test04_poly_path, "w") as f:
    json.dump(tta_test04_poly_data, f, indent=4)

## ata
with open(ata_test01_path, "w") as f:
    json.dump(ata_test01_data, f, indent=4)
with open(ata_test02_path, "w") as f:
    json.dump(ata_test02_data, f, indent=4)
with open(ata_test03_path, "w") as f:
    json.dump(ata_test03_data, f, indent=4)
with open(ata_test04_path, "w") as f:
    json.dump(ata_test04_data, f, indent=4)

with open(ata_test01_mono_path, "w") as f:
    json.dump(ata_test01_mono_data, f, indent=4)
with open(ata_test02_mono_path, "w") as f:
    json.dump(ata_test02_mono_data, f, indent=4)
with open(ata_test03_mono_path, "w") as f:
    json.dump(ata_test03_mono_data, f, indent=4)
with open(ata_test04_mono_path, "w") as f:
    json.dump(ata_test04_mono_data, f, indent=4)

with open(ata_test01_poly_path, "w") as f:
    json.dump(ata_test01_poly_data, f, indent=4)
with open(ata_test02_poly_path, "w") as f:
    json.dump(ata_test02_poly_data, f, indent=4)
with open(ata_test03_poly_path, "w") as f:
    json.dump(ata_test03_poly_data, f, indent=4)
with open(ata_test04_poly_path, "w") as f:
    json.dump(ata_test04_poly_data, f, indent=4)
