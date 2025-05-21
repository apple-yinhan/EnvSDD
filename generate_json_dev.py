import sys
import config
import pandas as pd
from tqdm import tqdm
import json
import os

metadata = pd.read_csv(config.envsdd_dev_split_path)
# # # Output metadata jsons
os.makedirs(config.metadata_json_file, exist_ok=True)
os.makedirs(f"{config.metadata_json_file}/tta/dev", exist_ok=True)
os.makedirs(f"{config.metadata_json_file}/ata/dev", exist_ok=True)

tta_train_meta_path = f"{config.metadata_json_file}/tta/dev/train.json"
tta_valid_meta_path = f"{config.metadata_json_file}/tta/dev/valid.json"
ata_train_meta_path = f"{config.metadata_json_file}/ata/dev/train.json"
ata_valid_meta_path = f"{config.metadata_json_file}/ata/dev/valid.json"

tta_train_data, tta_valid_data = [],[]
ata_train_data, ata_valid_data = [],[]
nums = metadata.shape[0]

for i in tqdm(range(nums)):
    wavename = metadata.iloc[i,0]
    dataset = metadata.iloc[i,1]
    usage = metadata.iloc[i,-1]
    real_info = {
        'file_path': f'{config.envsdd_dev_folder}/real_audio/{dataset}/{wavename}',
        'label': 'real',
        'attack_type': 'real',
        'generative_model': 'real',
    }
    if usage == 'train':
        tta_train_data.append(real_info)
        for model_name in ['audiogen', 'audioldm1', 'audioldm2']:
            tta_train_data.append({
                'file_path': f'{config.envsdd_dev_folder}/fake_audio/TTA/{model_name}/{dataset}/{wavename}',
                'label': 'fake',
                'attack_type': 'tta',
                'generative_model': model_name,
            })
        ata_train_data.append(real_info)
        for model_name in ['audioldm1']:
            ata_train_data.append({
                'file_path': f'{config.envsdd_dev_folder}/fake_audio/ATA/{model_name}/{dataset}/{wavename}',
                'label': 'fake',
                'attack_type': 'ata',
                'generative_model': model_name,
            })
    elif usage == 'validation':
        tta_valid_data.append(real_info)
        for model_name in ['audiogen', 'audioldm1', 'audioldm2']:
            tta_valid_data.append({
                'file_path': f'{config.envsdd_dev_folder}/fake_audio/TTA/{model_name}/{dataset}/{wavename}',
                'label': 'fake',
                'attack_type': 'tta',
                'generative_model': model_name,
            })
        ata_valid_data.append(real_info)
        for model_name in ['audioldm1']:
            ata_valid_data.append({
                'file_path': f'{config.envsdd_dev_folder}/fake_audio/ATA/{model_name}/{dataset}/{wavename}',
                'label': 'fake',
                'attack_type': 'ata',
                'generative_model': model_name,
            })

print(f'TTA Train:{len(tta_train_data)}, ATA Train:{len(ata_train_data)}.')
print(f'TTA Validation:{len(tta_valid_data)}, ATA Validation:{len(ata_valid_data)}.')

# # Save metadata to JSON files
with open(tta_train_meta_path, "w") as f:
    json.dump(tta_train_data, f, indent=4)
with open(tta_valid_meta_path, "w") as f:
    json.dump(tta_valid_data, f, indent=4)

with open(ata_train_meta_path, "w") as f:
    json.dump(ata_train_data, f, indent=4)
with open(ata_valid_meta_path, "w") as f:
    json.dump(ata_valid_data, f, indent=4)


