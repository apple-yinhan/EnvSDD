import json
import os
import pandas as pd
import config_v2 as config
from tqdm import tqdm

TYPE = "dev_track1" # dev_track1, dev_track2
os.makedirs(config.metadata_json_file, exist_ok=True)

if TYPE == "dev_track1":
    train_meta_path = f"{config.metadata_json_file}/dev_track1_train.json"
    valid_meta_path = f"{config.metadata_json_file}/dev_track1_valid.json"
    train_data, valid_data = [], []
    metadata = pd.read_csv(config.dev_track1_meta)
    nums = metadata.shape[0]
    for i in tqdm(range(nums)):
        wavename = metadata.iloc[i,0]
        dataset = metadata.iloc[i,1]
        usage = metadata.iloc[i,-1]
        real_info = {
            'file_path': f'{config.dev_track1_audio}/real_audio/{dataset}/{wavename}',
            'label': 'real',
            'attack_type': 'real',
            'generative_model': 'real',
        }
        if usage == 'train':
            train_data.append(real_info)
            for model_name in ['audiogen', 'audioldm1', 'audioldm2']:
                train_data.append({
                    'file_path': f'{config.dev_track1_audio}/fake_audio/TTA/{model_name}/{dataset}/{wavename}',
                    'label': 'fake',
                    'attack_type': 'tta',
                    'generative_model': model_name,
                })
            for model_name in ['audioldm1']:
                train_data.append({
                    'file_path': f'{config.dev_track1_audio}/fake_audio/ATA/{model_name}/{dataset}/{wavename}',
                    'label': 'fake',
                    'attack_type': 'ata',
                    'generative_model': model_name,
                })
        elif usage == 'validation':
            valid_data.append(real_info)
            for model_name in ['audiogen', 'audioldm1', 'audioldm2']:
                valid_data.append({
                    'file_path': f'{config.dev_track1_audio}/fake_audio/TTA/{model_name}/{dataset}/{wavename}',
                    'label': 'fake',
                    'attack_type': 'tta',
                    'generative_model': model_name,
                })
            for model_name in ['audioldm1']:
                valid_data.append({
                    'file_path': f'{config.dev_track1_audio}/fake_audio/ATA/{model_name}/{dataset}/{wavename}',
                    'label': 'fake',
                    'attack_type': 'ata',
                    'generative_model': model_name,
                })
elif TYPE == "dev_track2":
    train_meta_path = f"{config.metadata_json_file}/dev_track2_train.json"
    valid_meta_path = f"{config.metadata_json_file}/dev_track2_valid.json"
    train_data, valid_data = [], []
    # get data of dev track 1
    metadata = pd.read_csv(config.dev_track1_meta)
    nums = metadata.shape[0]
    for i in tqdm(range(nums)):
        wavename = metadata.iloc[i,0]
        dataset = metadata.iloc[i,1]
        usage = metadata.iloc[i,-1]
        real_info = {
            'file_path': f'{config.dev_track1_audio}/real_audio/{dataset}/{wavename}',
            'label': 'real',
            'attack_type': 'real',
            'generative_model': 'real',
        }
        if usage == 'train':
            train_data.append(real_info)
            for model_name in ['audiogen', 'audioldm1', 'audioldm2']:
                train_data.append({
                    'file_path': f'{config.dev_track1_audio}/fake_audio/TTA/{model_name}/{dataset}/{wavename}',
                    'label': 'fake',
                    'attack_type': 'tta',
                    'generative_model': model_name,
                })
            for model_name in ['audioldm1']:
                train_data.append({
                    'file_path': f'{config.dev_track1_audio}/fake_audio/ATA/{model_name}/{dataset}/{wavename}',
                    'label': 'fake',
                    'attack_type': 'ata',
                    'generative_model': model_name,
                })
        elif usage == 'validation':
            valid_data.append(real_info)
            for model_name in ['audiogen', 'audioldm1', 'audioldm2']:
                valid_data.append({
                    'file_path': f'{config.dev_track1_audio}/fake_audio/TTA/{model_name}/{dataset}/{wavename}',
                    'label': 'fake',
                    'attack_type': 'tta',
                    'generative_model': model_name,
                })
            for model_name in ['audioldm1']:
                valid_data.append({
                    'file_path': f'{config.dev_track1_audio}/fake_audio/ATA/{model_name}/{dataset}/{wavename}',
                    'label': 'fake',
                    'attack_type': 'ata',
                    'generative_model': model_name,
                })
    # get data from dev track 2
    metadata = pd.read_csv(config.dev_track2_meta)
    nums = metadata.shape[0]
    for i in tqdm(range(nums)):
        wavename, label, usage = metadata.iloc[i,0], metadata.iloc[i,1], metadata.iloc[i,2]
        if usage == "train":
            train_data.append({
                'file_path': f'{config.dev_track2_audio}/{wavename}',
                'label': label,
                'attack_type': 'xxx',
                'generative_model': 'xxx',
            })
        elif usage == "validation":
            valid_data.append({
                'file_path': f'{config.dev_track2_audio}/{wavename}',
                'label': label,
                'attack_type': 'xxx',
                'generative_model': 'xxx',
            })
elif TYPE == "eval_track1":
    eval_meta_path = f"{config.metadata_json_file}/eval_track1.json"
    eval_data = []
    metadata = pd.read_csv(config.eval_track1_meta)
    nums = metadata.shape[0]
    for i in tqdm(range(nums)):
        wavename, label = metadata.iloc[i,0], metadata.iloc[i,1]
        eval_data.append({
                'file_path': f'{config.eval_track1_audio}/{wavename}',
                'label': label,
                'attack_type': 'xxx',
                'generative_model': 'xxx',
            })
elif TYPE == "eval_track2":
    eval_meta_path = f"{config.metadata_json_file}/eval_track2.json"
    eval_data = []
    metadata = pd.read_csv(config.eval_track2_meta)
    nums = metadata.shape[0]
    for i in tqdm(range(nums)):
        wavename, label = metadata.iloc[i,0], metadata.iloc[i,1]
        eval_data.append({
                'file_path': f'{config.eval_track2_audio}/{wavename}',
                'label': label,
                'attack_type': 'xxx',
                'generative_model': 'xxx',
            })
elif TYPE == "test_track1":
    test_meta_path = f"{config.metadata_json_file}/test_track1.json"
    test_data = []
    metadata = pd.read_csv(config.test_track1_meta)
    nums = metadata.shape[0]
    for i in tqdm(range(nums)):
        wavename, source_wavename, source_dataset, faketype, generator = \
        metadata.iloc[i,0],metadata.iloc[i,1],metadata.iloc[i,2],metadata.iloc[i,3],metadata.iloc[i,4]
        if faketype == "real":
            label = "real"
        else:
            label = "fake"
        test_data.append({
                'file_path': f'{config.test_track1_audio}/{wavename}',
                'label': label,
                'attack_type': faketype,
                'generative_model': generator,
            })
elif TYPE == "test_track2":
    test_meta_path = f"{config.metadata_json_file}/test_track2.json"
    test_data = []
    metadata = pd.read_csv(config.test_track2_meta)
    nums = metadata.shape[0]
    for i in tqdm(range(nums)):
        wavename, source_wavename, source_dataset, faketype, generator = \
        metadata.iloc[i,0],metadata.iloc[i,1],metadata.iloc[i,2],metadata.iloc[i,3],metadata.iloc[i,4]
        if faketype == "real":
            label = "real"
        else:
            label = "fake"
        test_data.append({
                'file_path': f'{config.test_track2_audio}/{wavename}',
                'label': label,
                'attack_type': faketype,
                'generative_model': generator,
            })


if "test" in TYPE:
    # show information
    print(f"Type: {TYPE}")
    print(f'Test Samples: {len(test_data)}, saved in {test_meta_path}')
    # # Save metadata to JSON files
    with open(test_meta_path, "w") as f:
        json.dump(test_data, f, indent=4)
elif "eval" in TYPE:
    # show information
    print(f"Type: {TYPE}")
    print(f'Eval Samples: {len(eval_data)}, saved in {eval_meta_path}')
    # # Save metadata to JSON files
    with open(eval_meta_path, "w") as f:
        json.dump(eval_data, f, indent=4)
else:
    # show information
    print(f"Type: {TYPE}")
    print(f'Train Samples: {len(train_data)}, saved in {train_meta_path}')
    print(f'Validation Samples: {len(valid_data)}, saved in {valid_meta_path}')
    # # Save metadata to JSON files
    with open(train_meta_path, "w") as f:
        json.dump(train_data, f, indent=4)
    with open(valid_meta_path, "w") as f:
        json.dump(valid_data, f, indent=4)



