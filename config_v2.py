beats_path = '/home/yinhan/codes/sep4noiseSED/pretrained_models/BEATS_iter3_plus_AS2M.pt'
w2v2_path = '/home/yinhan/codes/audio_deepfake/networks/wav2vec/xlsr2_300m.pt'

main_folder = '/home/yinhan/codes/envsdd_challenge_2026'
# development
dev_track1_audio = '/home/yinhan/codes/audio_deepfake/datasets/EnvSDD/released/development'
dev_track1_meta = f'{main_folder}/metadata/dev_track1.csv'

dev_track2_audio = f'{main_folder}/dataset/dev_track2'
dev_track2_meta = f'{main_folder}/metadata/dev_track2.csv'

# evaluation
eval_track1_audio = f'{main_folder}/dataset/eval_track1'
eval_track1_meta = f'{main_folder}/metadata/eval_track1.csv'

eval_track2_audio = f'{main_folder}/dataset/eval_track2'
eval_track2_meta = f'{main_folder}/metadata/eval_track2.csv'


# test (final ranking)
test_track1_audio = f'{main_folder}/dataset/test_track1'
test_track1_meta = f'{main_folder}/metadata/test_track1.csv'

test_track2_audio = f'{main_folder}/dataset/test_track2'
test_track2_meta = f'{main_folder}/metadata/test_track2.csv'

metadata_json_file = f'{main_folder}/jsons'


