# EnvSDD
Official code for EnvSDD (Environmental Sound Deepfake Detection)

Arxiv: 

Abstact:
Audio generation systems now create very realistic soundscapes that can enhance media production, but also pose potential risks. Several studies have examined deepfakes in speech or singing voice. However, environmental sounds have different characteristics, which may make methods for detecting speech and singing deepfakes less effective for real-world sounds. In addition, existing datasets for environmental sound deepfake detection are limited in scale and audio types. To address this gap, we introduce EnvSDD, the first large-scale curated dataset designed for this task, consisting of 45.25 hours of real and 316.74 hours of fake audio. The test set includes diverse conditions to evaluate the generalizability, such as unseen generation models and unseen datasets. We also propose an audio deepfake detection system, based on a pre-trained audio foundation model. Results on EnvSDD show that our proposed system outperforms the state-of-the-art systems from speech and singing domains.

More information please refer to our demo page: https://envsdd.github.io/

## Dataset

Detailed structure of the dataset is shown in the following figure:

<p align="center">
  <img src="figs/dataset.png" alt="Dataset" width="600" />
</p>

- EnvSDD-Development: you can download from [https://zenodo.org/records/15220951](https://zenodo.org/records/15220951)
- EnvSDD-Test: you can download from [https://zenodo.org/records/15241138](https://zenodo.org/records/15241138)
- EnvSDD-Remain: available soon

Some parts of the dataset are temporarily not publicly available because we plan to host a challenge. We aim to ensure fairness and prevent data leakage prior to the event. The dataset will be made publicly available after the competition concludes. If you are interested in early access for research purposes or have any questions, please feel free to contact us at yinhan@mail.nwpu.edu.cn.
Thank you for your understanding!

## Train
- Step 1: prepare environment by running: <mark>pip install -r requirements.txt<mark>
- Step 2: prepare .json file for development by running: <mark>python generate_json_dev.py<mark>
<p align="center">
  <img src="figs/generate_json_dev.png" alt="Dataset" width="800" />
</p>
- Step 3: train your deepfake models by running: <mark>python main.py --exp_id 0 --model model_name<mark>

3 models are supported now: aasist, w2v2_aasist, beats_aasist.

PS: There are lots of arguments (eg. batchsize, eval ...) in the main.py, you can directly set in the terminal. It is ok if you do not have test.json during training, test.json will only be used when you activate "eval".
