# ML final Project -- Sound
---
1. [Kaggle Link](https://www.kaggle.com/c/freesound-audio-tagging)

2. Toolkit-version
  - keras 2.0.8
  - Tensorflow 1.4.0
  - h5py 2.7.1
  - sklearn 0.19.1
  - librosa 0.6.0
  - numpy, pandas
  - python standard Library

3. Usage

```shell
# data preprocessing
bash data_preprocessing.sh $input_data_root_dir_path
# download model
bash model_download.sh
# training
bash train.sh $duration(1, 2, 3, 5, 8, 13) $save_model_path
# predict
bash test.sh
```
Then, you can get the output `output.csv`.

Noted that your `$input_data_toor_dir` should be that:
```
root_dir/
├── audio_test
│   ├── *.wav
├── audio_train
│   ├── *.wav
├── sample_submission.csv
└── train.csv
```

