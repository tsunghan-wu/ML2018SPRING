import os
import wave
import librosa
import numpy as np
import pandas as pd

root_dir = sys.argv[2]
file = {
    'train': [os.path.join(root_dir, "train.csv"), os.path.join(root_dir, "audio_train")],
    'test' : [os.path.join(root_dir, "sample_submission.csv"), os.path.join(root_dir, "audio_test")]
}

data = {
    'train': pd.read_csv(file['train'][0]),
    'test' : pd.read_csv(file['test'][0]),
}
print (file)
exit()
AUDIO_duration = int(sys.argv[1])

def get_feature(mode):
    fname = [os.path.join(file[mode][1], x) for x in data[mode]['fname'].tolist()]
    ### global setting ###
    sample_rate = 44100
    audio_duration = AUDIO_duration
    n_mfcc = 40
    ######################
    ret = []
    for idx, x in enumerate(fname):
        if idx % 50 ==0:
            print(idx)
        X, sr = librosa.load(x, sr=sample_rate)
        X = np.resize(X, sample_rate*audio_duration)
        mfcc = librosa.feature.mfcc(X, sr=sample_rate, n_mfcc=n_mfcc)
        ret.append(mfcc)
    return np.array(ret)


def get_label():
    one_hot = pd.unique(data['train']['label']).tolist()
    Y = []
    for idx, row in data['train'].iterrows():
        Y.append(one_hot.index(row['label']))
    return np.array(Y), one_hot

trainY, index_list = get_label()
print(trainY.shape)
print(index_list)

testX = get_feature("test")
print(testX.shape)
trainX = get_feature("train")
print(trainX.shape)

np.save("./data/testX" + str(AUDIO_duration) + ".npy", testX)



