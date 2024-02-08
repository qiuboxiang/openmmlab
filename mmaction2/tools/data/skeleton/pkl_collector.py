import os
import pickle
result = []
path = r"""D:\openmmlab\mmaction2\work_dirs\tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb\badminton_video\val_output"""
for d in os.listdir(path):
    if d.endswith('.pkl'):
        with open(os.path.join(path, d), 'rb') as f:
            content = pickle.load(f)
        result.append(content)
with open('val.pkl', 'wb') as out:
    pickle.dump(result, out, protocol=pickle.HIGHEST_PROTOCOL)
