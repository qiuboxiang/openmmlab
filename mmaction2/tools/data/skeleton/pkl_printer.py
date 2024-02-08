import pickle

f = open(r'D:\openmmlab\mmaction2\work_dirs\tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb\badminton_video\train_output\S004A003.pkl', 'rb')
data = pickle.load(f)
print(data)
