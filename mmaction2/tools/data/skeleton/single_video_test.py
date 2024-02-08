from mmaction.apis import inference_skeleton, init_recognizer
import pickle

config_path = r'D:\openmmlab\mmaction2\configs\skeleton\posec3d\slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py'
checkpoint_path = r'D:\openmmlab\mmaction2\work_dirs\slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint\best_acc_top1_epoch_14.pth'  # 可以是本地路径
img_path = r'D:\openmmlab\mmaction2\work_dirs\tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb\badminton_video\train_output\S004A003.pkl'   # 您可以指定自己的图片路径

# 从配置文件和权重文件中构建模型
model = init_recognizer(config_path, checkpoint_path,
                        device="cuda:0")  # device 可以是 'cuda:0'
# 对单个视频进行测试
result = inference_skeleton(model, pickle.load(img_path), (1080, 1920))

print(result)
