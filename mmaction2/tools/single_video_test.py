from mmaction.apis import inference_recognizer, init_recognizer

config_path = r'D:\openmmlab\mmaction2\configs\recognition\slowonly\slowonly_r50_8xb16-4x16x1-256e_kinetics400-rgb.py'
checkpoint_path = r'D:\openmmlab\mmaction2\work_dirs\slowonly_r50_8xb16-4x16x1-256e_kinetics400-rgb\best_acc_top1_epoch_35.pth' # 可以是本地路径
img_path = r'D:\openmmlab\mmaction2\data\ShuttleSet\set\Kento_MOMOTA_CHOU_Tien_Chen_Fuzhou_Open_2019_Finals\B_split\B_F _ MS _ Kento MOMOTA (JPN) [1] vs. CHOU Tien Chen (TPE) [2] _ BWF 2019_0-07-55.mp4'   # 您可以指定自己的图片路径

# 从配置文件和权重文件中构建模型
model = init_recognizer(config_path, checkpoint_path, device="cuda:0")  # device 可以是 'cuda:0'
# 对单个视频进行测试
result = inference_recognizer(model, img_path)

print(result)