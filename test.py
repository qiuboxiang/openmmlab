from ppvideo import PaddleVideo
clas = PaddleVideo(model_name='ppTSM_v2', use_gpu=False)
video_file='data/example.avi'
clas.predict(video_file)