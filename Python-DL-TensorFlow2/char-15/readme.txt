人脸识别程序操作说明


一、目录说明
以下是指当前运行目录下
1、存放自己头像目录
input_dir = './data/face_recog/my_faces'
2、存放别人头像目录
input_dir = './data/face_recog/other_faces'
3、存放测试自己或别人的头像目录
input_dir='./data/face_recog/test_faces'

二、程序说明、执行步骤
把4个python脚本放在当前执行目录下
1、先处理自己、别人的头像
python process_image.py
2、构建模型、训练模型
python train_model.py
3、用新头像进行测试模型
python is_my_face.py

其中2、3将自动调用公共函数模块：share_fun.py


三、注意事项
1、需安装以下模块
除python3.6+及tensorflow 1+之外，需要另外安装opencv和dlib，如：
pip install opencv-python
pip install dlib
2、把python的lib目录加到环境变量中：
如linux，需在用户缺省目录下的.bashrc 文件中，添加以下语句
export LD_LIBRARY_PATH="/home/xxx/anaconda3/lib":"$LD_LIBRARY_PATH"
如果是windows环境，需要添加LD_LIBRARY_PATH环境变量



