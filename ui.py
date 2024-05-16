# -*- coding: utf-8 -*-
"""
-------------------------------------------------
File Name: ui.py.py
Author: GHZ
Create Date: 2024/2/7
Description：
-------------------------------------------------
"""
# -*- coding: utf-8 -*-
# 应该在界面启动的时候就将模型加载出来，设置tmp的目录来放中间的处理结果
import shutil
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import cv2
import torch
import os.path as osp
from Models.dnetv25v3 import TNet
import numpy as np
from Data import dataloaders
from PIL import Image
from torchvision import transforms

# 窗口主类
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda')

transform = transforms.Compose([
    transforms.ToTensor()
])

class MainWindow(QTabWidget):
    # 基本配置不动，然后只动第三个界面
    def __init__(self):
        # 初始化界面
        super().__init__()
        self.setWindowTitle('结直肠镜息肉图像分割系统')
        self.resize(1200, 800)
        self.setWindowIcon(QIcon("images/UI/lufei.png"))
        # 图片读取进程
        self.output_size = 480
        self.img2predict = "" #/home/ge/data_set/Kvasir/images/cju5i39mreass0817au8p22zy.jpg
        # # 初始化视频读取线程
        self.origin_shape = (352, 352)
        # 加载网络，图片单通道，分类为1。
        net = TNet()
        # 将网络拷贝到deivce中
        # net.to(device=device)
        # 加载模型参数
        # net.load_state_dict(torch.load("./trained weight/Trained modelsv25xiwt/FCBFormer_Kvasir.pt",strict=False))  # todo 模型位置
        state_dict = torch.load(
            "./trained weight/Trained modelsv25v3_73/FCBFormer_Kvasir.pt"
        )
        net.load_state_dict(state_dict["model_state_dict"], strict=False)
        # 测试模式
        net.eval()
        self.model = net.cuda()
        self.initUI()

    '''
    ***界面初始化***
    '''

    def initUI(self):
        # 图片检测子界面
        font_title = QFont('楷体', 16)
        font_main = QFont('楷体', 14)
        # 图片识别界面, 两个按钮，上传图片和显示结果
        img_detection_widget = QWidget()
        img_detection_layout = QVBoxLayout()
        img_detection_title = QLabel("图片分割功能")
        img_detection_title.setFont(font_title)
        mid_img_widget = QWidget()
        mid_img_layout = QHBoxLayout()
        self.left_img = QLabel()
        self.right_img = QLabel()
        self.left_img.setPixmap(QPixmap("images/UI/up.jpeg"))
        self.right_img.setPixmap(QPixmap("images/UI/right.jpeg"))
        self.left_img.setAlignment(Qt.AlignCenter)
        self.right_img.setAlignment(Qt.AlignCenter)
        mid_img_layout.addWidget(self.left_img)
        mid_img_layout.addStretch(0)
        mid_img_layout.addWidget(self.right_img)
        mid_img_widget.setLayout(mid_img_layout)
        up_img_button = QPushButton("上传图片")
        det_img_button = QPushButton("开始检测")
        up_img_button.clicked.connect(self.upload_img)
        det_img_button.clicked.connect(self.detect_img)
        up_img_button.setFont(font_main)
        det_img_button.setFont(font_main)
        up_img_button.setStyleSheet("QPushButton{color:white}"
                                    "QPushButton:hover{background-color: rgb(2,110,180);}"
                                    "QPushButton{background-color:rgb(48,124,208)}"
                                    "QPushButton{border:2px}"
                                    "QPushButton{border-radius:5px}"
                                    "QPushButton{padding:5px 5px}"
                                    "QPushButton{margin:5px 5px}")
        det_img_button.setStyleSheet("QPushButton{color:white}"
                                     "QPushButton:hover{background-color: rgb(2,110,180);}"
                                     "QPushButton{background-color:rgb(48,124,208)}"
                                     "QPushButton{border:2px}"
                                     "QPushButton{border-radius:5px}"
                                     "QPushButton{padding:5px 5px}"
                                     "QPushButton{margin:5px 5px}")
        img_detection_layout.addWidget(img_detection_title, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(mid_img_widget, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(up_img_button)
        img_detection_layout.addWidget(det_img_button)
        img_detection_widget.setLayout(img_detection_layout)

        # todo 关于界面
        about_widget = QWidget()
        about_layout = QVBoxLayout()
        about_title = QLabel('欢迎使用医学影像分割系统\n')  # todo 修改欢迎词语
        about_title.setFont(QFont('楷体', 18))
        about_title.setAlignment(Qt.AlignCenter)
        about_img = QLabel()
        about_img.setPixmap(QPixmap('images/UI/qq.png'))
        about_img.setAlignment(Qt.AlignCenter)

        about_layout.addWidget(about_title)
        about_layout.addStretch()
        about_layout.addWidget(about_img)
        about_layout.addStretch()
        #about_layout.addWidget(label_super)
        about_widget.setLayout(about_layout)

        pre_widget = QWidget()
        pre_layout = QVBoxLayout()
        pre_title = QLabel('欢迎使用医学影像分割系统\n')
        pre_title.setFont(QFont('楷体', 18))
        pre_title.setAlignment(Qt.AlignCenter)
        pre_img = QLabel()
        pre_img.setPixmap(QPixmap('images/UI/qq.png'))
        pre_img.setAlignment(Qt.AlignCenter)
        pre_layout.addWidget(pre_title)
        pre_layout.addStretch()
        pre_layout.addWidget(pre_img)
        pre_layout.addStretch()
        pre_widget.setLayout(pre_layout)

        post_widget = QWidget()
        post_layout = QVBoxLayout()
        post_title = QLabel('欢迎使用医学影像分割系统\n')
        post_title.setFont(QFont('楷体', 18))
        post_title.setAlignment(Qt.AlignCenter)
        post_img = QLabel()
        post_img.setPixmap(QPixmap('images/UI/qq.png'))
        post_img.setAlignment(Qt.AlignCenter)
        post_layout.addWidget(post_title)
        post_layout.addStretch()
        post_layout.addWidget(post_img)
        post_layout.addStretch()
        post_widget.setLayout(post_layout)

        self.left_img.setAlignment(Qt.AlignCenter)
        self.addTab(img_detection_widget, '图片检测')
        self.setTabIcon(0, QIcon('images/UI/lufei.png'))


    '''
    ***上传图片***
    '''

    def upload_img(self):
        # 选择录像文件进行读取
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg')
        print(fileName)
        if fileName:
            suffix = fileName.split(".")[-1]
            save_path = osp.join("./uires", "tmp_upload." + suffix)
            shutil.copy(fileName, save_path)
            # 应该调整一下图片的大小，然后统一防在一起
            im0 = cv2.imread(save_path)
            resize_scale = self.output_size / im0.shape[0]
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite("./uires/tmp_upload.jpg", im0)
            # self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))
            self.img2predict = fileName
            self.origin_shape = (im0.shape[1], im0.shape[0])
            self.left_img.setPixmap(QPixmap("./uires/tmp_upload.jpg"))


    '''
    ***检测图片***
    '''
    def detect_img(self):
        model = self.model
        output_size = self.output_size
        source = self.img2predict  # file/dir/URL/glob, 0 for webcam
        img = Image.open(source)
        temp = max(img.size)
        mask = Image.new('RGB', (temp, temp))
        mask.paste(img, (0, 0))
        mask = mask.resize((480, 480))
        img_data = transform(mask).cuda()
        img_data = torch.unsqueeze(img_data, dim=0)
        output, _ = model(img_data)
        predicted_map = np.array(output.cpu().detach().numpy())
        predicted_map = np.squeeze(predicted_map)
        predicted_map = predicted_map > 0
        cv2.imwrite("./uires/single_result.jpg", predicted_map * 255)
        self.right_img.setPixmap(QPixmap("./uires/single_result.jpg"))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
