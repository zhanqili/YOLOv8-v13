from datetime import datetime

from ultralytics import YOLO

if __name__ == '__main__':
    # 记录训练开始时间
    start_time = datetime.now()

    model = YOLO(model=r'yolov13n.pt')  # .pt类型的文件是从预训练模型的基础上进行训练
    #model = YOLO(model=r'ultralytics/cfg/models/v9/yolov9n.yaml')  # .yaml文件是从零开始训练，后缀n s x等是不同参数的模型，看具体应用场景

    # 开始训练
    model.train(data=r'D:\PycharmProjects\Pytorch_Yolo\ultralytics-yolo11\ultralytics\datasets\TT100K\TT100K.yaml',  # 填入训练数据集配置文件的路径
                imgsz=640,  # 该参数代表输入图像的尺寸，指定为 640x640
                epochs=100,  # 该参数代表训练的轮数，默认100
                # 但一般对于新数据集，我们还不知道这个数据集学习的难易程度，可以加大轮数，例如300，来找到更佳性能
                patience=20,
                batch=8,  # 每个批次中的图像数量。在训练过程中，数据被分成多个批次进行处理，每个批次包含一定数量的图像。
                # 这个参数确定了每个批次中包含的图像数量。特殊的是，如果设置为-1，则会自动调整批次大小，至你的显卡能容纳的最多图像数量。
                cache=False,
                workers=4,  # 该参数代表数据加载的工作线程数，出现显存爆了的话可以设置为0，默认是8

                device='0',  # 该参数代表用哪个显卡训练，0是GPU  CPU就是直接写CPU

                resume=False,  # 该参数代表是否从上一次中断的训练状态继续训练。设置为False表示从头开始新的训练。如果设置为True，则会加载上一次训练的模型权重和优化器状态，继续训练。

                name='yolo13_res',  # 该参数代表保存的结果文件夹名称（文件保存在上面project路径里）
                )