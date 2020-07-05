import pdb

import torch
import torchvision
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.ssd.data_preprocessing import PredictionTransform
# import numpy as np
from vision.ssd.config import mobilenetv1_ssd_config as config
import cv2

orig_image = cv2.imread('F:/DataSet/my_ssd_data/Images/our_16.png')
# pdb.set_trace()
image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
# pdb.set_trace()
transform = PredictionTransform(config.image_size,config.image_mean,config.image_std)
transfor_example = transform(image)
transfor_example = transfor_example.unsqueeze(0)
# pdb.set_trace()
class_names = [name.strip() for name in open('models/voc-model-labels.txt').readlines()]
net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
net.load('models/mb2-ssd-lite-Epoch-149-Loss-0.6894857048988342.pth')
# pdb.set_trace()
net.eval()
pdb.set_trace()
# example = torch.rand(1, 3, 300, 300)
traced_script_module = torch.jit.trace(net, transfor_example)
# pdb.set_trace()
# print(traced_script_module)
# traced_script_module.save("ssd_model.pt")