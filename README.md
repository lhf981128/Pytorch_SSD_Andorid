# Pytorch_SSD_Andorid
记录自己的andorid部署过程，网上有用ncnn部署的，好像要用到c++，感兴趣的可以了解一下。  
这个博主写的也很全，可以参考下:  
[MobileNetSSD通过Ncnn前向推理框架在Android端的使用-](https://blog.csdn.net/qq_33431368/article/details/85019234)  
代码部分：主要是在官网案例上结合自己的SSD代码做了点改动，先改变图片大小，再标准化，对输出的结果做极大值抑制。  
有一点我觉得可以改进，SSD预测代码部分（Predict.py）是将每个类别的预测分别进行极大值抑制，我直接将除背景外的所有结果做极大值抑制
