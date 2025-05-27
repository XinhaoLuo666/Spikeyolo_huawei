#代码说明
整体架构基于MindYOLO实现 ，.configs/SpikeYOLO 为SpikeYOLO的配置文件，其中，SpikeYOLO_base.yaml是具体网络架构。
所有网络层定义均位于mindyolo\models\layers\bottleneck.py
(如MS_DownSampling，MS_StandardConv，MS_AllConvBlock等在SpikeYOLO_base.yaml中出现的函数)
I-LIF神经元的实现同样位于改文件，为class mem_update.
其他训练方法，数据读取等模块未做改动。
