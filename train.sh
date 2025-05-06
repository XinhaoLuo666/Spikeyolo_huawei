export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0

python train.py --config ./configs/SpikeYOLO/yolov8s.yaml  --ms_mode 1