# Tiny Object Detection Challenge

## Train:

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=9001 tools/train_test_net.py --config configs/TinyPerson/FPN/baseline1/e2e_faster_rcnn_R_50_FPN_1x_cocostyle_baseline1.yaml
```

## Results

| AP50_tiny  | AP50_tiny1  | AP50_tiny2  | AP50_tiny3  | AP25_tiny  | AP75_tiny | mr50_tiny  |
|------------|-------------|-------------|-------------|------------|-----------|------------|
| 45.73 (25) | 27.99 (24)  | 49.41 (24)  | 57.78 (24)  | 67.93 (25) | 4.89 (25) | 88.18 (25) |

