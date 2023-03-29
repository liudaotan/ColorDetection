# A computer vision tool to compare glucose concentration.

## Introduction 
可以自动监测照片中的试管位置，并计算试管中的葡萄糖的颜色与对照组的近似程度。

## What's New?
1. 不需要指定对照组的葡萄糖样本的具体颜色
2. 可以自动探测图片中的试管所在位置
3. 按照和对照组中试管颜色的分布近似程度，给出待检样本的对应索引

## User Guidelines
1. 将待检测照片放在 data 文件夹下

2. 安装对应的库
```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

3. 运行指南
```angular2html
python3 start_detection.py img num1 num2
```

```
    parser.add_argument('img',  type=str, help='The name of image')
    parser.add_argument('num1',  type=int, help='The num of sample')
    parser.add_argument('num2',  type=int, help='The num of target sample')
    parser.add_argument('--threshold',  type=int, default=160, help='The threshold of object detection (usually in 100-200)')
    parser.add_argument('--debug',  type=int, default=0, help='The debug mode: 0 1 2')
   
```



