## ASD
### 描述

从视频中识别与音轨对应的说话人，并且确定裁剪位置和时间片位置，保存在pyavi的clips.json，裁剪出的视频位于clips文件夹内

基于[Light-ASD](https://github.com/Junhua-Liao/Light-ASD)二次开发

### 环境配置

```
conda create -n asd python==3.9.0
conda activate asd
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirement.txt
```

### 模型下载
配置好oss
```
sh download.sh
```

### 使用
1. 视频放在一个文件夹里，给定文件夹地址；
2. 处理部分视频的话，给出视频名，用逗号分隔，不提供默认处理所有视频；
3. 可以设置并行处理视频数量；
```
python video2clips.py --video_dir {dir of videos} --vid_id {video names separated with ','} --num_worker 2
```