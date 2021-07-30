# Dependence

```sh
$ pip3 install matplotlib
```

# VIM3

## Yolov3

```sh
$ cd darknet
$ python3 yolov3-picture.py --nb-file ./models/VIM3/yolov3.nb --so-lib ./libs/libnn_yolov3.so --input-picture-path ./data/1080p.bmp
$ python3 yolov3-cap.py --nb-file ./models/VIM3/yolov3.nb --so-lib ./libs/libnn_yolov3.so --video-device X
```

## Inception v3

```sh
$ cd tensorflow
$ python3 inceptionv3.py --nb-file ./models/VIM3/inceptionv3.nb --so-lib ./libs/libnn_inceptionv3.so --input-picture ./data/goldfish_299x299.jpg
```

# VIM3L

## Yolov3

```sh
$ cd darknet
$ python3 yolov3-picture.py --nb-file ./models/VIM3L/yolov3.nb --so-lib ./libs/libnn_yolov3.so --input-picture-path ./data/1080p.bmp
$ python3 yolov3-cap.py --nb-file ./models/VIM3L/yolov3.nb --so-lib ./libs/libnn_yolov3.so --video-device X
```

## Inception v3

```sh
$ cd tensorflow
$ python3 inceptionv3.py --nb-file ./models/VIM3L/inceptionv3.nb --so-lib ./libs/libnn_inceptionv3.so --input-picture ./data/goldfish_299x299.jpg
```
