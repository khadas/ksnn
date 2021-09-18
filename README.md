# Dependence

```sh
$ pip3 install matplotlib
```
# Darknet

## Yolov3

```sh
$ cd darknet
$ python3 yolov3-picture.py --nb-file ./models/VIM3/yolov3.nb --so-lib ./libs/libnn_yolov3.so --input-picture-path ./data/1080p.bmp
$ python3 yolov3-cap.py --nb-file ./models/VIM3/yolov3.nb --so-lib ./libs/libnn_yolov3.so --video-device X
```

# Tensorflow

## Inception v3

```sh
$ cd tensorflow
$ python3 inceptionv3.py --nb-file ./models/VIM3/inceptionv3.nb --so-lib ./libs/libnn_inceptionv3.so --input-picture ./data/goldfish_299x299.jpg
```

## Mobilenet SSD

```sh
$ cd tensorflow
$ python3 mobilenet_ssd_picture.py --nb-file ./models/VIM3/mobilenet_ssd.nb --so-lib ./libs/libnn_mobilenet_ssd.so --input-picture data/1080p.bmp
```

# Keras

## Xception

```sh
$ python3 xception.py --nb-file ./models/VIM3/xception.nb --so-lib ./libs/libnn_xception.so --input-picture data/goldfish_299x299.jpg
```

# caffe

## Mobilenet

```sh
$ python3 mobilenet.py --nb-file ./models/VIM3/mobilenet_caffe.nb --so-lib ./libs/libnn_mobilenet.so --input-picture data/goldfish_224x224.jpg
```

# Pytorch

## Resnet18

```sh
$ python3 resnet18.py --nb-file ./models/VIM3/resnet18.nb --so-lib ./libs/libnn_resnet18.so --input-picture data/goldfish_224x224.jpg
```

# ONNX

## Resnet50

```sh
$ python3 resnet50.py --nb-file ./models/VIM3/resnet50.nb --so-lib ./libs/libnn_resnet50.so --input-picture ./data/goldfish_224x224.jpg
```

# tflite

## MObilenet

```sh
$ python3 mobilenet.py --nb-file models/VIM3/mobilenet_tflite.nb --so-lib ./libs/libnn_mobilenet.so --input-picture ./data/goldfish_299x299.jpg
```


If your board is VIM3L, Please use VIM3L to replace VIM
