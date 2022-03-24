# run

```sh
$ python3 inceptionv3.py --model ./models/VIM3/inceptionv3.nb --library ./libs/libnn_inceptionv3.so --picture ./data/goldfish_299x299.jpg --level 0
$ python3 mobilenet_ssd_picture.py --model ./models/VIM3/mobilenet_ssd.nb --library ./libs/libnn_mobilenet_ssd.so --picture data/1080p.bmp --level 0
```

# Convert

```sh
$ ./convert \
--model-name inceptionv3 \
--platform tensorflow \
--model /home/yan/yan/Yan/models-zoo/tensorflow/inception/inception_v3_2016_08_28_frozen.pb \
--input-size-list '299,299,3' \
--inputs input \
--outputs InceptionV3/Predictions/Reshape_1 \
--mean-values '128 128 128 0.0078125' \
--quantized-dtype asymmetric_affine \
--source-files ./data/dataset/dataset0.txt \
--kboard VIM3 --print-level 1
```

```sh
$ ./convert \
--model-name mobilenet_ssd \
--platform tensorflow \
--model ~/yan/Yan/models-zoo/tensorflow/mobilenet_ssd/ssd_mobilenet_v1_coco_2017_11_17.pb \
--input-size-list '300,300,3' \
--inputs FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/batchnorm/mul_1 \
--outputs "'concat concat_1'" \
--mean-values '127.5 127.5 127.5 0.007843137' \
--quantized-dtype asymmetric_affine \
--source-files ./data/dataset/dataset0.txt \
--kboard VIM3 --print-level 1
```

If you use VIM3L , please use `VIM3L` to replace `VIM3`
