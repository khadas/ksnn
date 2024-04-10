# run

```sh
$ python3 mobilenet_ssd_picture.py --model ./models/VIM3/mobilenet_ssd_uint8.nb --library ./libs/libnn_mobilenet_ssd_uint8.so --picture data/1080p.bmp --level 0
```

# Convert

# uint8
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

# int8
```sh
$ ./convert \
--model-name mobilenet_ssd \
--platform tensorflow \
--model ~/yan/Yan/models-zoo/tensorflow/mobilenet_ssd/ssd_mobilenet_v1_coco_2017_11_17.pb \
--input-size-list '300,300,3' \
--inputs FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/batchnorm/mul_1 \
--outputs "'concat concat_1'" \
--mean-values '127.5 127.5 127.5 0.007843137' \
--quantized-dtype dynamic_fixed_point \
--qtype int8 \
--source-files ./data/dataset/dataset0.txt \
--kboard VIM3 --print-level 1
```

# int16
```sh
$ ./convert \
--model-name mobilenet_ssd \
--platform tensorflow \
--model ~/yan/Yan/models-zoo/tensorflow/mobilenet_ssd/ssd_mobilenet_v1_coco_2017_11_17.pb \
--input-size-list '300,300,3' \
--inputs FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/batchnorm/mul_1 \
--outputs "'concat concat_1'" \
--mean-values '127.5 127.5 127.5 0.007843137' \
--quantized-dtype dynamic_fixed_point \
--qtype int16 \
--source-files ./data/dataset/dataset0.txt \
--kboard VIM3 --print-level 1
```

If you use VIM3L , please use `VIM3L` to replace `VIM3`
