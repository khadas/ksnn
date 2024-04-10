# Run

```sh
$ python3 mobilenet.py --model ./models/VIM3/mobilenet_v2_uint8.nb --library ./libs/libnn_mobilenet_uint8.so --picture data/goldfish_224x224.jpg --level 0
```

# Convert

# uint8
```sh
./convert \
--model-name mobilenet_caffe \
--platform caffe \
--model /home/yan/yan/Yan/models-zoo/caffe/mobilenet_v2/mobilenet_v2.prototxt \
--weights /home/yan/yan/Yan/models-zoo/caffe/mobilenet_v2/mobilenet_v2.caffemodel \
--mean-values '103.94 116.78 123.68 0.01700102' \
--quantized-dtype asymmetric_affine \
--source-files ./data/dataset/dataset0.txt \
--kboard VIM3 --print-level 1
```

# int8
```sh
./convert \
--model-name mobilenet_caffe \
--platform caffe \
--model /home/yan/yan/Yan/models-zoo/caffe/mobilenet_v2/mobilenet_v2.prototxt \
--weights /home/yan/yan/Yan/models-zoo/caffe/mobilenet_v2/mobilenet_v2.caffemodel \
--mean-values '103.94 116.78 123.68 0.01700102' \
--quantized-dtype dynamic_fixed_point \
--qtype int8 \
--source-files ./data/dataset/dataset0.txt \
--kboard VIM3 --print-level 1
```

# int16
```sh
./convert \
--model-name mobilenet_caffe \
--platform caffe \
--model /home/yan/yan/Yan/models-zoo/caffe/mobilenet_v2/mobilenet_v2.prototxt \
--weights /home/yan/yan/Yan/models-zoo/caffe/mobilenet_v2/mobilenet_v2.caffemodel \
--mean-values '103.94 116.78 123.68 0.01700102' \
--quantized-dtype dynamic_fixed_point \
--qtype int16 \
--source-files ./data/dataset/dataset0.txt \
--kboard VIM3 --print-level 1
```

If you use VIM3L , please use `VIM3L` to replace `VIM3`
