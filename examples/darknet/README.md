# Run

```sh
$ python3 yolov3-picture.py --model ./models/VIM3/yolov3_uint8.nb --library ./libs/libnn_yolov3_uint8.so --picture ./data/1080p.bmp
$ python3 yolov3-cap.py --model ./models/VIM3/yolov3_uint8.nb --library ./libs/libnn_yolov3_uint8.so --device X
```

# Convert

# uint8
```sh
$ ./convert \
--model-name yolov3 \
--platform darknet \
--model /home/yan/yan/git/neural_network/darknet/darknet/cfg/yolov3.cfg \
--weights /home/yan/yan/Yan/models-zoo-big/yolov3.weights \
--mean-values '0 0 0 0.00390625' \
--quantized-dtype asymmetric_affine \
--source-files ./data/dataset/dataset0.txt \
--kboard VIM3 --print-level 1
```

# int8
```sh
$ ./convert \
--model-name yolov3 \
--platform darknet \
--model /home/yan/yan/git/neural_network/darknet/darknet/cfg/yolov3.cfg \
--weights /home/yan/yan/Yan/models-zoo-big/yolov3.weights \
--mean-values '0 0 0 0.00390625' \
--quantized-dtype dynamic_fixed_point \
--qtype int8 \
--source-files ./data/dataset/dataset0.txt \
--kboard VIM3 --print-level 1
```

# int16
```sh
$ ./convert \
--model-name yolov3 \
--platform darknet \
--model /home/yan/yan/git/neural_network/darknet/darknet/cfg/yolov3.cfg \
--weights /home/yan/yan/Yan/models-zoo-big/yolov3.weights \
--mean-values '0 0 0 0.00390625' \
--quantized-dtype dynamic_fixed_point \
--qtype int16 \
--source-files ./data/dataset/dataset0.txt \
--kboard VIM3 --print-level 1
```

If you use VIM3L , please use `VIM3L` to replace `VIM3`
