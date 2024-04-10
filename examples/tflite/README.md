# Run

```sh
$ python3 mobilenet.py --model models/VIM3/mobilenet_v1_uint8.nb --library ./libs/libnn_mobilenet_v1_uint8.so --picture ./data/goldfish_299x299.jpg --level 0
```

# Convert

# uint8
```sh
$ ./convert \
--model-name mobilenet \
--platform tflite \
--model /home/yan/yan/Yan/models-zoo/tflite/mobilenet_v1/mobilenet_v1.tflite \
--mean-values '127.5 127.5 127.5 0.007843137' \
--quantized-dtype asymmetric_affine \
--source-files ./data/dataset/dataset0.txt \
--kboard VIM3 --print-level 1
```

# int8
```sh
$ ./convert \
--model-name mobilenet \
--platform tflite \
--model /home/yan/yan/Yan/models-zoo/tflite/mobilenet_v1/mobilenet_v1.tflite \
--mean-values '127.5 127.5 127.5 0.007843137' \
--quantized-dtype dynamic_fixed_point \
--qtype int8 \
--source-files ./data/dataset/dataset0.txt \
--kboard VIM3 --print-level 1
```

# int16
```sh
$ ./convert \
--model-name mobilenet \
--platform tflite \
--model /home/yan/yan/Yan/models-zoo/tflite/mobilenet_v1/mobilenet_v1.tflite \
--mean-values '127.5 127.5 127.5 0.007843137' \
--quantized-dtype dynamic_fixed_point \
--qtype int16 \
--source-files ./data/dataset/dataset0.txt \
--kboard VIM3 --print-level 1
```

If you use VIM3L , please use `VIM3L` to replace `VIM3`
