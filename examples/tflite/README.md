# Run

```sh
$ python3 mobilenet.py --model models/VIM3/mobilenet_tflite.nb --library ./libs/libnn_mobilenet.so --picture ./data/goldfish_299x299.jpg --level 0
```

# Convert

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

If you use VIM3L , please use `VIM3L` to replace `VIM3`
