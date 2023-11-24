# Run

```sh
$ python3 yolov7_tiny-picture.py --model ./models/VIM3/yolov7_tiny.nb --library ./libs/libnn_yolov7_tiny.so --picture ./data/horses.jpg
$ python3 yolov7_tiny-cap.py --model ./models/VIM3/yolov7_tiny.nb --library ./libs/libnn_yolov7_tiny.so --device X
```

# Convert

```sh
$ ./convert \
--model-name yolov7_tiny \
--platform onnx \
--model ./yolov7_tiny.onnx \
--mean-values '0 0 0 0.00392156' \
--quantized-dtype asymmetric_affine \
--source-files ./data/dataset/dataset0.txt \
--kboard VIM3 --print-level 1
```

If you use VIM3L , please use `VIM3L` to replace `VIM3`
