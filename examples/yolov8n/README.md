# Run

```sh
$ python3 yolov8n-picture.py --model ./models/VIM3/yolov8n_uint8.nb --library ./libs/libnn_yolov8n_uint8.so --picture ./data/horses.jpg
$ python3 yolov8n-cap.py --model ./models/VIM3/yolov8n_uint8.nb --library ./libs/libnn_yolov8n_uint8.so --device X
```

# Convert

# uint8
```sh
$ ./convert \
--model-name yolov8n \
--platform onnx \
--model ./yolov8n.onnx \
--mean-values '0 0 0 0.00392156' \
--quantized-dtype asymmetric_affine \
--source-files ./data/dataset/dataset0.txt \
--kboard VIM3 --print-level 1
```

# int8
```sh
$ ./convert \
--model-name yolov8n \
--platform onnx \
--model ./yolov8n.onnx \
--mean-values '0 0 0 0.00392156' \
--quantized-dtype dynamic_fixed_point \
--qtype int8 \
--source-files ./data/dataset/dataset0.txt \
--kboard VIM3 --print-level 1
```

# int16
```sh
$ ./convert \
--model-name yolov8n \
--platform onnx \
--model ./yolov8n.onnx \
--mean-values '0 0 0 0.00392156' \
--quantized-dtype dynamic_fixed_point \
--qtype int16 \
--source-files ./data/dataset/dataset0.txt \
--kboard VIM3 --print-level 1
```

If you use VIM3L , please use `VIM3L` to replace `VIM3`
