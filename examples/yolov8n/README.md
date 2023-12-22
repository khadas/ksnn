# Run

```sh
$ python3 yolov8n-picture.py --model ./models/VIM3/yolov8n.nb --library ./libs/libnn_yolov8n.so --picture ./data/horses.jpg
$ python3 yolov8n-cap.py --model ./models/VIM3/yolov8n.nb --library ./libs/libnn_yolov8n.so --device X
```

# Convert

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


If you use VIM3L , please use `VIM3L` to replace `VIM3`
