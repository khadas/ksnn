# Run

```sh
$ python3 yolov3-picture.py --model ./models/VIM3/yolov3.nb --library ./libs/libnn_yolov3.so --picture ./data/1080p.bmp
$ python3 yolov3-cap.py --model ./models/VIM3/yolov3.nb --library ./libs/libnn_yolov3.so --video-device X
```

# Convert

```sh
$ ./convert \
--model-name yolov3 \
--platform darknet \
--model yolov3.cfg \
--weights yolov3.weights \
--mean-values '0,0,0,256' \
--quantized-dtype asymmetric_affine \
--kboard VIM3 --print-level 1
```

If you use VIM3L , please use `VIM3L` to replace `VIM3`
