# Run

```sh
$ python3 yolov3-picture.py --model ./models/VIM3/yolov3.nb --library ./libs/libnn_yolov3.so --picture ./data/1080p.bmp
$ python3 yolov3-cap.py --model ./models/VIM3/yolov3.nb --library ./libs/libnn_yolov3.so --device X
$ python3 yolov3-face.py --model ./models/VIM3/yolov3-face.nb --library ./libs/libnn_yolov3-face.so --device X
$ python3 hand-cap.py --model ./models/VIM3/hand.nb --library ./libs/libnn_hand.so --device X
$ python3 hand-tiny-cap.py --model ./models/VIM3/hand-tiny.nb --library ./libs/libnn_hand-tiny.so --device X
```

flask demo

```sh
$ python3 flask-yolov3.py --model ./models/VIM3/yolov3.nb --library ./libs/libnn_yolov3.so --device X
$ python3 flask-face.py --model ./models/VIM3/yolov3-face.nb --library ./libs/libnn_yolov3-face.so --device X
$ python3 flask-hand.py --model ./models/VIM3/hand.nb --library ./libs/libnn_hand.so --device X

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
