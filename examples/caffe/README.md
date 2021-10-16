# Run

```sh
$ python3 mobilenet.py --model ./models/VIM3/mobilenet_caffe.nb --library ./libs/libnn_mobilenet.so --picture data/goldfish_224x224.jpg --level 0
$ python3 openpose-multi-cap.py --model ./models/VIM3/openpose.nb --lbrary libs/libnn_openpose.so --device X --level 0
$ python3 openpose-multi-picture.py --model ./models/VIM3/openpose.nb --lbrary libs/libnn_openpose.so --picture data/person.jpg --level 0
$ python3 openpose-signle-cap.py --model ./models/VIM3/openpose.nb --lbrary libs/libnn_openpose.so --device X --level 0
$ python3 openpose-signle-picture.py --model ./models/VIM3/openpose.nb --lbrary libs/libnn_openpose.so --picture data/person.jpg --level 0
```

# Convert

```sh
./convert \
--model-name mobilenet_caffe \
--platform caffe \
--model mobilenet_v2.prototxt \
--weights mobilenet_v2.caffemodel \
--mean-values '103.94,116.78,123.68,58.82' \
--quantized-dtype asymmetric_affine \
--kboard VIM3 --print-level 1
```

```sh
$ ./convert \
--model-name openpose \
--platform caffe \
--model pose_deploy_linevec.prototxt \
--weights pose_iter_440000.caffemodel \
--mean-values '0,0,0,256' \
--quantized-dtype asymmetric_affine \
--kboard VIM3 --print-level 1
```

If you use VIM3L , please use `VIM3L` to replace `VIM3`
