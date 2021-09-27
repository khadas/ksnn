# Run

```sh
$ python3 mobilenet.py --model ./models/VIM3/mobilenet_caffe.nb --library ./libs/libnn_mobilenet.so --picture data/goldfish_224x224.jpg --level 0
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

If you use VIM3L , please use `VIM3L` to replace `VIM3`
