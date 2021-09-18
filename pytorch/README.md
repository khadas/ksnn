```sh
./convert \
--model-name mobilenet_caffe \
--convert-platform caffe \
--caffe-model-file mobilenet_v2.caffemodel \
--caffe-proto-file mobilenet_v2.prototxt \
--channel-mean-value '103.94 116.78 123.68 58.82' \
--quantized-dtype asymmetric_affine-u8 \
--source-file-path ../demo/data/validation_tf.txt \
--reorder-channel '2 1 0' \
--kboard VIM3
```

if you use VIM3L , please use `--kboard VIM3L`
