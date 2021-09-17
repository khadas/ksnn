```sh
./convert \
--model-name mobilenet_caffe \
--convert-platform caffe \
--caffe-model-file /path/to/mobilenet_v2.caffemodel \
--caffe-proto-file /path/to/mobilenet_v2.prototxt \
--channel-mean-value '103.94 116.78 123.68 58.82' \
--quantized-dtype asymmetric_affine-u8 \
--source-file-path ../demo/data/validation_tf.txt \
--reorder-channel '2 1 0' \
--kboard VIM3
```
