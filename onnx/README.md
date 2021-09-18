```sh
$ ./convert \
--model-name resnet50 \
--convert-platform onnx \
--onnx-model-file resnet50v2.onnx \
--source-file-path ../demo/data/validation_tf.txt \
--channel-mean-value '123.675 116.28 103.53 58.82' \
--quantized-dtype asymmetric_affine-u8 \
--reorder-channel '0 1 2' \
--kboard VIM3
```

if you use VIM3L , please use `--kboard VIM3L`
