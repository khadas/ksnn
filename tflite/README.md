```sh
$ ./convert \
--model-name mobilenet_tflite \
--convert-platform tflite \
--tflite-model-file mobilenet_v1.tflite \
--source-file-path ../demo/data/validation_tf.txt \
--channel-mean-value '127.5 127.5 127.5 127.5' \
--quantized-dtype asymmetric_affine-u8 \
--reorder-channel '0 1 2' \
--kboard VIM3
```

if you use VIM3L , please use `--kboard VIM3L`
