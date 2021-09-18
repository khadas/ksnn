
```sh
$ ./convert \
--model-name inception_v3 \
--convert-platform tensorflow \
--tf-inputs input --tf-input-size-list '299,299,3' \
--tf-outputs InceptionV3/Predictions/Reshape_1 \
--tf-model-file inception_v3_2016_08_28_frozen.pb \
--source-file-path ../demo/data/validation_tf.txt \
--channel-mean-value '128 128 128 128' \
--quantized-dtype asymmetric_affine-u8 \
--reorder-channel '0 1 2' \
--kboard VIM3
```

```sh
$ ./convert \
--model-name mobilenet_ssd \
--convert-platform tensorflow \
--tf-inputs FeatureExtractor/MobilenetV1/MobilenetV1/Conv2d_0/BatchNorm/batchnorm/mul_1 \
--tf-input-size-list '300,300,3' \
--tf-outputs "'concat concat_1'" \
--tf-model-file ssd_mobilenet_v1.pb \
--source-file-path ../demo/data/validation_tf.txt \
--channel-mean-value '127.5 127.5 127.5 127.5' \
--quantized-dtype asymmetric_affine-u8 \
--reorder-channel '0 1 2' \
--kboard VIM3
```

if you use VIM3L , please use `--kboard VIM3L`
