# yolov3 convert parameters

```sh
$ ./convert \
--model-name yolov3 \
--convert-platform darknet \
--darknet-net-input yolov3.cfg \
--darknet-weight-input yolov3.weights \
--source-file-path ../demo/data/validation_tf.txt \
--channel-mean-value '0 0 0 256' \
--quantized-dtype asymmetric_affine-u8 \
--reorder-channel '2 1 0' \
--kboard VIM3
```
if you use VIM3L , please use `--kboard VIM3L`
