# Run

```sh
$ python3 resnet50.py --model ./models/VIM3/resnet50.nb --library ./libs/libnn_resnet50.so --picture ./data/goldfish_224x224.jpg --level 0
```

# convert

```sh
$ ./convert \
--model-name resnet50 \
--platform onnx \
--model /home/yan/yan/Yan/models-zoo/onnx/resnet50/resnet50v2.onnx \
--mean-values '123.675 116.28 103.53 0.01700102' \
--quantized-dtype asymmetric_affine \
--source-files ./data/dataset/dataset0.txt \
--kboard VIM3 --print-level 1
```

If you use VIM3L , please use `VIM3L` to replace `VIM3`
