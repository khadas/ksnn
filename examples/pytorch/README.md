# Run

```sh
$ python3 resnet18.py --model ./models/VIM3/resnet18.nb --library ./libs/libnn_resnet18.so --picture data/goldfish_224x224.jpg --level 0
```

# Convert

```sh
./convert \
--model-name resnet18 --platform pytorch \
--model /home/yan/yan/Yan/models-zoo/pytorch/resnet18/resnet18.pt \
--input-size-list '3,224,224' \
--mean-values '103.94 116.78 123.68 0.01700102' \
--quantized-dtype asymmetric_affine \
--source-files ./data/dataset/dataset0.txt \
--kboard VIM3 --print-level 1
```

If you use VIM3L , please use `VIM3L` to replace `VIM3`
