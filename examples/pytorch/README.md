# Run

```sh
$ python3 resnet18.py --model ./models/VIM3/resnet18.nb --library ./libs/libnn_resnet18.so --input-picture data/goldfish_224x224.jpg --level 0
```

# Convert

```sh
./convert \
--model-name resnet18 \
--platform pytorch \
--model resnet18.pt \
--input-size-list '3,224,224' \
--mean-values '103.94,116.78,123.68,58.82' \
--quantized-dtype asymmetric_affine \
--kboard VIM3 --print-level 1
```

If you use VIM3L , please use `VIM3L` to replace `VIM3`
