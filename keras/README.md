# Run

```sh
$ python3 xception.py --model ./models/VIM3/xception.nb --library ./libs/libnn_xception.so --picture data/goldfish_299x299.jpg
```

# Convert

```sh
$ ./convert \
--model-name xception \
--platform keras \
--model xception.h5 \
--mean-values '127.5,127.5,127.5,127.5' \
--quantized-dtype asymmetric_affine \
--kboard VIM3 --print-level 1
```

If you use VIM3L , please use `VIM3L` to replace `VIM3`
