#!/bin/bash
curl -L -o flickr8k.zip \
  https://www.kaggle.com/api/v1/datasets/download/adityajn105/flickr8k
unzip flickr8k.zip
rm flickr8k.zip