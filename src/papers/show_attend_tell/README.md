This is the soft attention implementation.

**Download data**
```sh
cd src/papers/show_attend_tell/data
sh download.sh
```

**Build vocab**
```sh
uv run src/papers/show_attend_tell/data/build_vocab.py --dataset_path src/papers/show_attend_tell/data --freq_threshold 5 --vocab_output src/papers/show_attend_tell/data/vocab.pkl
```

**Train**
```sh
uv run src/papers/show_attend_tell/train.py --image_dir src/papers/show_attend_tell/data/Images --caption_file src/papers/show_attend_tell/data/captions.txt --vocab_path src/papers/show_attend_tell/data/vocab.pkl --num_epochs 10 --save_dir src/papers/show_attend_tell/checkpoints --finetune_encoder
```

**Inference**
```sh
uv run src/papers/show_attend_tell/inference.py --image_path src/papers/show_attend_tell/data/Images/1024138940_f1fefbdce1.jpg --ckpt_path src/papers/show_attend_tell/checkpoints/best.ckpt --vocab_path src/papers/show_attend_tell/data/vocab.pkl --beam_size 3
# Example output:
# Predicted Caption: two dogs are playing in the sand .
```