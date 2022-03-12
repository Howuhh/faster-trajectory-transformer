# Faster Trajectory Transformer

![video](media/halfcheetah.gif)
![video](media/walker2d.gif)
![video](media/hopper.gif)

# Training

For runs configuration see parameters in `configs/medium`.

```bash
python scripts/train.py --config="configs/medium/halfcheetah_medium" --device="cuda" --seed="42"
```

# Evaluation

All available evaluation parameters can be seen in `configs/eval_vase.yaml`.
Here parameters are set to match configs from original implementation by [@jannerm](https://github.com/jannerm). 

```bash
# you can override every config value from command line

# halfcheetah-medium-v2
python scripts/eval.py \ 
    --config="configs/eval_base.yaml" --device="cuda" --seed="42" \
    checkpoints_path="pretrained/halfcheetah" \
    beam_context=5 \ 
    beam_steps=5 \
    beam_width=32

# hopper-medium-v2
python scripts/eval.py \ 
    --config="configs/eval_base.yaml" --device="cuda" --seed="42" \
    checkpoints_path="pretrained/hopper" \
    beam_context=5 \ 
    beam_steps=15 \
    beam_width=128

# walker2d-medium-v2
python scripts/eval.py \ 
    --config="configs/eval_base.yaml" --device="cuda" --seed="42" \
    checkpoints_path="pretrained/walker2d" \
    beam_context=5 \ 
    beam_steps=15 \
    beam_width=128
```

