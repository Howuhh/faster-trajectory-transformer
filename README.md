# Faster Trajectory Transformer

![video](media/halfcheetah.gif)
![video](media/walker2d.gif)
![video](media/hopper.gif)

**WARN**: Pay attention to the [issue](https://github.com/Howuhh/faster-trajectory-transformer/issues/3#issue-1286177514) with the bug, which is not fixed yet. It will be fixed eventually tho.

This is reimplementation of Trajectory Transformer, introduced in **Offline Reinforcement Learning 
as One Big Sequence Modeling Problem** [paper](https://arxiv.org/abs/2106.02039). 

The original implementation has few problems with inference speed, namely quadratic attention during
inference and sequential rollouts. The former slows down planning a lot, while the latter does not 
allow to do rollouts in parallel and utilize GPU to the full.

Still, even after all changes, it is not that fast compared to traditional methods such as PPO or SAC/DDPG. 
However, the gains are huge, what used to take hours now takes a dozen minutes (25 rollouts, 1k steps each, for example).
Training time remains the same, though.

## Changes

### Attention caching

During beam search we're only predicting one token at a time. So with the naive implementation model will make a lot of 
unnecessary computations to recompute attention maps for full past context. However it is not necessary, 
as it was already computed when the previous token was predicted. **All we need is to cache it**!

Actually, attention caching is a common thing in NLP field, but a lot of RL practitioners may
not be familiar with NLP, so the code also can be educational.

### Vectorized rollouts

Vectorized environments allow batching beam search planning and select actions in parallel, which is a lot faster 
if you need to evaluate agent on number of episodes (or seeds) during training.  


## Training

I trained it on D4RL medium datasets to validate that everything is OK. Scores seem to be very close to the original.
Pretrained models are [available](pretrained).

All training parameters can be seen in training [configs](configs/medium). 
Also, all datasets for [D4RL](https://sites.google.com/view/d4rl/home) Gym tasks are supported.

```bash
python scripts/train.py --config="configs/medium/halfcheetah_medium" --device="cuda" --seed="42"
```

## Evaluation

Available evaluation parameters can be seen in validation [config](configs/eval_base.yaml).
Here parameters are set to match evaluation configs from original implementation by [@jannerm](https://github.com/jannerm/trajectory-transformer). 

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

## References
```
@inproceedings{janner2021sequence,
  title = {Offline Reinforcement Learning as One Big Sequence Modeling Problem},
  author = {Michael Janner and Qiyang Li and Sergey Levine},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2021},
}
```
