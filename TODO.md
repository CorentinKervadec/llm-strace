# TODO List

## Adding new models
- [ ] Llama 3
- [ ] Llama 2
- [ ] Llama 1
- [ ] Gemma 3
- [ ] Gemma 2
- [ ] Gemma 1
- [ ] OLMo 1
- [ ] Stable LM 2
- [ ] Stable LM 1
- [ ] OPT

## Fixing bugs
- [ ] Qwen 2 and 2.5: unstable, often raise reconstruction errors
- [ ] OLMO 2: 7B failed (nothing ran)
- [ ] Qwen 3: 1.7B and 8B failed. Looks like the model's output is wrong (0.6B and 4B worked fine)
- [ ] Qwen 2: 1.5B and 7B failed. Loss and/or entropy is not finite.
- [ ] Qwen 2.5: 1.5B and 7B failed. (same as above)

## Analyse Scripts
- [ ] Plot all models on the same plot

## Scaling up
- [ ] Implement a one-layer-at-a-time forward pass for extraction and evaluation in order to support for larger model size.
- [ ] Implement a strategy to allows larger context size 

## Others
- [ ] The reconstruct tolerance is too high (in sanity check)