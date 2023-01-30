# Hungry Hungry Hippos (H3)

This repository provides the official implementation of H3 from the
following paper.

**Hungry Hungry Hippos: Towards Language Modeling with State Space Models**  
Tri Dao\*, Daniel Y. Fu\*,  Khaled K. Saab, Armin W. Thomas, Atri Rudra, Christopher Ré  
International Conference on Learning Representations, 2023. Notable top-25% (spotlight).  
Paper: https://arxiv.org/abs/2212.14052

![H3](assets/banner.png)

# Code & model release

You can find model weights on the Hugging Face Hub here (under "Files and Versions" for each model):
* [125M](https://huggingface.co/danfu09/H3-125M)
* [355M](https://huggingface.co/danfu09/H3-355M)
* [1.3B](https://huggingface.co/danfu09/H3-1.3B)
* [2.7B](https://huggingface.co/danfu09/H3-2.7B)

## Loading weights and running inference

Examples of how to load the weights and run inference are given in `benchmarks/benchmark_generation.py` and `examples/generate_text_h3.py`.

Here's an example of how to download and run our 125M model (you may need to `pip install flash-attn`):

```
git lfs install
git clone https://huggingface.co/danfu09/H3-125M

git clone https://github.com/HazyResearch/H3.git

PYTHONPATH=$(pwd)/H3 python H3/examples/generate_text_h3.py --ckpt H3-125M/model.pt --prompt "Hungry Hungry Hippos: Towards Language Modeling With State Space Models is a new language model that" --dmodel 768 --nlayer 12 --attn-layer-idx 6 --nheads=12
```

You should get an output like this (may change due to sampling in the text generation):
```
Number of parameters: 126387456
Hungry Hungry Hippos: Towards Language Modeling With State Space Models is a new language model that uses state-space models to create a human-like vocabulary that can help improve human understanding and judgment of language. It takes a human's past experience of language, and tries to capture their cognitive patterns. State Space Models helps the researchers make sense of language in its own terms, which helps users learn about their language of choice. State Space Models is used to develop a set of languages for researchers in an effort to help them develop more intelligent language models. The goal is to increase and develop a human-like language model using state space models. It is hoped that it will aid people to do more work to develop a language that is more
```

Here's the summary of model sizes for each model:
| **Model** | **dmodel** | **nlayer** | **nheads** |
| :-------- | :--------: | :--------: | :--------: |
| **125M**  |    768     |     12     |     12     |
| **355M**  |   1024     |     24     |     16     |
| **1.3B**  |   2048     |     24     |     16     |
| **2.7B**  |   2560     |     32     |     20     |

See `examples/README.md` for examples about how to load all these models and run them!

## Acknowledgments
Some of the files related to S4D and HiPPO initialization are
adapted from the https://github.com/HazyResearch/state-spaces.

## Citation
If you use this codebase, or otherwise found our work valuable, please cite:
```
@inproceedings{dao2023hungry,
  title={Hungry {H}ungry {H}ippos: Towards Language Modeling with State Space Models},
  author={Dao, Tri and Fu, Daniel Y. and Saab, Khaled K. and Thomas, Armin W.
  and Rudra, Atri and R{\'e}, Christopher},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```
