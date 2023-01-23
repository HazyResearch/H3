# Hungry Hungry Hippos (H3)

This repository provides the official implementation of H3 from the
following paper.

**Hungry Hungry Hippos: Towards Language Modeling with State Space Models**  
Tri Dao\*, Daniel Y. Fu\*,  Khaled K. Saab, Armin W. Thomas, Atri Rudra, Christopher Ré  
International Conference on Learning Representations, 2023. Notable top-25% (spotlight).
Paper: https://arxiv.org/abs/2212.14052

![H3](assets/banner.png)

# Code & model release

You can find model weights on the HuggingFace Hub here (under "Files and Versions" for each model):
* [125M](https://huggingface.co/danfu09/H3-125M)
* [355M](https://huggingface.co/danfu09/H3-355M)
* [1.3B](https://huggingface.co/danfu09/H3-1.3B)
* [2.7B](https://huggingface.co/danfu09/H3-2.7B)

An example of how to load the weights is given in `benchmarks/benchmark_generation.py`.
More examples coming soon!

## Acknowledgments
Some of the files related to S4D and HiPPO initialization are
adapted from the https://github.com/HazyResearch/state-spaces.

## Citation
If you use this codebase, or otherwise found our work valuable, please cite:
```
@inproceedings{dao2023hungry,
  title={Hungry {H}ungry {Hippos}: Towards Language Modeling with State Space Models},
  author={Dao, Tri and Fu, Daniel Y. and Saab, Khaled K. and Thomas, Armin W.
  and Rudra, Atri and R{\'e}, Christopher},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```
