# Loading and Running H3 Models

You can use the `generate_text_h3.py` script to generate outputs from our H3 models.
Examples of how to download and generate text with each model are given below, along with sample output.

For any of these models, you can play around with these parameters to get different output, or longer output:
* `--genlen`: Maximum number of tokens to generate after the prompt. By default, 128.
* `--top_p`, `--top_k`: Generation hyperparameters for the sampling procedure. By default, 0.9 and 50.11

## 125M
Commands to download and run the 125M model. You will need ~3GB GPU memory:
```
git clone https://huggingface.co/danfu09/H3-125M

PYTHONPATH=$(pwd)/H3 python H3/examples/generate_text_h3.py --ckpt H3-125M/model.pt --prompt "Hungry Hungry Hippos: Towards Language Modeling With State Space Models is a new language model that" --dmodel 768 --nlayer 12 --attn-layer-idx 6 --nheads=12
```
Output:
> Hungry Hungry Hippos: Towards Language Modeling With State Space Models is a new language model that uses state-space models to create a human-like vocabulary that can help improve human understanding and judgment of language. It takes a human's past experience of language, and tries to capture their cognitive patterns. State Space Models helps the researchers make sense of language in its own terms, which helps users learn about their language of choice. State Space Models is used to develop a set of languages for researchers in an effort to help them develop more intelligent language models. The goal is to increase and develop a human-like language model using state space models. It is hoped that it will aid people to do more work to develop a language that is more

## 355M
Commands to download and run the 355M model. You will need ~6GB GPU memory:
```
git clone https://huggingface.co/danfu09/H3-355M

PYTHONPATH=$(pwd)/H3 python H3/examples/generate_text_h3.py --ckpt H3-355M/model.pt --prompt "Hungry Hungry Hippos: Towards Language Modeling With State Space Models is a new language model that" --dmodel 1024 --nlayer 24 --attn-layer-idx 8 16 --nheads 16
```
Output:
> Hungry Hungry Hippos: Towards Language Modeling With State Space Models is a new language model that addresses the need for a model that generalizes to arbitrary environments, such that models that have not been tested can easily be retrained. A recent work that extended our model to a novel scenario was proposed in 2014, “Learning-To-Language”: A Neural Language Model with Tempering. The model can learn to understand arbitrary domains, from a single language-to-language interface through to a language that can be understood by multiple languages. This model has been developed within a framework of natural language processing but could be applicable to other domains such as machine translation or dialogue systems. I have tested our system in a variety of domains

## 1.3B
Commands to download and run the 1.3B model. You will need ~11GB GPU memory:
```
git clone https://huggingface.co/danfu09/H3-1.3B

PYTHONPATH=$(pwd)/H3 python H3/examples/generate_text_h3.py --ckpt H3-1.3B/model.pt --prompt "Hungry Hungry Hippos: Towards Language Modeling With State Space Models is a new language model that" --dmodel 2048 --nlayer 24 --attn-layer-idx 8 16 --nheads 16
```
Output:
> Hungry Hungry Hippos: Towards Language Modeling With State Space Models is a new language model that predicts the next word in the queue or history from a hidden state in a multistage LSTM.
>
> Mimic Language Model is a new language model that predicts the text next sentence in the queue or history.<|endoftext|>

## 2.7B
We have two 2.7B models: one with three attention layers, and one with two attention layers using Rotary embeddings.
Both of these models get the same val PPL on the PILE (5.4 PPL, vs. GPT-Neo 5.7 PPL).
We will also release a 2.7B with 2 vanilla attention layers soon!

These models both need ~21GB GPU memory.

You can download both models with this command:
```
git clone https://huggingface.co/danfu09/H3-2.7B
```

You can run the 2-attention model with this command:
```
PYTHONPATH=$(pwd)/H3 python H3/examples/generate_text_h3.py --ckpt H3-2.7B/model-2attn-rotary.pt --prompt "Hungry Hungry Hippos: Towards Language Modeling With State Space Models is a new language model that" --dmodel 2560 --nlayer 32 --attn-layer-idx 10 21 --nheads 20 --rotary_emb_dim 64
```
Output:
> Hungry Hungry Hippos: Towards Language Modeling With State Space Models is a new language model that provides a state-space representation of the input words and makes predictions of next sentences given previous sentences. It was released on September 18, 2013 at ICML 2013 conference.
> 
> We evaluated the language model based on five measures, all of which use the state space representation: perplexity (PEN), Mean Reciprocal Rank (MRR), normalized discounted cumulative gain (NCG) and negative log-likelihood. The state space representation was implemented with a one-shot technique. We used the language model results in the same way as in the original paper.
> 
> In this paper we compared four language model methods, one-

Wow, turns out we released H3 10 years ago... imagine that!

You can run the 3-attention model with this command:
```
PYTHONPATH=$(pwd)/H3 python H3/examples/generate_text_h3.py --ckpt H3-2.7B/model-3attn.pt --prompt "Hungry Hungry Hippos: Towards Language Modeling With State Space Models is a new language model that" --dmodel 2560 --nlayer 32 --attn-layer-idx 8 16 24 --nheads 20
```
Output:
> Hungry Hungry Hippos: Towards Language Modeling With State Space Models is a new language model that employs state-space models. Such models have gained popularity in machine learning because learning them can be understood in terms of Bayesian inference. They are more general than standard neural network models, making them easier to deploy and implement (see references and papers at the end).
> 
> Categories:
> Machine Learning<|endoftext|>
