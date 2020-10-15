# Improving Language Understanding by Generative Pre-Training (GPT)

paper: https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf

code: 

## Abstract

* NLU tasks
  * textual entailment/ question answering/ semantic similarity assessmen/document classification
* Although large unlabeled text corpora are abundant, labeled data for learning these specific tasks is scarce, making it challenging for discriminatively trained models to perform adequately.
* large gains on these tasks can be realized 
  * by **generative pre-training of a language model** on a <u>diverse corpus of unlabeled text</u>, 
  * followed by **discriminative fine-tuning** on each **specific task**. 
* make **use of task-aware input transformations** during fine-tuning to achieve effective transfer 
  * while requiring minimal changes to the model architecture. 
*  **general task-agnostic model** outperforms ~~discriminatively trained models(use architectures specifically crafted for each task)~~ 
  * significantly improving upon the state of the art in 9 out of the 12 tasks studied.
  * achieve absolute improvements of 8.9% on commonsense reasoning (Stories Cloze Test), 5.7% on question answering (RACE), and 1.5% on textual entailment (MultiNLI)

## 1 Introduction

## 2 Related Work

### 1) Semi-supervised learning for NLP 

### 2) Unsupervised pre-training

### 3) Auxiliary training objectives 



## 3 Framework

* Our training procedure consists of two stages

  1) learning a high-capacity language model on a large corpus of text.

  2) followed by a fine-tuning stage, 

  - adapt the model to a discriminative task with labeled data.

### 3.1 Unsupervised pre-training

* Given an unsupervised corpus of tokens $U = {u_1, . . . , u_n}$,
  * use a standard **language modeling objective** to maximize the following likelihood
    * $L_1(U) = X i log P(u_i |u_{i−k}, . . . , u_{i−1}; Θ)$
      * $k$ :the size of the context window,
      * $P$: the conditional probability
        * is modeled using a neural network with parameters Θ. 
        * These parameters are trained using stochastic gradient descent [51].
  * use a **multi-layer Transformer decoder** [34] for the language model,
    * is a variant of the transformer [62]. 
    * This model applies a multi-headed self-attention operation over the input context tokens followed by position-wise feedforward layers to produce an output distribution over target tokens:
    * $h_0 = UW_e + W_p$
    * $h_l = transformer\_block(h_{l−1})∀i ∈ [1, n] $
    * $P(u) = softmax(h_nW^T_e )$
      * U = (u−k, . . . , u−1):context vector of tokens
      * n: the number of layers
      * $W_e$: token embedding matrix
      * $Wp$: position embedding matrix

### 3.2 Supervised fine-tuning

### 3.3 Task-specific input transformations

