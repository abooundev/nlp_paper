# Improving Language Understanding by Generative Pre-Training (GPT)

* paper: https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf

* code
  - tf: https://github.com/openai/finetune-transformer-lm
  - pytorch: https://github.com/huggingface/pytorch-openai-transformer-lm



![img](https://1.bp.blogspot.com/-RLAbr6kPNUo/W9is5FwUXmI/AAAAAAAADeU/5y9466Zoyoc96vqLjbruLK8i_t8qEdHnQCLcBGAs/s1600/image3.png)

## Abstract

* NLU tasks

  * textual entailment/ question answering/ semantic similarity assessmen/document classification

* Although large unlabeled text corpora are abundant, labeled data for learning these specific tasks is scarce, making it challenging for discriminatively trained models to perform adequately.

* large gains on these tasks can be realized 

  * by **generative pre-training of a language model** on a <u>diverse corpus of unlabeled text</u>, 

  * followed by **discriminative fine-tuning** on each **specific task**. 

    |                    | pre-training                         | fine-tuning                                        |
    | ------------------ | ------------------------------------ | -------------------------------------------------- |
    | **characteristic** | generative                           | discriminative                                     |
    | **problem**        | a language model                     | each specific task                                 |
    | **data**           | diverse corpus of **unlabeled text** | **labeled data** for learning these specific tasks |
    | **learning type**  | unsupervised                         | supervised                                         |

  

* make <span style="color:blue">**use of task-aware input transformations** </span>during fine-tuning to achieve effective transfer 

  * while requiring minimal changes to the model architecture. 

* <span style="color:red"> **general task-agnostic model**</span> outperforms ~~discriminatively trained models(use architectures specifically crafted for each task)~~ 

  * significantly improving upon the state of the art in 9 out of the 12 tasks studied.
  * achieve absolute improvements of 8.9% on commonsense reasoning (Stories Cloze Test), 5.7% on question answering (RACE), and 1.5% on textual entailment (MultiNLI)

## 1 Introduction

## 2 Related Work

### 1) Semi-supervised learning for NLP 

### 2) Unsupervised pre-training

### 3) Auxiliary training objectives 



## 3 Framework

* Our training procedure consists of two stages

  1) learning a high-capacity **language model** on a large corpus of text.

  2) followed by a **fine-tuning** stage, 

  - adapt the model to a discriminative task with labeled data.

* corpus 

  |               | dataset                              | notation                                                     |
  | ------------- | ------------------------------------ | ------------------------------------------------------------ |
  | $\mathcal{U}$ | an **unsupervised corpus** of tokens | $\mathcal{U} = \{u_1, . . . , u_n\}$                         |
  | $\mathcal{C}$ | a **labeled dataset**                | each instance consists of **a sequence of input tokens, $x^1, . . . , x^m$**, along with **a label $y$** |

  ![image-20201016191556484](/Users/csg/Library/Application Support/typora-user-images/image-20201016191556484.png)

![image-20201016192941884](/Users/csg/Library/Application Support/typora-user-images/image-20201016192941884.png)

|                     | Unsupervised pre-traning                                     | Supervised fine-tuning                                       |
| ------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| corpus              | an **unsupervised corpus** of tokens  $\mathcal{U} = \{u_1, . . . , u_n\}$ | a **labeled dataset **$C$ (each instance consists of a sequence of input tokens, $x^1, . . . , x^m$, along with a label $y$) |
| objective  function | $L_1(\mathcal{U}) = \sum_{i} log P(u_iu_{i−k}, . . . , u_{i−1};Θ)$ | $L_2(\mathcal{C}) = \sum_{(x,y)} log P(y|x^1,...,x^m)$ $L_3(\mathcal{C}) = L_2(\mathcal{C}) + λ ∗ L_1(\mathcal{C})$ |
| model               | $h_0 = UW_e + W_p$$h_l = transformer\_block(h_{l−1})∀i ∈ [1, n]$ $P(u) = softmax(h_nW^T_e )$ | $P(y|x^1, . . . , x^m)=softmax(h^m_lWy)$                     |
| input               | contiguous sequences of text                                 | convert structured inputs into an ordered sequence           |



### 3.1 Unsupervised pre-training

* Given an **unsupervised corpus** of tokens  $\mathcal{U} = \{u_1, . . . , u_n\}$,

  * use a standard **language modeling objective** to **maximize the following likelihood**:
    $$
    L_1(\mathcal{U}) = \sum_{i} log P(u_i |u_{i−k}, . . . , u_{i−1};Θ)
    $$

    * $k$ : size of the context window,
    * $P$: conditional probability
      * is modeled using a neural network with parameters $Θ$. 
      * These parameters are trained using stochastic gradient descent [51].

  

  * use a **multi-layer Transformer decoder** [34] for the language model,

    * is a variant of the transformer [62]. 
      * a **multi-headed self-attention** operation over the input context tokens 
      * followed by **position-wise feedforward layers** to produce an output distribution over target tokens:

    $$
    h_0 = UW_e + W_p
    $$

    $$
    h_l = transformer\_block(h_{l−1})∀i ∈ [1, n]
    $$

    $$
    P(u) = softmax(h_nW^T_e )
    $$

    * $U = (u_k, . . . , u_1)$ : context vector of tokens

    * $n$: the number of layers

    * $W_e$: token embedding matrix

    * $Wp$: position embedding matrix

      

### 3.2 Supervised fine-tuning

* After training the model with the objective in Eq. 1, 

  * **adapt the parameters** to the **supervised target task**. 

* a **labeled dataset** $C$

  * each instance consists of **a sequence of input tokens, $x^1, . . . , x^m$**, along with **a label $y$**. 

* The inputs 

  * passed through our **pre-trained model** to **<u>obtain the final transformer block’s activation $h_m^l$</u>** , 
  * is then fed into an **added linear output layer** with parameters $Wy$ to predict $y$:

  $$
  P(y|x^1, . . . , x^m) = softmax(h_m^lWy)
  $$

* This gives us the following **objective** to **maximize**:
  $$
  L_2(\mathcal{C}) = \sum_{(x,y)} log P(y|x^1, . . . , x^m)
  $$

* including **language modeling** as an <u>auxiliary objective</u> to the **fine-tuning** helped learning by 

  * (a) improving **generalization** of the supervised model, 

  * (b) **accelerating** convergence. 

  * we optimize the following **objective** (with **weight λ**):
    $$
    L_3(\mathcal{C}) = L_2(\mathcal{C}) + λ ∗ L_1(\mathcal{C})
    $$



### 3.3 Task-specific input transformations

* task input
  * some tasks(text classification) can directly fine-tune our model as described above. 
  * other tasks(question answering or textual entailment) have **structured inputs** 
    * such as ordered sentence pairs, or triplets of document, question, and answers. 
* Previous work  
  * proposed learning task specific architectures on top of transferred representations [44].
  * Such an approach re-introduces a significant amount of task-specific customization 
  * does not use transfer learning for these additional architectural components. 
* use a **traversal-style approach** [52], 
  * convert structured inputs into an ordered sequence that our pre-trained model can process. 
* All transformations include adding randomly initialized start and end tokens (<s><e>)

|                                            | input transformations                                        | note                                                         |
| ------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Textual entailment                         | **concatenate the premise $p$ and hypothesis $h$ token sequences**, with a delimiter token ($) in between |                                                              |
| Similarity                                 | modify the input sequence to contain both possible sentence orderings (with a delimiter in between) and process each **independently to produce two sequence representations $h_m^l$** which are **added element-wise** before being fed into the linear output layer. | there is no inherent ordering of the two sentences being compared.<br/> |
| Question Answering & Commonsense Reasoning | concatenate the document context and question with each possible answer, adding a delimiter token in between to get $[z; q; \$; a_k]$. Each of these sequences are processed independently with our model and then **normalized via a softmax layer** to produce an output distribution over possible answers | given a context document $z$, a question $q$, and a set of possible answers ${a_k}$. |

 ![img](http://jalammar.github.io/images/openai-input%20transformations.png)

# 4. Experiments 

| step         | task                       | Dataset                                                      |
| ------------ | -------------------------- | ------------------------------------------------------------ |
| training  LM |                            | BooksCorpus dataset [71]                                     |
| task         | Natural language inference | SNLI [5], MultiNLI [66], Question NLI [64], RTE [4], SciTail [25] |
|              | Question Answering         | RACE [30], Story Cloze [40]                                  |
|              | Sentence similarity        | MSR Paraphrase Corpus [14], Quora Question Pairs [9], STS Benchmark [6] |
|              | Classification             | Stanford Sentiment Treebank-2 [54], CoLA [65]                |

### 4.1 Setup Unsupervised pre-training 

*  BooksCorpus dataset [71] 
   * It contains over 7,000 unique unpublished books from a variety of genres(Adventure, Fantasy,  Romance)
   * An alternative dataset, the 1B Word Benchmark, is approximately the same size but is shuffled at a sentence level - destroying long-range structure. 

|             | detail                                                       |
| ----------- | ------------------------------------------------------------ |
| model  spec | - a **12-layer** **decoder-only** transformer <br/>- masked self-attention heads (768 dimensional states and 12 attention heads)<br/>- Adam optimization <br/>- a max learning rate of 2.5e-4<br/>- learning rate was increased linearly from zero over the first 2000 updates and annealed to 0 using a cosine schedule<br/>- **100 epochs** on **minibatches of 64** randomly sampled, contiguous sequences of **512 tokens**. <br/>- a simple weight initialization of N(0, 0.02) <br/>- a bytepair encoding (BPE) vocabulary with 40,000 merges [53] and residual, embedding, and attention dropouts with a rate of 0.1 for regularization<br/>- a modified version of L2 regularization with w = 0.01 on all non bias or gain weights<br/>- Gaussian Error Linear Unit (GELU)<br/>- **learned position embeddings** |
| fine-tuning | - reuse the hyperparameter settings from unsupervised pre-training. <br/>- add dropout to the classifier with a rate of 0.1. <br/>- For most tasks, a learning rate of 6.25e-5 and a **batchsize of 32**. <br/>- **3 epochs** of training was sufficient for most cases. <br/>- use **a linear learning rate decay** schedule with warmup over 0.2% of training. <br/>- **λ** was set to **0.5** |

* use the ftfy library2 
  * to clean the raw text in BooksCorpus, standardize some punctuation and whitespace
* use the spaCy tokenizer.3



[출처]

[Radford, A., Narasimhan, K., Salimans, T., & Sutskever, I. (2018). Improving language understanding by generative pre-training.]( https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)

[The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)](http://jalammar.github.io/illustrated-bert/)

[고려대 강필성 교수님 강의](
