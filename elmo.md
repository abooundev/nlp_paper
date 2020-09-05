# Deep contextualized word representations

## 3 ELMo: Embeddings from Language Models

- ELMo word representations 
  - are **functions of the entire input sentence**
  - are computed on **top of two-layer biLMs** with **character convolution**s (Sec. 3.1),
  - as a linear function of the internal network states (Sec. 3.2).
- This setup allows us to do **semi-supervised learning**
  -  **biLM** 
     - is pretrained at a large scale (Sec. 3.4) 
     - easily incorporated into a wide range of existing neural NLP architectures (Sec. 3.3).

## 3.1 Bidirectional language models

<img src="/Users/csg/Library/Application Support/typora-user-images/image-20200904205654280.png" alt="image-20200904205654280" style="zoom:50%;" />

<img src="/Users/csg/Library/Application Support/typora-user-images/image-20200904205124476.png" alt="image-20200904205124476" style="zoom:50%;" />

<img src="/Users/csg/Library/Application Support/typora-user-images/image-20200904204703636.png" alt="image-20200904204703636" style="zoom:50%;" />

- ### forward language model 

  - Given a sequence of N tokens $(t_1, t_2, ..., t_3)$

  - computes the probability of the sequence 

  - by modeling the probability of token $t_k$ given the history  $(t_1, t_2, ..., t_{k-1})$ : 

    <img src="/Users/csg/Library/Application Support/typora-user-images/image-20200904151634514.png" alt="image-20200904151634514" style="zoom:60%;" />

    

- Recent state-of-the-art neural language models

  -  compute **a context-independent token representation** $x^{LM}_K$ (via token embeddings or a CNN over characters) 
  -  pass it through L layers of forward LSTMs. 

* At each position $k$, each LSTM layer 

  * outputs **a context-dependent representation** $\overrightarrow{h^{LM}_{k,j}}$ where $j = 1,...,L$.
  * The top layer LSTM output, $\overrightarrow{h^{LM}_{k,j}}$  , is used to predict the next token $t_{k+1}$ with a Softmax layer.

* ### backward LM 

  * is similar to a forward LM, except it runs over the sequence in reverse

  * predicting the previous token given the future context:

    <img src="/Users/csg/Library/Application Support/typora-user-images/image-20200904200550169.png" alt="image-20200904200550169" style="zoom:50%;" />

  * It can be implemented in an analogous way to a forward LM, 

  * with each backward LSTM layer $j$ in a $L$ layer deep model producing representations $\overleftarrow{h^{LM}_{k,j}}$ of $t_k$ given $(t_{k+1},...,t_N)$.

* ### A biLM 

  * combines both a forward and backward LM

  * jointly maximizes the log likelihood of the forward and backward directions:

    <img src="/Users/csg/Library/Application Support/typora-user-images/image-20200904201012206.png" alt="image-20200904201012206" style="zoom:50%;" />

  * *We tie the parameters for both the token representation (Θx ) and Softmax layer (Θs) in the forward and backward direction while maintaining separate parameters for the LSTMs in each direction.*

  * we share some weights between directions 

  * a new approach for learning word representations that are a linear combination of the biLM layers.

## 3.2 ELMo

<img src="/Users/csg/Library/Application Support/typora-user-images/image-20200904205400548.png" alt="image-20200904205400548" style="zoom:50%;" />

![image-20200904205550133](/Users/csg/Library/Application Support/typora-user-images/image-20200904205550133.png)

* ELMo

  * a **task specific combination** of the **intermediate layer representations** in the biLM

  * For each token $t_k$, 

    * a $L$-layer biLM computes a set of $2L + 1$ representations

    <img src="/Users/csg/Library/Application Support/typora-user-images/image-20200904151548428.png" alt="image-20200904151548428" style="zoom:50%;" />

    * $R_k$: token $k$ representation
    * $h^{LM}_{k,0}$: token layer
    * $h^{LM}_{k,j} = [\overrightarrow{h^{LM}_{k,j}}; \overleftarrow{h^{LM}_{k,j}}]$ for each biLSTM layer.

  * For inclusion in a downstream model, 

    * ELMo **collapses all layers in $R$ into a single vector**,

      <img src="/Users/csg/Library/Application Support/typora-user-images/image-20200904154543714.png" alt="image-20200904154543714" style="zoom:50%;" />

    * we compute **a task specific weighting** of all biLM layers:

      <img src="/Users/csg/Library/Application Support/typora-user-images/image-20200904153928744.png" alt="image-20200904153928744" style="zoom:50%;" />

      * $s^{task}$: softmax-normalized weights 
      * $γ^{task}$: scalar parameter
        * allows the **task mode**l to scale the entire ELMo vector. 
        * $γ$ is of practical importance to aid the **optimization** process (see supplemental material for details). 
      * Considering that the **activations** of each biLM layer have a different distribution, in some cases it also helped to apply **layer normalization** to each biLM layer before weighting.


## 3.3 Using biLMs for supervised NLP tasks

<img src="/Users/csg/Library/Application Support/typora-user-images/image-20200904205752709.png" alt="image-20200904205752709" style="zoom:50%;" />

* Given a pre-trained biLM and a supervised architecture for a target NLP task, 
* use a pre-trained biLM to improve the task mode
  * run the biLM
  * record all of the layer representations for each word
  * let the **end task model** learn **a linear combination of these representations**
    * (1) consider the lowest layers of the supervised model without the biLM

      * ~~Most supervised NLP models share a common architecture at the lowest layers~~, allowing us to add ELMo in a consistent, unified manner. 
      * Given a sequence of tokens $(t_1,...,t_N)$, it is standard to form **a context-independent token representation** $x_k$ for each token position using **pre-trained word embeddings** and optionally character-based representations. Then,
      * ~~the model forms a context-sensitive representation $h_k$, typically using either bidirectional RNNs, CNNs, or feed forward networks.~~

    * (2) To add ELMo to the supervised model
    * first **freeze** the weights of the biLM
      * <u>**concatenate**</u> the ELMo vector $ELMo^{task}_k$ **with** $x_k$ 
      * **pass** the ELMo enhanced representation $[x_k; ELMo^{task}_k]$ **into** the **task RNN**. 
* we observe further improvements for some tasks (e.g., SNLI, SQuAD) by also including ELMo at the **output of the task RNN** 

  * *by introducing another set of output specific linear weights* 
  * replacing $h_k$ with$[h_k; ELMo^{task}_k]$  
* As the remainder of the supervised model remains unchanged, these additions can happen within the context of more complex neural models. 
  * For example, see the SNLI experiments(Sec. 4) where a bi-attention layer follows the biLSTMs, 
  * or the coreference resolution experiments where a clustering model is layered on top of the biLSTMs.
* it beneficial 
  * to add **a moderate amount of dropout** to ELMo
  * (in some cases) to **regularize** the ELMo weights by adding $λ∥w∥^2_2$ to the loss. 
    * This imposes an inductive bias on the ELMo weights to stay close to an average of all biLM layers.



## 3.4 Pre-trained bidirectional language model architecture

* To balance overall language model perplexity with model size and computational requirements for downstream tasks **while maintaining a purely character-based input representation**, 

  * we halved all embedding and hidden dimensions from the single best model CNN-BIG-LSTM in Jo ́zefowicz et al. (2016).
  * The final model uses 
    * **L = 2** biLSTM layers with 4096 units 
    * **512 dimension** projections 
    * a **residual connection** from the first to second layer.

* the biLM provides **three layers of representations** for each input token, including **those** outside the training set due to the purely **character input.** 

* After training for 10 epochs on the 1B Word Benchmark (Chelba et al., 2014), 

  * the average forward and backward perplexities is 39.7,
  * we found the forward and backward perplexities to be approximately equal, with the backward value slightly lower.

  
