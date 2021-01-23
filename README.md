# NLP Paper

![image](https://user-images.githubusercontent.com/65707664/105603151-e61e5a00-5dd8-11eb-81b3-7e1328bf6ccd.png)

> Parameter counts of several recently released pretrained language models.
>
> source: DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter

# 1. Deep Learning from scrach 2_sub study

> 모두의 연구소 풀잎스쿨 11기 밑바닥부터 더 딥하게 배워보자 딥러닝 서브스터디 
>
> 매주 1회 진행 (20/5/28~7/10)

### paper1 (05/28/20)

- [Efficient Estimation of Word Representations in Vector Space (original word2vec paper)](https://arxiv.org/pdf/1301.3781.pdf)

### paper2 (06/04/20)

- [Distributed Representations of Words and Phrases and their Compositionality (negative sampling paper)](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)

### paper3 (06/11/20)

- [Understanding LSTM Networks (blog post overview)](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [The Unreasonable Effectiveness of Recurrent Neural Networks (blog post overview)](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)

### paper4 (06/18/20, cancel)

- [Sequence to Sequence Learning with Neural Networks (original seq2seq NMT paper)](https://arxiv.org/pdf/1409.3215.pdf)

### paper5 (06/25/20)

- [Neural Machine Translation by Jointly Learning to Align and Translate (original seq2seq+attention paper)](https://arxiv.org/pdf/1409.0473.pdf)

### paper6 in main study (06/30/20)

- [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)

### paper7 in main study (07/07/20)

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)

### paper8 (07/10/20)

- [Deep contextualized word representations(ELMO)](https://arxiv.org/pdf/1802.05365.pdf)

# 2. beyondBERT

> 모두의 연구소 풀잎스쿨 11.5기 beyondBERT 
>
> 매주 1회 진행 (20/06/20~8/29)

### week02 (06/20/20)

- [The Bottom-up Evolution of Representations in the Transformer: A Study with Machine Translation and Language Modeling Objectives](https://arxiv.org/abs/1909.01380)
- [How multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502)

### week03 (06/27/20)

- [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942)
- [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555)

### week04 (07/04/20)

- [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461)
- [Data Augmentation using Pre-trained Transformer Models](https://arxiv.org/abs/2003.02245)

### week05 (07/11/20)

- [Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451)
- [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)

### week06 (07/18/20)

- [Mask-Predict: Parallel Decoding of Conditional Masked Language Models](https://arxiv.org/abs/1904.09324)
- [Unsupervised Data Augmentation for Consistency Training](https://arxiv.org/abs/1904.12848)
  -> Don't Stop Pretraining: Adapt Language Models to Domains and Tasks

### week07 (07/25/20)

- [You Impress Me: Dialogue Generation via Mutual Persona Perception](https://arxiv.org/abs/2004.05388)
- [Recipes for building an open-domain chatbot](https://arxiv.org/abs/2004.13637)

### week08 (08/01/20)

- [ToD-BERT: Pre-trained Natural Language Understanding for Task-Oriented Dialogues](https://arxiv.org/abs/2004.06871)
- [A Simple Language Model for Task-Oriented Dialogue](https://arxiv.org/abs/2005.00796)

### week09 (08/08/20)

- [ReCoSa: Detecting the Relevant Contexts with Self-Attention for Multi-turn Dialogue Generation](https://arxiv.org/abs/1907.05339)
- [FastBERT: a Self-distilling BERT with Adaptive Inference Time](https://arxiv.org/abs/2004.02178)

### week10 (08/22/20)

- [PoWER-BERT: Accelerating BERT inference for Classification Tasks](https://arxiv.org/abs/2001.08950)
- [TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351)

### week11 (08/29/20)

- [GPT3: Language Models are Few-Shot Learners](https://arxiv.org/pdf/2005.14165.pdf)
- [T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf)

# 3. Model Implematation (+ Code Review)

> NLP paper reading 및 model implementation 스터디 
>
> 매주 1회 (20/07/20~현재 진행중)

| No   | Model       | Framework(code)                                              | Paper                                                        | Author                                                       | Submission date |
| ---- | ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | :-------------- |
| S1-1 | Transformer | [Tensorflow(tutorial)](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/text/transformer.ipynb) | [Attention Is All You Need](https://arxiv.org/abs/1706.03762.pdf) | Google                                                       | 2017/6          |
| S1-2 | ELMO        | [AllenNLP(GitHub)](https://github.com/allenai/allennlp/blob/main/allennlp/modules/elmo.py) | [Deep contextualized word representations](https://arxiv.org/abs/1802.05365.pdf) | [AllenNLP](https://allennlp.org/elmo)                        | 2018/2          |
| S1-3 | GPT         | [TensorFlow(github)](https://github.com/openai/finetune-transformer-lm) | Improving Language Understanding with Unsupervised Learning  | [OpenAI(post)](https://openai.com/blog/language-unsupervised/) | 2018/6          |
| S1-4 | BERT        | [TensorFlow(github)](https://github.com/google-research/bert) | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805.pdf) | Google                                                       | 2018/10         |
| S2-1 | GPT2        | [TensorFlow(github)](https://github.com/openai/gpt-2)        | [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | [OpenAI(post)](https://openai.com/blog/better-language-models/) | 2019/2          |
| S2-2 | MASS        | [(github)](https://github.com/microsoft/MASS)                | [MASS: Masked Sequence to Sequence Pre-training for Language Generation](https://arxiv.org/abs/1905.02450.pdf) | Microsoft                                                    | 2019/5          |
| S2-3 | XLNet       |                                                              | [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237) | +Google                                                      | 2019/6          |
| S2-4 | RoBERTa     | [(github)](https://github.com/pytorch/fairseq/)              | [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) | Facebook                                                     | 2019/7          |
| S2-5 | ALBERT      |                                                              | [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942) |                                                              | 2019/9          |
| S2-6 | DistilBERT  |                                                              | [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/pdf/1910.01108.pdf) | Hugging Face                                                 | 2019/10         |
| S2-7 | BART        |                                                              | [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) |                                                              | 2019/10         |
| S2-8 | ELECTRA     |                                                              | [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) |                                                              | 2020/3          |

```markdown
* Transformer
- 구현 언어:
- 구현 레러펀스:
- code:

* ELMO
- 구현 언어:
- 구현 레러펀스:
- code:

* GPT
- 구현 언어:
- 구현 레러펀스:
- code:

* BERT
- 구현 언어:
- 구현 레러펀스:
- code:

* GPT2
- 구현 언어:
- 구현 레러펀스:
- code:
```

## Season1: 2020.7~12

### week1 (07/20/20) 

 - Study Planning

### week2 (08/10/20) 

 - Transformer: architecture

### week3 (08/21/20)

- Transformer: label smoothing/beam search

### week4 (08/28/20)

- Transformer: trainning/multi-GPU/experiment

### week5 (09/04/20) 

 - ELMo paper review

### week6 (09/14/20)

- ELMo char-CNN layer

### week7 (09/21/20) -> update할것

- model

### week8 (09/28/20)

* model 

###  week9 (10/05/20)

* model

###  week10 (10/12/20)

* model

###  week11 (10/19/20)

* model

###  week12 (10/26/20)

* model

###  week13 (11/02/20)

* model

###  week14 (11/09/20)

* model

###  week15 (11/16/20)

* model

###  week16 (11/23/20)

* model

###  week17 (11/30/20)

* model

###  week18 (12/07/20)

* model

###  week19 (12/14/20)

* BERT

###  week20 (12/21/20)

* BERT

--------------------------------------

## Season2. 2021.1~

###  week21 (1/16/21)

* GPT2 paper discussion(1) (~2.2 Input Representation) 

###  week22 (1/20/21)

* GPT2 paper discussion(2) (3. Experiments~)

###  week23 (1/28/21)

* GPT2 paper discussion(2) (3. Experiments~)

###  week24 (2/4/21)

* model 

###  week25 (2/18/21)

* model 

###  week26 (2/25/21)

* model 

###  week27 (3/4/21)

* model 

###  week28 (3/11/21)

* model 

###  week29 (3/18/21)

* model 

###  week20 (3/25/21)

* model 







