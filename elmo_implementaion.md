# ELMO implementaion

## Tensorflow prediction

source: https://tfhub.dev/google/elmo/3

### overview

Computes **contextualized word representations** 

- using character-based word representations and bidirectional LSTMs, 
- as described in the paper "Deep contextualized word representations" [1].

This modules supports **inputs** both in the form of 

 * raw text strings
 * tokenized text strings.

The module **outputs** 

 * **fixed embeddings at each LSTM layer**
 * **a learnable aggregation of the 3 layers**
* **a fixed mean-pooled vector representation of the input.**

The complex architecture achieves state of the art results on several benchmarks. 

Note that this is a very computationally expensive module compared to word embedding modules that only perform embedding lookups. The use of an accelerator is recommended.

#### Trainable parameters

The module exposes **4 trainable scalar weights** for layer aggregation.

#### Example use

We set the `trainable` parameter to `True` when creating the module 

* so that the 4 scalar weights (as described in the paper) can be trained. 

In this setting, the module still keeps all other parameters fixed.



### Inputs

The module [defines two signatures](https://www.tensorflow.org/hub/tf1_hub_module#applying_a_module):

​	* `default`

​	*  `tokens`

With the `default` signature, 

* the module takes untokenized sentences as input. 
* The input tensor is a `string` tensor with shape `[batch_size]`.
*  The module tokenizes each string by **splitting on spaces**.

With the `tokens` signature, the module takes tokenized sentences as input. The input tensor is a `string` tensor with shape `[batch_size, max_length]` and an `int32` tensor with shape `[batch_size]` corresponding to the sentence length. The length input is necessary to exclude padding in the case of sentences with varying length.

### Outputs

The output dictionary contains:

- `word_emb`: the character-based word representations 
  - with shape `[batch_size, max_length, 512]`.
- `lstm_outputs1`: the first LSTM hidden state 
  - with shape `[batch_size, max_length, 1024]`.
- `lstm_outputs2`: the second LSTM hidden state 
  - with shape `[batch_size, max_length, 1024]`.
- `elmo`: the weighted sum of the 3 layers, where the weights are trainable. 
  - This tensor has shape `[batch_size, max_length, 1024]`
- `default`: a fixed mean-pooling of all contextualized word representations 
  - with shape `[batch_size, 1024]`.



## AllenAI bilm-tf

source: https://github.com/allenai/bilm-tf

test 코드 돌려보기

### training

- The script `bin/train_elmo.py` has hyperparameters for training the model. 

- The original model was trained on 3 GTX 1080 for 10 epochs, taking about two weeks.

- For input processing,

  -  we used the raw 1 Billion Word Benchmark dataset [here](http://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz),(1.7G)

  -  the existing vocabulary of **793,471 tokens**, including `<S>`, `</S>` and `<UNK>`. 

    - You can find our vocabulary file [here](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/vocab-2016-09-10.txt). 

    - ```
      </S>
      <S>
      <UNK>
      the
      ,
      .
      to
      of
      and
      a
      in
      "
      's
      that
      for
      on
      is
      The
      was
      ```

  - At the model input, all text used the full character based representation, including tokens outside the vocab. 

  - For the softmax output we replaced OOV tokens with `<UNK>`.

* The model was trained with a fixed size window of 20 tokens. 
* The batches were constructed 
  * by padding sentences with `<S>` and `</S>`, 
  * then packing tokens from one or more sentences into each row to fill completely fill each batch. Partial sentences and the LSTM states were carried over from batch to batch so that the language model could use information across batches for context, but backpropogation was broken at each batch boundary.

```
export CUDA_VISIBLE_DEVICES=0,1,2
python bin/train_elmo.py \
    --train_prefix='/path/to/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/*' \
    --vocab_file /path/to/vocab-2016-09-10.txt \
    --save_dir /output_path/to/checkpoint
```



### train_elmo.py

- batch_size = 128
- n_train_tokens = 768648884
- n_tokens_vocab =  793471

```python
import argparse
import numpy as np
from bilm.training import train, load_options_latest_checkpoint, load_vocab
from bilm.data import BidirectionalLMDataset


def main(args):
    # load the vocab
    vocab = load_vocab(args.vocab_file, 50)

    # define the options
    batch_size = 128  # batch size for each GPU
    n_gpus = 3

    # number of tokens in training data (this for 1B Word Benchmark)
    n_train_tokens = 768648884

    options = {
     'bidirectional': True,

      'char_cnn': {'activation': 'relu',
      'embedding': {'dim': 16},
      'filters': [[1, 32],
       [2, 32],
       [3, 64],
       [4, 128],
       [5, 256],
       [6, 512],
       [7, 1024]],
      'max_characters_per_token': 50,
      'n_characters': 261,
      'n_highway': 2},
    
     'dropout': 0.1,
    
     'lstm': {
      'cell_clip': 3,
      'dim': 4096,
      'n_layers': 2,
      'proj_clip': 3,
      'projection_dim': 512,
      'use_skip_connections': True},
    
     'all_clip_norm_val': 10.0,
    
     'n_epochs': 10,
     'n_train_tokens': n_train_tokens,
     'batch_size': batch_size,
     'n_tokens_vocab': vocab.size,
     'unroll_steps': 20,
     'n_negative_samples_batch': 8192,
    }

    prefix = args.train_prefix
    data = BidirectionalLMDataset(prefix, vocab, test=False,
                                      shuffle_on_load=True)

    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir
    train(options, data, n_gpus, tf_save_dir, tf_log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', help='Vocabulary file')
    parser.add_argument('--train_prefix', help='Prefix for train files')

    args = parser.parse_args()
    main(args)
```



### BidirectionalLMDataset



### def load_covab()

```python
def load_vocab(vocab_file, max_word_length=None):
    if max_word_length:
        return UnicodeCharsVocabulary(vocab_file, max_word_length,
                                      validate_file=True)
    else:
        return Vocabulary(vocab_file, validate_file=True)
```



### class UnicodeCharsVocabulary()

*    **a character id**
     *    that is used to map words to arrays of character ids.
     *    The character ids are defined by **ord(c) for c in word.encode('utf-8')**
     *    This limits the total number of possible char ids to **256**.
     *    char ids 0-255 come from utf-8 encoding bytes
*    To this we add **5 additional special ids**: 
     *    begin sentence ->  self.bos_char = 256
     *    end sentence -> self.eos_char = 257 
     *    begin word -> self.bow_char = 258
     *    end word ->  self.eow_char = 259
     *    padding -> self.pad_char = 260
*    for prediction, we **add +1 to the output ids** from this class to create a special padding id (=0)
     *    we suggest you use the `Batcher`, `TokenBatcher`, and `LMDataset` classes 

```python
class UnicodeCharsVocabulary(Vocabulary):
    """Vocabulary containing character-level and word level information.
    Has a word vocabulary that is used to lookup word ids and
    a character id that is used to map words to arrays of character ids.
    The character ids are defined by ord(c) for c in word.encode('utf-8')
    This limits the total number of possible char ids to 256.
    To this we add 5 additional special ids: begin sentence, end sentence,
        begin word, end word and padding.
    WARNING: for prediction, we add +1 to the output ids from this
    class to create a special padding id (=0).  As a result, we suggest
    you use the `Batcher`, `TokenBatcher`, and `LMDataset` classes instead
    of this lower level class.  If you are using this lower level class,
    then be sure to add the +1 appropriately, otherwise embeddings computed
    from the pre-trained model will be useless.
    """
    def __init__(self, filename, max_word_length, **kwargs):
        super(UnicodeCharsVocabulary, self).__init__(filename, **kwargs)
        self._max_word_length = max_word_length # 50

        # char ids 0-255 come from utf-8 encoding bytes
        # assign 256-300 to special chars
        self.bos_char = 256  # <begin sentence>
        self.eos_char = 257  # <end sentence>
        self.bow_char = 258  # <begin word>
        self.eow_char = 259  # <end word>
        self.pad_char = 260 # <padding>

        num_words = len(self._id_to_word)

        self._word_char_ids = np.zeros([num_words, max_word_length], #num_words, 50
            dtype=np.int32)

        # the charcter representation of the begin/end of sentence characters
        def _make_bos_eos(c):
            r = np.zeros([self.max_word_length], dtype=np.int32) #50
            r[:] = self.pad_char
            r[0] = self.bow_char
            r[1] = c
            r[2] = self.eow_char
            return r
          
        self.bos_chars = _make_bos_eos(self.bos_char)
        self.eos_chars = _make_bos_eos(self.eos_char)

        for i, word in enumerate(self._id_to_word):
            self._word_char_ids[i] = self._convert_word_to_char_ids(word)

        self._word_char_ids[self.bos] = self.bos_chars
        self._word_char_ids[self.eos] = self.eos_chars
        # TODO: properly handle <UNK>

    @property
    def word_char_ids(self):
        return self._word_char_ids

    @property
    def max_word_length(self):
        return self._max_word_length

    def _convert_word_to_char_ids(self, word):
        code = np.zeros([self.max_word_length], dtype=np.int32) #50
        code[:] = self.pad_char

        #######################################################
        word_encoded = word.encode('utf-8', 'ignore')[:(self.max_word_length-2)] #50-2=48
        
        code[0] = self.bow_char #begin of word
        
        for k, chr_id in enumerate(word_encoded, start=1):
            code[k] = chr_id
        
        code[len(word_encoded) + 1] = self.eow_char #end of word

        return code

    def word_to_char_ids(self, word):
        if word in self._word_to_id:
            return self._word_char_ids[self._word_to_id[word]]
        else:
            return self._convert_word_to_char_ids(word)

    def encode_chars(self, sentence, reverse=False, split=True):
        '''
        Encode the sentence as a white space delimited string of tokens.
        '''
        if split:
            chars_ids = [self.word_to_char_ids(cur_word)
                     for cur_word in sentence.split()]
        else:
            chars_ids = [self.word_to_char_ids(cur_word)
                     for cur_word in sentence]
        if reverse:
            return np.vstack([self.eos_chars] + chars_ids + [self.bos_chars])
        else:
            return np.vstack([self.bos_chars] + chars_ids + [self.eos_chars])

```



### class Vocabulary()

* encode
  * Convert **a sentence** to **a list of ids**, **with special tokens** added.
  * Sentence is **a single string** with **tokens separated by whitespace**.
  * If reverse, then the sentence is assumed to be reversed, and this method will swap the BOS/EOS tokens appropriately.
* decode 
  * Convert **a list of ids** to **a sentence**, with space inserted.

```python
class Vocabulary(object):
    '''
    A token vocabulary.  Holds a map from token to ids and provides
    a method for encoding text to a sequence of ids.
    '''
    def __init__(self, filename, validate_file=False):
        '''
        filename = the vocabulary file.  It is a flat text file with one
            (normalized) token per line.  In addition, the file should also
            contain the special tokens <S>, </S>, <UNK> (case sensitive).
        '''
        self._id_to_word = []
        self._word_to_id = {}
        self._unk = -1 #word_name == '<UNK>'
        self._bos = -1 #word_name == '<S>'
        self._eos = -1 #word_name == '</S>'

        with open(filename) as f:
            idx = 0
            for line in f:
                word_name = line.strip()
                if word_name == '<S>':
                    self._bos = idx
                elif word_name == '</S>':
                    self._eos = idx
                elif word_name == '<UNK>':
                    self._unk = idx
                if word_name == '!!!MAXTERMID':
                    continue

                self._id_to_word.append(word_name)
                self._word_to_id[word_name] = idx
                idx += 1

        # check to ensure file has special tokens
        if validate_file:
            if self._bos == -1 or self._eos == -1 or self._unk == -1:
                raise ValueError("Ensure the vocabulary file has "
                                 "<S>, </S>, <UNK> tokens")

    @property
    def bos(self):
        return self._bos

    @property
    def eos(self):
        return self._eos

    @property
    def unk(self):
        return self._unk

    @property
    def size(self):
        return len(self._id_to_word)

    def word_to_id(self, word):
        if word in self._word_to_id:
            return self._word_to_id[word]
        return self.unk

    def id_to_word(self, cur_id):
        return self._id_to_word[cur_id]

    def decode(self, cur_ids):
        """Convert a list of ids to a sentence, with space inserted."""
        return ' '.join([self.id_to_word(cur_id) for cur_id in cur_ids])

    def encode(self, sentence, reverse=False, split=True):
        """Convert a sentence to a list of ids, with special tokens added.
        Sentence is a single string with tokens separated by whitespace.
        If reverse, then the sentence is assumed to be reversed, and
            this method will swap the BOS/EOS tokens appropriately."""

        if split:
            word_ids = [
                self.word_to_id(cur_word) for cur_word in sentence.split()
            ]
        else:
            word_ids = [self.word_to_id(cur_word) for cur_word in sentence]

        if reverse:
            return np.array([self.eos] + word_ids + [self.bos], dtype=np.int32)
        else:
            return np.array([self.bos] + word_ids + [self.eos], dtype=np.int32)


```



### def train()

-cpu/ gpu

-multi-gpu

-> transformer

```python
def train(options, data, n_gpus, tf_save_dir, tf_log_dir,
          restart_ckpt_file=None):

    # not restarting so save the options
    if restart_ckpt_file is None:
        with open(os.path.join(tf_save_dir, 'options.json'), 'w') as fout:
            fout.write(json.dumps(options))

    #------------------------------------------------------------------------------#
    with tf.device('/cpu:0'):
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        # set up the optimizer
        lr = options.get('learning_rate', 0.2)
        opt = tf.train.AdagradOptimizer(learning_rate=lr,
                                        initial_accumulator_value=1.0)

        # calculate the gradients on each GPU
        tower_grads = []
        models = []
        train_perplexity = tf.get_variable(
            'train_perplexity', [],
            initializer=tf.constant_initializer(0.0), trainable=False)
        norm_summaries = []
        
        ######################################################################
        for k in range(n_gpus):
            with tf.device('/gpu:%d' % k):
                with tf.variable_scope('lm', reuse=k > 0):
                    # calculate the loss for one model replica and get
                    #   lstm states
                    model = LanguageModel(options, True) #init() -> build()
                    loss = model.total_loss
                    models.append(model)
                    
                    # get gradients
                    grads = opt.compute_gradients(
                        loss * options['unroll_steps'],
                        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE,
                    )
                    tower_grads.append(grads)
                    
                    # keep track of loss across all GPUs
                    train_perplexity += loss
				######################################################################
        
        print_variable_summary()

        # calculate the mean of each gradient across all GPUs
        grads = average_gradients(tower_grads, options['batch_size'], options)
        grads, norm_summary_ops = clip_grads(grads, options, True, global_step)
        norm_summaries.extend(norm_summary_ops)

        # log the training perplexity
        train_perplexity = tf.exp(train_perplexity / n_gpus)
        perplexity_summmary = tf.summary.scalar(
            'train_perplexity', train_perplexity)

        # some histogram summaries.  all models use the same parameters
        # so only need to summarize one
        histogram_summaries = [
            tf.summary.histogram('token_embedding', models[0].embedding)
        ]
        
        # tensors of the output from the LSTM layer
        lstm_out = tf.get_collection('lstm_output_embeddings')
        histogram_summaries.append(
                tf.summary.histogram('lstm_embedding_0', lstm_out[0]))
        if options.get('bidirectional', False):
            # also have the backward embedding
            histogram_summaries.append(
                tf.summary.histogram('lstm_embedding_1', lstm_out[1]))

        # apply the gradients to create the training operation
        train_op = opt.apply_gradients(grads, global_step=global_step)

        # histograms of variables
        for v in tf.global_variables():
            histogram_summaries.append(tf.summary.histogram(v.name.replace(":", "_"), v))

        # get the gradient updates -- these aren't histograms, but we'll
        # only update them when histograms are computed
        histogram_summaries.extend(
            summary_gradient_updates(grads, opt, lr))

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
        summary_op = tf.summary.merge(
            [perplexity_summmary] + norm_summaries
        )
        hist_summary_op = tf.summary.merge(histogram_summaries)

        init = tf.initialize_all_variables()

    #------------------------------------------------------------------------------#    
    # do the training loop
    bidirectional = options.get('bidirectional', False)
    with tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True)) as sess:
        sess.run(init)

        # load the checkpoint data if needed
        if restart_ckpt_file is not None:
            loader = tf.train.Saver()
            loader.restore(sess, restart_ckpt_file)
            
        summary_writer = tf.summary.FileWriter(tf_log_dir, sess.graph)

        # For each batch:
        # Get a batch of data from the generator. The generator will
        # yield batches of size batch_size * n_gpus that are sliced
        # and fed for each required placeholer.
        #
        # We also need to be careful with the LSTM states.  We will
        # collect the final LSTM states after each batch, then feed
        # them back in as the initial state for the next batch

        batch_size = options['batch_size']
        unroll_steps = options['unroll_steps']
        n_train_tokens = options.get('n_train_tokens', 768648884)
        n_tokens_per_batch = batch_size * unroll_steps * n_gpus
        n_batches_per_epoch = int(n_train_tokens / n_tokens_per_batch)
        n_batches_total = options['n_epochs'] * n_batches_per_epoch
        print("Training for %s epochs and %s batches" % (
            options['n_epochs'], n_batches_total))

        # get the initial lstm states
        init_state_tensors = []
        final_state_tensors = []
        for model in models:
            init_state_tensors.extend(model.init_lstm_state)
            final_state_tensors.extend(model.final_lstm_state)

        char_inputs = 'char_cnn' in options
        if char_inputs:
            max_chars = options['char_cnn']['max_characters_per_token']

        if not char_inputs:
            feed_dict = {
                model.token_ids:
                    np.zeros([batch_size, unroll_steps], dtype=np.int64)
                for model in models
            }
        else:
            feed_dict = {
                model.tokens_characters:
                    np.zeros([batch_size, unroll_steps, max_chars],
                             dtype=np.int32)
                for model in models
            }

        if bidirectional:
            if not char_inputs:
                feed_dict.update({
                    model.token_ids_reverse:
                        np.zeros([batch_size, unroll_steps], dtype=np.int64)
                    for model in models
                })
            else:
                feed_dict.update({
                    model.tokens_characters_reverse:
                        np.zeros([batch_size, unroll_steps, max_chars],
                                 dtype=np.int32)
                    for model in models
                })

        init_state_values = sess.run(init_state_tensors, feed_dict=feed_dict)

        t1 = time.time()
        data_gen = data.iter_batches(batch_size * n_gpus, unroll_steps)
        for batch_no, batch in enumerate(data_gen, start=1):

            # slice the input in the batch for the feed_dict
            X = batch
            feed_dict = {t: v for t, v in zip(
                                        init_state_tensors, init_state_values)}
            for k in range(n_gpus):
                model = models[k]
                start = k * batch_size
                end = (k + 1) * batch_size

                feed_dict.update(
                    _get_feed_dict_from_X(X, start, end, model,
                                          char_inputs, bidirectional)
                )

            # This runs the train_op, summaries and the "final_state_tensors"
            #   which just returns the tensors, passing in the initial
            #   state tensors, token ids and next token ids
            if batch_no % 1250 != 0:
                ret = sess.run(
                    [train_op, summary_op, train_perplexity] +
                                                final_state_tensors,
                    feed_dict=feed_dict
                )

                # first three entries of ret are:
                #  train_op, summary_op, train_perplexity
                # last entries are the final states -- set them to
                # init_state_values
                # for next batch
                init_state_values = ret[3:]

            else:
                # also run the histogram summaries
                ret = sess.run(
                    [train_op, summary_op, train_perplexity, hist_summary_op] + 
                                                final_state_tensors,
                    feed_dict=feed_dict
                )
                init_state_values = ret[4:]
                

            if batch_no % 1250 == 0:
                summary_writer.add_summary(ret[3], batch_no)
            if batch_no % 100 == 0:
                # write the summaries to tensorboard and display perplexity
                summary_writer.add_summary(ret[1], batch_no)
                print("Batch %s, train_perplexity=%s" % (batch_no, ret[2]))
                print("Total time: %s" % (time.time() - t1))

            if (batch_no % 1250 == 0) or (batch_no == n_batches_total):
                # save the model
                checkpoint_path = os.path.join(tf_save_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=global_step)

            if batch_no == n_batches_total:
                # done training!
                break

```



#### class LanguageModel()

* def __init__(self, options, is_training):
* def _build_word_embeddings(self):
* def _build_word_char_embeddings(self):
* **def _build(self):**
* def _build_loss(self, lstm_outputs):

```python
class LanguageModel(object):
    '''
    A class to build the tensorflow computational graph for NLMs
    All hyperparameters and model configuration is specified in a dictionary
    of 'options'.
    is_training is a boolean used to control behavior of dropout layers
        and softmax.  Set to False for testing.
    The LSTM cell is controlled by the 'lstm' key in options
    Here is an example:
     'lstm': {
      'cell_clip': 5,
      'dim': 4096,
      'n_layers': 2,
      'proj_clip': 5,
      'projection_dim': 512,
      'use_skip_connections': True},
        'projection_dim' is assumed token embedding size and LSTM output size.
        'dim' is the hidden state size.
        Set 'dim' == 'projection_dim' to skip a projection layer.
    '''
    
    #------------------------------------------------------------------------------#
    def __init__(self, options, is_training):
        self.options = options
        self.is_training = is_training
        self.bidirectional = options.get('bidirectional', False)

        # use word or char inputs?
        self.char_inputs = 'char_cnn' in self.options

        # for the loss function
        self.share_embedding_softmax = options.get(
            'share_embedding_softmax', False)
        if self.char_inputs and self.share_embedding_softmax:
            raise ValueError("Sharing softmax and embedding weights requires "
                             "word input")

        self.sample_softmax = options.get('sample_softmax', True)

        self._build()
  
    #------------------------------------------------------------------------------#
    def _build_word_embeddings(self):
        n_tokens_vocab = self.options['n_tokens_vocab'] #793471
        batch_size = self.options['batch_size'] #128
        unroll_steps = self.options['unroll_steps'] #20

        # LSTM options
        projection_dim = self.options['lstm']['projection_dim'] #512

        # the input token_ids and word embeddings
        self.token_ids = tf.placeholder(DTYPE_INT,
                               shape=(batch_size, unroll_steps), #(128, 20)
                               name='token_ids')
        
        #####################################################################
        # the word embeddings
        with tf.device("/cpu:0"):
            self.embedding_weights = tf.get_variable(
                "embedding", [n_tokens_vocab, projection_dim],
                dtype=DTYPE,
            )
            self.embedding = tf.nn.embedding_lookup(self.embedding_weights,
                                                self.token_ids)

        #####################################################################
        # if a bidirectional LM then make placeholders for reverse
        # model and embeddings
        if self.bidirectional:
            self.token_ids_reverse = tf.placeholder(DTYPE_INT,
                               shape=(batch_size, unroll_steps), #(128, 20)
                               name='token_ids_reverse')
            with tf.device("/cpu:0"):
                self.embedding_reverse = tf.nn.embedding_lookup(
                    self.embedding_weights, self.token_ids_reverse)

    
    #------------------------------------------------------------------------------#
   	    '''
        options contains key 'char_cnn': {
        'n_characters': 262,
        # includes the start / end characters
        'max_characters_per_token': 50,
        'filters': [
            [1, 32],
            [2, 32],
            [3, 64],
            [4, 128],
            [5, 256],
            [6, 512],
            [7, 512]
        ],
        'activation': 'tanh',
        # for the character embedding
        'embedding': {'dim': 16}
        # for highway layers
        # if omitted, then no highway layers
        'n_highway': 2,
        }
        '''
      
  	def _build_word_char_embeddings(self):

        #####################################################################
        batch_size = self.options['batch_size'] #128
        unroll_steps = self.options['unroll_steps'] #20
        projection_dim = self.options['lstm']['projection_dim'] #512
    
    		#####################################################################
        cnn_options = self.options['char_cnn']
        
        filters = cnn_options['filters']
        n_filters = sum(f[1] for f in filters)
       
        max_chars = cnn_options['max_characters_per_token'] #50
        char_embed_dim = cnn_options['embedding']['dim'] #16
        
        n_chars = cnn_options['n_characters'] #261
        if n_chars != 261:
            raise InvalidNumberOfCharacters(
                    "Set n_characters=261 for training see the README.md"
            )
            
        if cnn_options['activation'] == 'tanh':
            activation = tf.nn.tanh
        elif cnn_options['activation'] == 'relu':
            activation = tf.nn.relu
				
        #########################################################
        # the input character ids 
        self.tokens_characters = tf.placeholder(DTYPE_INT,
                                   shape=(batch_size, unroll_steps, max_chars),
                                   name='tokens_characters')
        
        # the character embeddings
        with tf.device("/cpu:0"):
            self.embedding_weights = tf.get_variable(
                    "char_embed", [n_chars, char_embed_dim],
                    dtype=DTYPE,
                    initializer=tf.random_uniform_initializer(-1.0, 1.0)
            )
            
            # shape (batch_size, unroll_steps, max_chars, embed_dim)
            # (128, 20, 50, 16)
            self.char_embedding = tf.nn.embedding_lookup(self.embedding_weights,
                                                    self.tokens_characters)

            #########################################################
            if self.bidirectional:
                self.tokens_characters_reverse = tf.placeholder(DTYPE_INT,
                                   shape=(batch_size, unroll_steps, max_chars),
                                   name='tokens_characters_reverse')
                
                self.char_embedding_reverse = tf.nn.embedding_lookup(
                    self.embedding_weights, self.tokens_characters_reverse)

				
    		#------------------------------------------------------------------------------#
        # the convolutions
        def make_convolutions(inp, reuse):
            with tf.variable_scope('CNN', reuse=reuse) as scope:
                convolutions = []
                """
                'filters': [
                    [1, 32],
                    [2, 32],
                    [3, 64],
                    [4, 128],
                    [5, 256],
                    [6, 512],
                    [7, 512]
                ],
                """
                for i, (width, num) in enumerate(filters):
                    if cnn_options['activation'] == 'relu':
                        # He initialization for ReLU activation
                        # with char embeddings init between -1 and 1
                        #w_init = tf.random_normal_initializer(
                        #    mean=0.0,
                        #    stddev=np.sqrt(2.0 / (width * char_embed_dim))
                        #)

                        # Kim et al 2015, +/- 0.05
                        w_init = tf.random_uniform_initializer(
                            minval=-0.05, maxval=0.05)
                    elif cnn_options['activation'] == 'tanh':
                        # glorot init
                        w_init = tf.random_normal_initializer(
                            mean=0.0,
                            stddev=np.sqrt(1.0 / (width * char_embed_dim))
                        )
                        
                    w = tf.get_variable(
                        "W_cnn_%s" % i,
                        [1, width, char_embed_dim, num], #[1, width, 16, num]
                        initializer=w_init,
                        dtype=DTYPE)
                    b = tf.get_variable(
                        "b_cnn_%s" % i, [num], dtype=DTYPE, #[num]
                        initializer=tf.constant_initializer(0.0))

                    conv = tf.nn.conv2d(
                            inp, w,
                            strides=[1, 1, 1, 1],
                            padding="VALID") + b
                    
                    # now max pool
                    conv = tf.nn.max_pool(
                            conv, [1, 1, max_chars-width+1, 1],
                            [1, 1, 1, 1], 'VALID')

                    # activation
                    conv = activation(conv)
                    conv = tf.squeeze(conv, squeeze_dims=[2])

                    convolutions.append(conv)

            return tf.concat(convolutions, 2)
	
  			#------------------------------------------------------------------------------#
        # for first model, this is False, for others it's True
        reuse = tf.get_variable_scope().reuse
        embedding = make_convolutions(self.char_embedding, reuse)

        self.token_embedding_layers = [embedding]

        if self.bidirectional:
            # re-use the CNN weights from forward pass
            embedding_reverse = make_convolutions(
                self.char_embedding_reverse, True)
	
  			#------------------------------------------------------------------------------#
        # for highway and projection layers:
        #   reshape from (batch_size, n_tokens, dim) to
        n_highway = cnn_options.get('n_highway') #2
        use_highway = n_highway is not None and n_highway > 0 #True
        use_proj = n_filters != projection_dim

        ############################################################################
        if use_highway or use_proj:
            embedding = tf.reshape(embedding, [-1, n_filters])
            if self.bidirectional:
                embedding_reverse = tf.reshape(embedding_reverse,
                    [-1, n_filters])

        ############################################################################        
        # set up weights for projection
        if use_proj:
            assert n_filters > projection_dim
            with tf.variable_scope('CNN_proj') as scope:
                    W_proj_cnn = tf.get_variable(
                        "W_proj", [n_filters, projection_dim],
                        initializer=tf.random_normal_initializer(
                            mean=0.0, stddev=np.sqrt(1.0 / n_filters)),
                        dtype=DTYPE)
                    b_proj_cnn = tf.get_variable(
                        "b_proj", [projection_dim],
                        initializer=tf.constant_initializer(0.0),
                        dtype=DTYPE)
                  
				############################################################################
        # apply highways layers
        def high(x, ww_carry, bb_carry, ww_tr, bb_tr):
            carry_gate = tf.nn.sigmoid(tf.matmul(x, ww_carry) + bb_carry)
            transform_gate = tf.nn.relu(tf.matmul(x, ww_tr) + bb_tr)
            return carry_gate * transform_gate + (1.0 - carry_gate) * x

        ############################################################################
        if use_highway:
            highway_dim = n_filters

            for i in range(n_highway):
                with tf.variable_scope('CNN_high_%s' % i) as scope:
                    W_carry = tf.get_variable(
                        'W_carry', [highway_dim, highway_dim],
                        # glorit init
                        initializer=tf.random_normal_initializer(
                            mean=0.0, stddev=np.sqrt(1.0 / highway_dim)),
                        dtype=DTYPE)
                    b_carry = tf.get_variable(
                        'b_carry', [highway_dim],
                        initializer=tf.constant_initializer(-2.0),
                        dtype=DTYPE)
                    W_transform = tf.get_variable(
                        'W_transform', [highway_dim, highway_dim],
                        initializer=tf.random_normal_initializer(
                            mean=0.0, stddev=np.sqrt(1.0 / highway_dim)),
                        dtype=DTYPE)
                    b_transform = tf.get_variable(
                        'b_transform', [highway_dim],
                        initializer=tf.constant_initializer(0.0),
                        dtype=DTYPE)

                #####################################################
                # apply highway
                embedding = high(embedding, W_carry, b_carry,
                                 W_transform, b_transform)
                
                # apply highway for bidirectional 
                if self.bidirectional:
                    embedding_reverse = high(embedding_reverse,
                                             W_carry, b_carry,
                                             W_transform, b_transform)
                
                self.token_embedding_layers.append(
                    tf.reshape(embedding, 
                        [batch_size, unroll_steps, highway_dim])
                )

        ############################################################################        
        # finally project down to projection dim if needed
        if use_proj:
            embedding = tf.matmul(embedding, W_proj_cnn) + b_proj_cnn
            
            if self.bidirectional:
                embedding_reverse = tf.matmul(embedding_reverse, W_proj_cnn) \
                    + b_proj_cnn
            
            self.token_embedding_layers.append(
                tf.reshape(embedding,
                        [batch_size, unroll_steps, projection_dim])
            )

        ############################################################################          
        # reshape back to (batch_size, tokens, dim)
        if use_highway or use_proj:
            shp = [batch_size, unroll_steps, projection_dim]
            embedding = tf.reshape(embedding, shp)
            if self.bidirectional:
                embedding_reverse = tf.reshape(embedding_reverse, shp)

        ############################################################################        
        # at last assign attributes for remainder of the model
        self.embedding = embedding
        if self.bidirectional:
            self.embedding_reverse = embedding_reverse
		
    #------------------------------------------------------------------------------#
    def _build(self):
        # size of input options
        n_tokens_vocab = self.options['n_tokens_vocab'] #793471
        batch_size = self.options['batch_size'] #128
        unroll_steps = self.options['unroll_steps'] #20

        # LSTM options
        lstm_dim = self.options['lstm']['dim'] #4096
        projection_dim = self.options['lstm']['projection_dim'] #512
        n_lstm_layers = self.options['lstm'].get('n_layers', 1) #2
        dropout = self.options['dropout'] #0.1
        keep_prob = 1.0 - dropout #0.9

        ############################################################################   
        if self.char_inputs:
            self._build_word_char_embeddings()
        else:
            self._build_word_embeddings()

        ############################################################################   
        # now the LSTMs
        # these will collect the initial states for the forward
        #   (and reverse LSTMs if we are doing bidirectional)
        self.init_lstm_state = []
        self.final_lstm_state = []

        # reverse 확인
        # get the LSTM inputs
        if self.bidirectional:
            lstm_inputs = [self.embedding, self.embedding_reverse]
        else:
            lstm_inputs = [self.embedding]

        # clipping?
        # now compute the LSTM outputs
        cell_clip = self.options['lstm'].get('cell_clip') #3
        proj_clip = self.options['lstm'].get('proj_clip') #3

        use_skip_connections = self.options['lstm'].get(
                                            'use_skip_connections') #True
        if use_skip_connections:
            print("USING SKIP CONNECTIONS")

        lstm_outputs = []
        
        #lstm_inputs = [self.embedding, self.embedding_reverse]
        for lstm_num, lstm_input in enumerate(lstm_inputs):
            lstm_cells = []
            
            ###########################################################
            # 셀 만들기
            for i in range(n_lstm_layers): #2
                if projection_dim < lstm_dim: #512 < 4096
                    # are projecting down output
                    lstm_cell = tf.nn.rnn_cell.LSTMCell(
                        lstm_dim, #4096
                      	num_proj=projection_dim, #512
                        cell_clip=cell_clip, 
                      	proj_clip=proj_clip)
                else:
                    lstm_cell = tf.nn.rnn_cell.LSTMCell(
                        lstm_dim,
                        cell_clip=cell_clip, 
                      	proj_clip=proj_clip)
								
                # 적용 부분 확인
                if use_skip_connections:
                    # ResidualWrapper adds inputs to outputs
                    if i == 0:
                        # don't add skip connection from token embedding to
                        # 1st layer output
                        pass
                    else:
                        # add a skip connection
                        lstm_cell = tf.nn.rnn_cell.ResidualWrapper(lstm_cell)

                # add dropout
                if self.is_training:
                    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell,
                        input_keep_prob=keep_prob)

                lstm_cells.append(lstm_cell)
	
  					###########################################################
            # LSTM 구조 
      			if n_lstm_layers > 1:
                lstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_cells)
            else:
                lstm_cell = lstm_cells[0]
						
            ###########################################################
            # LSTM 학습
            with tf.control_dependencies([lstm_input]):
                self.init_lstm_state.append(
                    lstm_cell.zero_state(batch_size, DTYPE))
                
                # NOTE: this variable scope is for backward compatibility
                # with existing models...
                if self.bidirectional:
                    with tf.variable_scope('RNN_%s' % lstm_num):
                        _lstm_output_unpacked, final_state = tf.nn.static_rnn(
                            lstm_cell,
                            tf.unstack(lstm_input, axis=1),
                            initial_state=self.init_lstm_state[-1])
                else:
                    _lstm_output_unpacked, final_state = tf.nn.static_rnn(
                        lstm_cell,
                        tf.unstack(lstm_input, axis=1),
                        initial_state=self.init_lstm_state[-1])
                
                self.final_lstm_state.append(final_state)
	
  					###########################################################
        		# LSTM 결과
      			# (batch_size * unroll_steps, 512)
            lstm_output_flat = tf.reshape(
                tf.stack(_lstm_output_unpacked, axis=1), [-1, projection_dim])
            
            if self.is_training:
                # add dropout to output
                lstm_output_flat = tf.nn.dropout(lstm_output_flat,
                    keep_prob)
            
            tf.add_to_collection('lstm_output_embeddings',
                _lstm_output_unpacked)

            lstm_outputs.append(lstm_output_flat)

        self._build_loss(lstm_outputs)

    
    #------------------------------------------------------------------------------#
    def _build_loss(self, lstm_outputs):
        '''
        Create:
            self.total_loss: total loss op for training
            self.softmax_W, softmax_b: the softmax variables
            self.next_token_id / _reverse: placeholders for gold input
        '''
        batch_size = self.options['batch_size'] #128
        unroll_steps = self.options['unroll_steps'] #50

        n_tokens_vocab = self.options['n_tokens_vocab'] #793471

        # DEFINE next_token_id and *_reverse placeholders for the gold input
        def _get_next_token_placeholders(suffix):
            name = 'next_token_id' + suffix
            id_placeholder = tf.placeholder(DTYPE_INT,
                                   shape=(batch_size, unroll_steps),
                                   name=name)
            return id_placeholder

        # get the window and weight placeholders
        self.next_token_id = _get_next_token_placeholders('')
        if self.bidirectional:
            self.next_token_id_reverse = _get_next_token_placeholders(
                '_reverse')

        # DEFINE THE SOFTMAX VARIABLES
        # get the dimension of the softmax weights
        # softmax dimension is the size of the output projection_dim
        softmax_dim = self.options['lstm']['projection_dim']

        # the output softmax variables -- they are shared if bidirectional
        if self.share_embedding_softmax:
            # softmax_W is just the embedding layer
            self.softmax_W = self.embedding_weights

        with tf.variable_scope('softmax'), tf.device('/cpu:0'):
            # Glorit init (std=(1.0 / sqrt(fan_in))
            softmax_init = tf.random_normal_initializer(0.0,
                1.0 / np.sqrt(softmax_dim))
            if not self.share_embedding_softmax:
                self.softmax_W = tf.get_variable(
                    'W', [n_tokens_vocab, softmax_dim],
                    dtype=DTYPE,
                    initializer=softmax_init
                )
            self.softmax_b = tf.get_variable(
                'b', [n_tokens_vocab],
                dtype=DTYPE,
                initializer=tf.constant_initializer(0.0))

        # now calculate losses
        # loss for each direction of the LSTM
        self.individual_losses = []

        if self.bidirectional:
            next_ids = [self.next_token_id, self.next_token_id_reverse]
        else:
            next_ids = [self.next_token_id]

        for id_placeholder, lstm_output_flat in zip(next_ids, lstm_outputs):
            # flatten the LSTM output and next token id gold to shape:
            # (batch_size * unroll_steps, softmax_dim)
            # Flatten and reshape the token_id placeholders
            next_token_id_flat = tf.reshape(id_placeholder, [-1, 1])

            with tf.control_dependencies([lstm_output_flat]):
                if self.is_training and self.sample_softmax:
                    losses = tf.nn.sampled_softmax_loss(
                                   self.softmax_W, self.softmax_b,
                                   next_token_id_flat, lstm_output_flat,
                                   self.options['n_negative_samples_batch'],
                                   self.options['n_tokens_vocab'],
                                   num_true=1)

                else:
                    # get the full softmax loss
                    output_scores = tf.matmul(
                        lstm_output_flat,
                        tf.transpose(self.softmax_W)
                    ) + self.softmax_b
                    # NOTE: tf.nn.sparse_softmax_cross_entropy_with_logits
                    #   expects unnormalized output since it performs the
                    #   softmax internally
                    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=output_scores,
                        labels=tf.squeeze(next_token_id_flat, squeeze_dims=[1])
                    )

            self.individual_losses.append(tf.reduce_mean(losses))

        # backpropagation -> 각 방향 확인
        # now make the total loss -- it's the mean of the individual losses
        if self.bidirectional:
            self.total_loss = 0.5 * (self.individual_losses[0]
                                    + self.individual_losses[1])
        else:
            self.total_loss = self.individual_losses[0]
```



### regularize

