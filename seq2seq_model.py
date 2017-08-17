# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Sequence-to-sequence model with an attention mechanism."""


from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import range  # pylint: disable=redefined-builtin
import tensorflow as tf
import seq2seq
import data_utils
#from config import global_memory,global_output
#global global_memory,global_output
#from tensorflow.models.rnn.translate import data_utils

class Seq2SeqModel(object):
  """Sequence-to-sequence model with attention and for multiple buckets.

  This class implements a multi-layer recurrent neural network as encoder,
  and an attention-based decoder. This is the same as the model described in
  this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
  or into the seq2seq library for complete model implementation.
  This class also allows to use GRU cells in addition to LSTM cells, and
  sampled softmax to handle large output vocabulary size. A single-layer
  version of this model, but with bi-directional encoder, was presented in
    http://arxiv.org/abs/1409.0473
  and sampled softmax is described in Section 3 of the following paper.
    http://arxiv.org/abs/1412.2007
  """

  def __init__(self, vocab_size, embedding_dim, buckets, size,
               num_layers, max_gradient_norm, batch_size, learning_rate,
               learning_rate_decay_factor, use_lstm=False,
               num_samples=512, forward_only=False,train_mode=True,name='Seq2SeqModel'):
    """Create the model.

    Args:
      source_vocab_size: size of the source vocabulary.
      target_vocab_size: size of the target vocabulary.
      buckets: a list of pairs (I, O), where I specifies maximum input length
        that will be processed in that bucket, and O specifies maximum output
        length. Training instances that have inputs longer than I or outputs
        longer than O will be pushed to the next bucket and padded accordingly.
        We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
      size: number of units in each layer of the model.
      num_layers: number of layers in the model.
      max_gradient_norm: gradients will be clipped to maximally this norm.
      batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
      learning_rate: learning rate to start with.
      learning_rate_decay_factor: decay learning rate by this much when needed.
      use_lstm: if true, we use LSTM cells instead of GRU cells.
      num_samples: number of samples for sampled softmax.
      forward_only: if set, we do not construct the backward pass in the model.
    """
    with tf.variable_scope(name) as vs:
      self.vocab_size = vocab_size
      #self.target_vocab_size = target_vocab_size
      #print(type(target_vocab_size))
      self.buckets = buckets
      self.batch_size = batch_size
      self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
      self.learning_rate_decay_op = self.learning_rate.assign(
          self.learning_rate * learning_rate_decay_factor)
      self.global_step = tf.Variable(0, trainable=False)
      # If we use sampled softmax, we need an output projection.
      output_projection = None
      softmax_loss_function = None
      self.embeddings = tf.get_variable(name='embeddings',shape=[self.vocab_size,embedding_dim],initializer=tf.random_uniform_initializer())
      # Sampled softmax only makes sense if we sample less than vocabulary size.
      if num_samples > 0 and num_samples < self.vocab_size:
        w = tf.get_variable("proj_w", [size, self.vocab_size])
        w_t = tf.transpose(w)
        b = tf.get_variable("proj_b", [self.vocab_size])
        output_projection = (w, b)
        # hidden_size = 128
        # output_size = 1


        # def weighted_sampled_loss(labels,inputs):#bug fixed
        #   labels = tf.reshape(labels, [-1, 1])
        #   inputs_ = tf.matmul(inputs,w) + b
        #   with tf.variable_scope('mlp_weight_loss') as vs:
        #
        #     weight = tf.nn.relu(tf.matmul(inputs,w_i)+b_i,name='input_relu')
        #     weight = tf.nn.relu(tf.matmul(weight,w_h)+b_h,name='hidden_relu')
        #     weight = tf.nn.relu(tf.matmul(weight,w_o),name='output_relu')
        #     weight = tf.reshape(weight,shape=[-1])
        #    #labels_ = tf.one_hot(labels,self.target_vocab_size,1,0)
        #   losses_ = tf.nn.sampled_softmax_loss(w_t, b,labels,inputs, num_samples, self.target_vocab_size)
        #   #losses_ = tf.nn.softmax_cross_entropy_with_logits(labels=labels_, logits=inputs_)
        #   #print('losses_shape:',losses_.get_shape())
        #   weight = tf.nn.softmax(losses_)
        #   return tf.multiply(weight,losses_)
        wi_cell = tf.contrib.rnn.GRUCell(10)
        wo_cell = tf.contrib.rnn.GRUCell(10)
        def weight(inputs,outputs):
          q,_ = tf.contrib.rnn.static_rnn(wi_cell,inputs,dtype=tf.float32)
          a,_ = tf.contrib.rnn.static_rnn(wo_cell,outputs,dtype=tf.float32)
          return tf.reduce_mean(q[-1],axis=-1) - tf.reduce_mean(a[-1],axis=-1)
        def sampled_loss(labels,inputs):
          labels = tf.reshape(labels,[-1,1])
          return tf.nn.sampled_softmax_loss(w_t, b,labels,inputs, num_samples,
                   self.vocab_size)
        # if train_mode:
        #   softmax_loss_function = weighted_sampled_loss
        # else:
        softmax_loss_function =sampled_loss

      # Create the internal multi-layer cell for our RNN.
      # The seq2seq function: we use embedding for the input and attention.
      def seq2seq_f(encoder_inputs=None, decoder_inputs=None, do_decode=False):
            return seq2seq.embedding_attention_seq2seq(
              encoder_inputs, decoder_inputs, tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.GRUCell(size) for i in range(num_layers)]),
              # num_encoder_symbols=source_vocab_size,
              # num_decoder_symbols=target_vocab_size,
              # embedding_size=size,
              num_symbols=self.vocab_size,
              embeddings=self.embeddings,
              output_projection=output_projection,
              feed_previous=do_decode)

      # Feeds for inputs.
      self.encoder_inputs = []
      self.decoder_inputs = []
      self.target_weights = []
      for i in range(buckets[-1][0]):  # Last bucket is the biggest one.
        self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                  name="encoder{0}".format(i)))
      for i in range(buckets[-1][1] + 1):
        self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="decoder{0}".format(i)))
        self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                name="weight{0}".format(i)))

      # Our targets are decoder inputs shifted by one.
      targets = [self.decoder_inputs[i + 1]
                 for i in range(len(self.decoder_inputs) - 1)]
      emb_encoder_inputs = [tf.nn.embedding_lookup(self.embeddings,ele) for ele in self.encoder_inputs]
      emb_decoder_inputs = [tf.nn.embedding_lookup(self.embeddings, ele) for ele in self.decoder_inputs]
      # Training outputs and losses.
      if forward_only:
        self.outputs, self.losses = seq2seq.model_with_buckets(
            emb_encoder_inputs,
            emb_decoder_inputs, targets,
            self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),
            softmax_loss_function=softmax_loss_function)
        # If we use output projection, we need to project outputs for decoding.
        if output_projection is not None:
          for b in range(len(buckets)):
            self.outputs[b] = [
                [tf.matmul(output_, output_projection[0]) + output_projection[1] for output_ in output]
                for output in self.outputs[b]]  
      else:
        self.outputs, self.losses = seq2seq.model_with_buckets(
            emb_encoder_inputs,
            emb_decoder_inputs, targets,
            self.target_weights, buckets,
            lambda x, y: seq2seq_f(x, y, False),
            softmax_loss_function=softmax_loss_function)

      # Gradients and SGD update operation for training the model.
      params = []
      for ele in tf.trainable_variables():
        if ele.name.startswith(name):
          params.append(ele)
      if not forward_only:
        self.gradient_norms = []
        self.updates = []
        opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        for b in range(len(buckets)):
          gradients = tf.gradients(self.losses[b], params)
          clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                         max_gradient_norm)
          self.gradient_norms.append(norm)
          self.updates.append(opt.apply_gradients(
              zip(clipped_gradients, params), global_step=self.global_step))
  
      self.saver = tf.train.Saver(params)
      self.variables = []
      for ele in tf.global_variables():
        if ele.name.startswith(name):
          self.variables.append(ele)  
  def step(self, session, encoder_inputs, decoder_inputs, target_weights,
           bucket_id, forward_only,position=None):
    """Run a step of the model feeding the given inputs.

    Args:
      session: tensorflow session to use.
      encoder_inputs: list of numpy int vectors to feed as encoder inputs.
      decoder_inputs: list of numpy int vectors to feed as decoder inputs.
      target_weights: list of numpy float vectors to feed as target weights.
      bucket_id: which bucket of the model to use.
      forward_only: whether to do the backward step or only forward.

    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      average perplexity, and the outputs.

    Raises:
      ValueError: if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """
    # Check if the sizes match.
    #print(tf.global_variables())
    #print(decoder_inputs)
    #print('position:%d'%position)
   
    encoder_size, decoder_size = self.buckets[bucket_id]
    #print(encoder_inputs)
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(target_weights) != decoder_size:
      raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights), decoder_size))

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for l in range(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
      # print(self.encoder_inputs[l].name,encoder_inputs[l])
    for l in range(decoder_size):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]

    # Since our targets are decoder inputs shifted by one, we need one more.
    last_target = self.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only and position is None:
      output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                     self.gradient_norms[bucket_id],  # Gradient norm.
                     self.losses[bucket_id]]  # Loss for this batch.
    elif not forward_only and position is not None:
      output_feed = [tf.nn.softmax(self.outputs[bucket_id][l]) for l in range(decoder_size)]
    else:
      output_feed = [self.losses[bucket_id]]  # Loss for this batch.
      for l in range(decoder_size):  # Output logits.
        output_feed.append(self.outputs[bucket_id][l])
    outputs = session.run(output_feed, input_feed)
    if not forward_only and position is None:
      return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    elif not forward_only and position is not None:
      return None,None,outputs[position]
    else:
      return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.
 
  def get_batch(self, data, bucket_id,train_mode=True):
    """Get a random batch of data from the specified bucket, prepare for step.

    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.

    Args:
      data: a tuple of size len(self.buckets) in which each element contains
        lists of pairs of input and output data that we use to create a batch.
      bucket_id: integer, which bucket to get the batch for.

    Returns:
      The triple (encoder_inputs, decoder_inputs, target_weights) for
      the constructed batch that has the proper format to call step(...) later.
    """
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    if train_mode:
      for _ in range(self.batch_size):

        encoder_input, decoder_input = random.choice(data[bucket_id])

        # Encoder inputs are padded and then reversed.
        encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
        encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

        # Decoder inputs get an extra "GO" symbol, and are padded then.
        decoder_pad_size = decoder_size - len(decoder_input) - 1
        decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                              [data_utils.PAD_ID] * decoder_pad_size)
        #print(decoder_inputs)
    else:
      encoder_inputs, decoder_inputs = [], []
      for ele in data[bucket_id]:
        encoder_input, decoder_input = ele

        # Encoder inputs are padded and then reversed.
        encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
        encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

        # Decoder inputs get an extra "GO" symbol, and are padded then.
        decoder_pad_size = decoder_size - len(decoder_input) - 1
        decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                                  [data_utils.PAD_ID] * decoder_pad_size)
    # Now we create batch-major vectors from the data selected above.


    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in range(encoder_size):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in range(self.batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in range(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in range(self.batch_size)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(self.batch_size, dtype=np.float32)
      for batch_idx in range(self.batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights

  def embeddingBasedEvaluation(predict,target):
    predict = [tf.nn.embedding_lookup(self.embeddings,ele) for ele in predict]
    target = [tf.nn.embedding_lookup(self.embeddings,ele) for ele in target]
    def cosine_similarity(v1,v2):
        #print(v1.shape)
        assert v1.shape[0]==1
        try:
          return np.sum(v1*v2,axis=1)/(np.sqrt(np.sum(v1**2))*np.sqrt(np.sum(v2**2,axis=1)))
        except:
          return ValueError
    Gpt = 0
    emb_t = np.array(target)
    for ele in predict:
      Gpt += np.max(cosine_similarity(ele,emb_t))
    Gtp = 0
    emb_p = np.array(predict)
    for ele in predict:
      Gtp += np.max(cosine_similarity(ele,emb_p))
    GM =  0.5*(Gpt/len(predict)+Gtp/len(target))
    tmp_p = np.sum(emb_p,axis=0)
    tmp_t = np.sum(emb_t,axis=0)
    EV = cosine_similarity(tmp_p/np.sqrt(np.sum(tmp_p**2)),tmp_t/np.sqrt(np.sum(tmp_t**2)))
    #calculate Vector Extrema
    max_ = np.max(emb_p,axis=0)
    min_ = np.min(emb_p,axis=0)
    abs_min_ = np.abs(min_)
    VE_p = np.sign(max_-abs_min_)*np.max([max_,abs_min_],axis=0)
    max_ = np.max(emb_t,axis=0)
    min_ = np.min(emb_t,axis=0)
    abs_min_ = np.abs(min_)
    VE_t = np.sign(max_-abs_min_)*np.max([max_,abs_min_],axis=0)
    VE = cosine_similarity(VE_p,VE_t)
    return GM,EV,VE
'''
if __name__ == '__main__':
    a = [np.random.random([1,128]) for i in range(5)]
    b = [np.random.random([1,128]) for i in range(10)]
    #print(embeddingBasedEvaluation(a,b))
'''
