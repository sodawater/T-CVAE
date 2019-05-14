import tensorflow as tf
import numpy as np
from modules import *
import math
from tensorflow.python.layers import core as layers_core
from data_utils import *
import random
class TCVAE():
    def __init__(self, hparams, mode):
        self.hparams = hparams
        self.vocab_size = hparams.from_vocab_size
        self.num_units = hparams.num_units
        self.emb_dim = hparams.emb_dim
        self.num_layers = hparams.num_layers
        self.num_heads = hparams.num_heads
        self.learning_rate = tf.Variable(float(hparams.learning_rate), trainable=False)
        self.clip_value = hparams.clip_value
        self.max_story_length = 105
        self.max_single_length = 25
        self.latent_dim = hparams.latent_dim
        self.dropout_rate = hparams.dropout_rate
        self.init_weight = hparams.init_weight
        self.flag = True
        self.mode = mode
        self.batch_size = hparams.batch_size
        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.is_training = True
        else:
            self.is_training = False

        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            self.input_ids = tf.placeholder(tf.int32, [None, None])
            self.input_scopes = tf.placeholder(tf.int32, [None, None])
            self.input_positions = tf.placeholder(tf.int32, [None, None])
            self.input_masks = tf.placeholder(tf.int32, [None, None, None])
            self.input_lens = tf.placeholder(tf.int32, [None])
            self.targets = tf.placeholder(tf.int32, [None, None])
            self.weights = tf.placeholder(tf.float32, [None, None])
            self.input_windows = tf.placeholder(tf.float32, [None, 4, None])
            self.which = tf.placeholder(tf.int32, [None])

        else:
            self.input_ids = tf.placeholder(tf.int32, [None, None])
            self.input_scopes = tf.placeholder(tf.int32, [None, None])
            self.input_positions = tf.placeholder(tf.int32, [None, None])
            self.input_masks = tf.placeholder(tf.int32, [None, None, None])
            self.input_lens = tf.placeholder(tf.int32, [None])
            self.input_windows = tf.placeholder(tf.float32, [None, 4, None])
            self.which = tf.placeholder(tf.int32, [None])


        with tf.variable_scope("embedding") as scope:
            self.word_embeddings = tf.Variable(self.init_matrix([self.vocab_size, self.emb_dim]))
            # self.word_embeddings = tf.Variable(hparams.embeddings, trainable=True)
            self.scope_embeddings = tf.Variable(self.init_matrix([9, int(self.emb_dim/2)]))

        with tf.variable_scope("project"):
            self.output_layer = layers_core.Dense(self.vocab_size, use_bias=True)
            self.mid_output_layer = layers_core.Dense(self.vocab_size, use_bias=True)
            self.input_layer = layers_core.Dense(self.num_units, use_bias=False)


        with tf.variable_scope("encoder") as scope:
            self.word_emb = tf.nn.embedding_lookup(self.word_embeddings, self.input_ids)
            self.scope_emb = tf.nn.embedding_lookup(self.scope_embeddings, self.input_scopes)
            self.pos_emb = positional_encoding(self.input_positions, self.batch_size, self.max_single_length, int(self.emb_dim/2))

            # self.embs = self.word_emb + self.scope_emb + self.pos_emb
            self.embs = tf.concat([self.word_emb, self.scope_emb, self.pos_emb], axis=2)
            inputs = self.input_layer(self.embs)


            self.query = tf.get_variable("w_Q", [1, self.num_units], dtype=tf.float32)
            windows = tf.transpose(self.input_windows, [1, 0, 2])
            layers_outputs = []

            post_inputs = inputs
            for i in range(self.num_layers):
                with tf.variable_scope("num_layers_{}".format(i)):
                    outputs = multihead_attention(queries=inputs,
                                                  keys=inputs,
                                                  query_length=self.input_lens,
                                                  key_length=self.input_lens,
                                                  num_units=self.num_units,
                                                  num_heads=self.num_heads,
                                                  dropout_rate=self.dropout_rate,
                                                  is_training=self.is_training,
                                                  using_mask=True,
                                                  mymasks=self.input_masks,
                                                  scope="self_attention")

                    outputs = outputs + inputs
                    inputs = normalize(outputs)

                    outputs = feedforward(inputs, [self.num_units * 2, self.num_units], is_training=self.is_training, dropout_rate=self.dropout_rate, scope="f1")
                    outputs = outputs + inputs
                    inputs = normalize(outputs)

                    post_outputs = multihead_attention(queries=post_inputs,
                                                  keys=post_inputs,
                                                  query_length=self.input_lens,
                                                  key_length=self.input_lens,
                                                  num_units=self.num_units,
                                                  num_heads=self.num_heads,
                                                  dropout_rate=self.dropout_rate,
                                                  is_training=self.is_training,
                                                  using_mask=False,
                                                  mymasks=None,
                                                  scope="self_attention",
                                                  reuse=tf.AUTO_REUSE
                                                 )


                    post_outputs = post_outputs + post_inputs
                    post_inputs = normalize(post_outputs)

                    post_outputs = feedforward(post_inputs, [self.num_units * 2, self.num_units], is_training=self.is_training,
                                          dropout_rate=self.dropout_rate, scope="f1",reuse=tf.AUTO_REUSE)
                    post_outputs = post_outputs + inputs
                    post_inputs = normalize(post_outputs)

            big_window = windows[0] + windows[1] + windows[2] + windows[3]
            post_encode, weight = w_encoder_attention(self.query,
                                                 post_inputs,
                                                 self.input_lens,
                                                 num_units=self.num_units,
                                                 num_heads=self.num_heads,
                                                 dropout_rate=self.dropout_rate,
                                                 is_training=self.is_training,
                                                 using_mask=False,
                                                 mymasks=None,
                                                 scope="concentrate_attention"
                                                 )

            prior_encode, weight = w_encoder_attention(self.query,
                                                      inputs,
                                                      self.input_lens,
                                                      num_units=self.num_units,
                                                      num_heads=self.num_heads,
                                                      dropout_rate=self.dropout_rate,
                                                      is_training=self.is_training,
                                                      using_mask=True,
                                                      mymasks=big_window,
                                                      scope="concentrate_attention",
                                                       reuse=tf.AUTO_REUSE
                                                      )

            post_mulogvar = tf.layers.dense(post_encode, self.latent_dim * 2, use_bias=False, name="post_fc")
            post_mu, post_logvar = tf.split(post_mulogvar, 2, axis=1)

            prior_mulogvar = tf.layers.dense(tf.layers.dense(prior_encode, 256, activation=tf.nn.tanh), self.latent_dim * 2, use_bias=False, name="prior_fc")
            prior_mu, prior_logvar = tf.split(prior_mulogvar, 2, axis=1)


            if self.mode != tf.contrib.learn.ModeKeys.INFER:
                latent_sample = sample_gaussian(post_mu, post_logvar)
            else:
                latent_sample = sample_gaussian(prior_mu, prior_logvar)


            self.latent_sample = latent_sample
            latent_sample = tf.tile(tf.expand_dims(latent_sample, 1), [1, self.max_story_length, 1])
            inputs = tf.concat([inputs, latent_sample], axis=2)
            inputs = tf.layers.dense(inputs, self.num_units, activation=tf.tanh, use_bias=False, name="last")


            self.logits = self.output_layer(inputs)
            self.s = self.logits
            self.sample_id = tf.argmax(self.logits, axis=2)
            # self.sample_id = tf.argmax(self.weight_probs, axis=2)
        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            with tf.variable_scope("loss") as scope:
                self.global_step = tf.Variable(0, trainable=False)
                crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=self.logits)

                self.total_loss = tf.reduce_sum(crossent * self.weights)

                kl_weights = tf.minimum(tf.to_float(self.global_step) / 20000, 1.0)
                kld = gaussian_kld(post_mu, post_logvar, prior_mu, prior_logvar)
                self.loss = tf.reduce_mean(crossent * self.weights) + tf.reduce_mean(kld) * kl_weights




        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            with tf.variable_scope("train_op") as scope:
                optimizer = tf.train.AdamOptimizer(0.0001, beta1=0.9, beta2=0.99, epsilon=1e-9)
                gradients, v = zip(*optimizer.compute_gradients(self.loss))
                gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                self.train_op = optimizer.apply_gradients(zip(gradients, v), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables())


    def get_batch(self, data, no_random=False, id=0, which=0):
        hparams = self.hparams
        input_scopes = []
        input_ids = []
        input_positions = []
        input_lens = []
        input_masks = []
        input_which = []
        input_windows = []
        targets = []
        weights = []
        for i in range(hparams.batch_size):
            if no_random:
                x = data[(id + i) % len(data)]
                which_stn = (id + i) % 5
                # which_stn = which
            else:
                x = random.choice(data)
                which_stn = random.randint(0, 4)

            input_which.append(which_stn)
            mask = []
            input_scope = []
            input_id = []
            input_position = []
            input_mask = []
            target = []
            weight = []
            for j in range(0,5):
                input_id.append(GO_ID)
                input_scope.append(j)
                input_position.append(0)
                for k in range(0, len(x[j])):
                    input_id.append(x[j][k])
                    input_scope.append(j)
                    input_position.append(k + 1)
                    target.append(x[j][k])
                    if j == which_stn:
                        weight.append(1.0)
                        mask.append(0)
                    else:
                        weight.append(0.0)
                        mask.append(1)
                target.append(EOS_ID)
                if j == which_stn:
                    weight.append(1.0)
                    mask.append(0)
                else:
                    weight.append(0.0)
                    mask.append(1)
                input_id.append(EOS_ID)
                input_scope.append(j)
                input_position.append(len(x[j]) + 1)
                target.append(GO_ID)
                if j == which_stn:
                    weight.append(0.0)
                    mask.append(0)
                else:
                    weight.append(0.0)
                    mask.append(1)
                if j == which_stn:
                    for k in range(len(x[j]) + 2, self.max_single_length):
                        input_id.append(PAD_ID)
                        input_scope.append(j)
                        input_position.append(k)
                        target.append(PAD_ID)
                        weight.append(0.0)
                        mask.append(0)
            input_lens.append(len(input_id))
            for k in range(0, self.max_story_length - input_lens[i]):
                input_id.append(PAD_ID)
                input_scope.append(4)
                input_position.append(0)
                target.append(PAD_ID)
                weight.append(0.0)
                mask.append(0)


            input_ids.append(input_id)
            input_scopes.append(input_scope)
            input_positions.append(input_position)
            targets.append(target)
            weights.append(weight)

            tmp_mask = mask.copy()
            last = 0
            window = []

            for k in range(0,5):
                start = last
                if k != 4:
                    last = input_scope.index(k + 1)
                else:
                    last = self.max_story_length
                if k != which_stn:
                    window.append([0] * start + [1] * (last - start) + [0] * (self.max_story_length - last))
            input_windows.append(window)

            for k in range(input_lens[i]):
                if input_scope[k] != which_stn:

                    input_mask.append(mask)
                else:
                    tmp_mask[k] = 1
                    input_mask.append(tmp_mask.copy())

            for k in range(input_lens[i], self.max_story_length):
                input_mask.append(mask)


            input_mask = np.array(input_mask)
            input_masks.append(input_mask)

        return input_ids, input_scopes, input_positions, input_masks, input_lens, input_which, targets, weights, input_windows

    def train_step(self, sess, data):
        input_ids, input_scopes, input_positions, input_masks, input_lens, input_which, targets, weights, input_windows = self.get_batch(data)
        feed = {
            self.input_ids:input_ids,
            self.input_scopes:input_scopes,
            self.input_positions:input_positions,
            self.input_masks:input_masks,
            self.input_lens:input_lens,
            self.weights:weights,
            self.targets:targets,
            self.input_windows:input_windows,
            self.which: input_which
        }
        word_nums = sum(sum(weight) for weight in weights)
        loss, global_step, _, total_loss = sess.run([self.loss, self.global_step, self.train_op, self.total_loss],
                                                        feed_dict=feed)
        return total_loss, global_step, word_nums

    def eval_step(self, sess, data, no_random=False, id=0):
        input_ids, input_scopes, input_positions, input_masks, input_lens, input_which, targets, weights, input_windows = self.get_batch(
            data, no_random, id)
        feed = {
            self.input_ids: input_ids,
            self.input_scopes: input_scopes,
            self.input_positions: input_positions,
            self.input_masks: input_masks,
            self.input_lens: input_lens,
            self.weights: weights,
            self.targets: targets,
            self.input_windows: input_windows,
            self.which: input_which
        }
        loss, logits = sess.run([self.total_loss, self.logits],
                                                feed_dict=feed)
        word_nums = sum(sum(weight) for weight in weights)
        return loss, word_nums

    def infer_step(self, sess, data, no_random=False, id=0, which=0):
        input_ids, input_scopes, input_positions, input_masks, input_lens, input_which, targets, weights, input_windows = self.get_batch(
            data, no_random, id, which=which)
        start_pos = []
        given = []
        ans = []
        predict = []
        hparams = self.hparams
        for i in range(self.hparams.batch_size):
            start_pos.append(input_scopes[i].index(input_which[i]))
            given.append(input_ids[i][:start_pos[i]] + [UNK_ID] * self.max_single_length + input_ids[i][start_pos[i] + self.max_single_length:])
            ans.append(input_ids[i][start_pos[i]: start_pos[i]+self.max_single_length].copy())
            predict.append([])

        for i in range(self.max_single_length - 1):
            feed = {
                self.input_ids: input_ids,
                self.input_scopes: input_scopes,
                self.input_positions: input_positions,
                self.input_masks: input_masks,
                self.input_lens: input_lens,
                self.input_windows: input_windows,
                self.which: input_which
            }
            sample_id = sess.run(self.sample_id, feed_dict=feed)
            for batch in range(self.hparams.batch_size):
                input_ids[batch][start_pos[batch] + i + 1] = sample_id[batch][start_pos[batch] + i]
                predict[batch].append(sample_id[batch][start_pos[batch] + i])
        return given, ans, predict





    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)