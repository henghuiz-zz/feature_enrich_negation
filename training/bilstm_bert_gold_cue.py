import tensorflow as tf
import numpy as np
import os
import time
import pandas as pd
from sklearn.metrics import f1_score

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('save_model_dir',
                    'ckpt/bert_gold_cue/speculation_biology_abstract/bilstm_char/cv0/',
                    'path to the directory for saving the model')
flags.DEFINE_string('eval_model_dir',
                    'ckpt/bert_gold_cue/speculation_biology_abstract/bilstm_char/',
                    'path to the directory for evaluation')
flags.DEFINE_string('training_data_path',
                    'data/tfrecords_bert/gold_cue/speculation_biology_abstract/',
                    'path for the training tfrecords for training')

flags.DEFINE_bool('use_context', True, 'to use full bert model or just its subword embedding')
flags.DEFINE_bool('use_pos', False, 'to use pos features or not')
flags.DEFINE_bool('use_dep', False, 'to use dep features or not')
flags.DEFINE_bool('use_path', False, 'to use path features or not')
flags.DEFINE_bool('use_lpath', False, 'to use lpath features or not')
flags.DEFINE_bool('use_cp', False, 'to use constituency parsing features or not')

flags.DEFINE_float('dropout_rate', 0.5, 'the dropout rate of the CNN or RNN')
flags.DEFINE_integer('hidden_state', 400, 'Number of hidden state')
flags.DEFINE_integer('depth', 2, 'Depth of rnn models')

flags.DEFINE_integer('batch_size', 32, 'Batch size of the model')
flags.DEFINE_integer('num_epochs', 200, 'Number of the epochs in training')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate of the optimizer')
flags.DEFINE_integer('cv', 0, 'which cv split should be used')
flags.DEFINE_integer('seed', 0, 'which random seed to use')

flags.DEFINE_bool('training', True, 'training or testing model')


def bidirectional_rnn_func(x, l, train=True):
  max_len = tf.shape(x)[1]
  m_d = tf.sequence_mask(l, max_len, dtype=tf.float32)
  m_d = tf.expand_dims(m_d, -1)

  rnn_func = tf.contrib.cudnn_rnn.CudnnLSTM

  dr = FLAGS.dropout_rate if train else 0

  all_fw_cells = rnn_func(num_layers=FLAGS.depth,
                          num_units=FLAGS.hidden_state,
                          direction='bidirectional',
                          dropout=dr)

  x = tf.transpose(x, [1, 0, 2])
  if train:
    x = tf.layers.dropout(x, rate=0.2)

  rnn_state, _ = all_fw_cells(x)

  rnn_state = tf.transpose(rnn_state, [1, 0, 2])
  output = tf.concat(rnn_state, axis=-1) * m_d

  return output


def annotation_func_train(x,
                          y,
                          l,
                          pos,
                          dep,
                          path,
                          lpath,
                          cp,
                          token_id,
                          cue,
                          mask,
                          train,
                          reuse,
                          scope='vanilla_rnn'):
  with tf.variable_scope(scope, reuse=reuse):
    seq_len = tf.shape(x)[1]
    l = tf.cast(l, tf.int32)

    mask = tf.cast(mask, tf.float32)
    # project to some low dimension

    token_embedding = x

    gc = tf.one_hot(cue, depth=2)

    input_feature = [token_embedding, gc]

    if FLAGS.use_pos:
      pos_embeddings = tf.get_variable('pos_embedding',
                                       shape=[100, 10],
                                       initializer=tf.glorot_uniform_initializer())
      pos_feature = tf.nn.embedding_lookup(pos_embeddings, pos)
      input_feature.append(pos_feature)

    if FLAGS.use_dep:
      dep_embeddings = tf.get_variable('dep_embedding',
                                       shape=[100, 10],
                                       initializer=tf.glorot_uniform_initializer())
      dep_feature = tf.nn.embedding_lookup(dep_embeddings, dep)
      input_feature.append(dep_feature)

    if FLAGS.use_path:
      path_embeddings = tf.get_variable('path_embedding',
                                        shape=[2000, 10],
                                        initializer=tf.glorot_uniform_initializer())
      path_feature = tf.nn.embedding_lookup(path_embeddings, path)
      input_feature.append(path_feature)

    if FLAGS.use_lpath:
      lpath = tf.expand_dims(tf.cast(lpath, tf.float32), -1)
      lpath_feature = tf.layers.dense(lpath, units=10, activation=None)
      input_feature.append(lpath_feature)

    if FLAGS.use_cp:
      cp = tf.expand_dims(tf.cast(cp, tf.float32), -1)
      cp_feature = tf.layers.dense(cp, units=10, activation=None)
      input_feature.append(cp_feature)

    final_embedding = tf.concat(input_feature, axis=-1)

    # find logit
    first_pass = bidirectional_rnn_func(final_embedding, l, train)
    logits = tf.layers.dense(first_pass, 1, activation=None)
    logits = tf.squeeze(logits, -1)

    step_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=y, logits=logits, weights=mask)

    prediction = tf.cast(logits > 0, tf.int32)

    if train:
      train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(step_loss)
      return step_loss, train_step, prediction
    else:
      return step_loss, prediction


def _parse_function(example_proto):
  contexts, features = tf.parse_single_sequence_example(
      example_proto,
      context_features={
          "length": tf.FixedLenFeature([], dtype=tf.int64),
      },
      sequence_features={
          "token": tf.FixedLenSequenceFeature([1024], dtype=tf.float32),
          "embedding": tf.FixedLenSequenceFeature([1024], dtype=tf.float32),
          "span": tf.FixedLenSequenceFeature([], dtype=tf.int64),
          "cue": tf.FixedLenSequenceFeature([], dtype=tf.int64),
          "pos": tf.FixedLenSequenceFeature([], dtype=tf.int64),
          "dep": tf.FixedLenSequenceFeature([], dtype=tf.int64),
          "path": tf.FixedLenSequenceFeature([], dtype=tf.int64),
          "lpath": tf.FixedLenSequenceFeature([], dtype=tf.int64),
          "cp": tf.FixedLenSequenceFeature([], dtype=tf.int64),
          "token_id": tf.FixedLenSequenceFeature([], dtype=tf.int64),
          "masks": tf.FixedLenSequenceFeature([], dtype=tf.int64),
      })

  if FLAGS.use_context:
    token = features["token"]
  else:
    token = features["embedding"]

  return token, features["span"], contexts["length"], \
         features["pos"], features["dep"], features["path"], \
         features["lpath"], features["cp"], features["token_id"], features["cue"], features["masks"]


def generate_iterator_ops(filenames, train=True, reuse=False):
  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.map(_parse_function, num_parallel_calls=8)

  if train:
    dataset = dataset.shuffle(buffer_size=2 * FLAGS.batch_size)

  dataset = dataset.padded_batch(
      FLAGS.batch_size,
      ([tf.Dimension(None), tf.Dimension(1024)], [tf.Dimension(None)], [], [tf.Dimension(None)],
       [tf.Dimension(None)], [tf.Dimension(None)], [tf.Dimension(None)], [tf.Dimension(None)],
       [tf.Dimension(None)], [tf.Dimension(None)], [tf.Dimension(None)]))
  data_iterator = dataset.make_initializable_iterator()
  # grab a batch of data
  next_x, next_y, next_l, next_pos, next_dep, next_path, next_lp, next_cp, next_id, next_c, next_mask = data_iterator.get_next(
  )

  ops = annotation_func_train(next_x,
                              next_y,
                              next_l,
                              next_pos,
                              next_dep,
                              next_path,
                              next_lp,
                              next_cp,
                              next_id,
                              next_c,
                              next_mask,
                              train=train,
                              reuse=reuse)

  ops = list(ops)
  ops.append(next_y)
  ops.append(next_mask)

  return data_iterator, ops


def find_f1(y_pred_list, y_list, mask_list):
  all_y = []
  all_y_pred = []

  for y_pred, y, mask in zip(y_pred_list, y_list, mask_list):
    for yp, ygt, m in zip(y_pred, y, mask):
      if m > 0:
        all_y.append(yp)
        all_y_pred.append(ygt)

  f1 = 0
  try:
    f1 = f1_score(all_y, all_y_pred)
  except:
    pass

  return f1


def find_pcs(y_pred_list, y_list, mask_list):
  accuracy = []

  for y_pred, y, mask in zip(y_pred_list, y_list, mask_list):
    y_ = []
    yp_ = []
    for yp, ygt, m in zip(y_pred, y, mask):
      if m > 0:
        y_.append(yp)
        yp_.append(ygt)

    y_ = np.array(y_)
    yp_ = np.array(yp_)
    correct = np.sum(y_ != yp_) == 0
    accuracy.append(correct)

  return np.mean(accuracy)


def run_one_epoch_train(sess, iterator, ops):
  """Proceed a epoch of training/validation"""
  all_loss = []
  all_y_pred = []
  all_y = []
  all_l = []
  sess.run(iterator.initializer)

  while True:
    try:
      results = sess.run(ops)
      all_loss.append(results[0])

      all_y_pred += list(results[-3])
      all_y += list(results[-2])
      all_l += list(results[-1])
    except tf.errors.OutOfRangeError:
      break

  f1 = find_f1(all_y_pred, all_y, all_l)
  pcs = find_pcs(all_y_pred, all_y, all_l)

  return np.mean(all_loss), f1, pcs


def train():
  all_files = os.listdir(FLAGS.training_data_path)
  all_files = [item for item in all_files if item[-3:] == 'tfr']

  train_files = [FLAGS.training_data_path + item for item in all_files if item[8] != str(FLAGS.cv)]
  valid_files = [FLAGS.training_data_path + item for item in all_files if item[8] == str(FLAGS.cv)]

  with tf.Graph().as_default():
    tf.set_random_seed(FLAGS.seed)
    train_iter, train_op = generate_iterator_ops(train_files, train=True, reuse=False)
    valid_iter, valid_op = generate_iterator_ops(valid_files, train=False, reuse=True)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    save_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) + \
                tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS)

    save_vars = [item for item in save_vars if 'Adam' not in item.name]

    saver = tf.train.Saver(save_vars)

    if not tf.gfile.IsDirectory(FLAGS.save_model_dir):
      tf.gfile.MakeDirs(FLAGS.save_model_dir)

    table = []

    with tf.Session(config=config) as sess:
      max_f1 = 0
      sess.run(tf.global_variables_initializer())
      for epoch in range(FLAGS.num_epochs):
        tic = time.time()
        train_loss, train_f1, train_pcs = run_one_epoch_train(sess, train_iter, train_op)
        valid_loss, valid_f1, valid_pcs = run_one_epoch_train(sess, valid_iter, valid_op)
        toc = time.time() - tic

        print(
            "Epoch %d: train loss %.4f, F1 %.4f, PCS: %.4f, valid loss %.4f, F1 %.4f, PCS: %.4f, elapsed time %.1f s"
            % (epoch, train_loss, train_f1, train_pcs, valid_loss, valid_f1, valid_pcs, toc))

        table.append([epoch, train_loss, train_f1, train_pcs, valid_loss, valid_f1, valid_pcs])

        if valid_f1 > max_f1:
          max_f1 = valid_f1
          saver.save(sess,
                     FLAGS.save_model_dir + 'best_model',
                     write_state=False,
                     write_meta_graph=False)

        saver.save(sess,
                   FLAGS.save_model_dir + 'final_model',
                   write_state=False,
                   write_meta_graph=False)

      table = pd.DataFrame(table)
      table.columns = [
          'epoch', 'loss:train', 'f1:train', 'pcs:train', 'loss:test', 'f1:test', 'pcs:test'
      ]
      table.to_csv(FLAGS.save_model_dir + 'precess.csv', index=False)


def evaluation():
  all_files = os.listdir(FLAGS.training_data_path)
  valid_files = [FLAGS.training_data_path + item for item in all_files if item[8] == str(FLAGS.cv)]

  with tf.Graph().as_default():
    valid_iter, valid_op = generate_iterator_ops(valid_files, train=False, reuse=False)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
      sess.run(tf.global_variables_initializer())
      saver.restore(sess, FLAGS.save_model_dir + 'final_model')

      valid_loss, valid_f1, valid_pcs = run_one_epoch_train(sess, valid_iter, valid_op)

      print(valid_f1, valid_pcs)


def main(_):
  if FLAGS.training:
    train()
  else:
    evaluation()


if __name__ == '__main__':
  tf.app.run()
