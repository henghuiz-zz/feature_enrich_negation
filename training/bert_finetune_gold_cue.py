import sys
import os
import tensorflow as tf
import numpy as np
import pandas as pd
sys.path.append('.')

from bert.modeling import BertConfig
from augmented_bert import BertModel
from bert.optimization import create_optimizer
from sklearn.metrics import f1_score

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('save_model_dir',
                    'ckpt/bert_finetune_gold_cue/speculation_biology_abstract/bilstm_char/cv0/',
                    'path to the directory for saving the model')
flags.DEFINE_string('training_data_path',
                    'data/tfrecords_bert_ft/gold_cue/speculation_biology_abstract/',
                    'path for the training tfrecords for training')

flags.DEFINE_integer('cv', 0, 'which cv split should be used')
flags.DEFINE_integer('seed', 0, 'which random seed to use')
flags.DEFINE_string('model_name', 'uncased_L-24_H-1024_A-16', 'which bert model to use')

flags.DEFINE_float('learning_rate', 1e-6, 'learning rate of the optimizer')
flags.DEFINE_float('warmup_proportion', 0.1, 'warm up portion of the optimizer')
flags.DEFINE_integer('num_train_epochs', 20, 'training epoch')
flags.DEFINE_integer('batch_size', 8, 'batch size')

flags.DEFINE_bool('training', True, 'training or testing model')


MODEL_NAME = 'uncased_L-24_H-1024_A-16'
PRETRAIN_MODEL_PATH = '/home/henghuiz/HugeData/word_vector/bert/' + MODEL_NAME
CONFIG_FILE = os.path.join(PRETRAIN_MODEL_PATH, 'bert_config.json')

train_tokens = int(1200 * 9 / FLAGS.batch_size)
num_train_step = int(train_tokens * FLAGS.num_train_epochs / FLAGS.batch_size)
num_warmup_steps = int(num_train_step * FLAGS.warmup_proportion)


def negation_model(x, y, pos, dep, path, lpath, cp, c, m, l, train=True, reuse=False):
  y_ = tf.cast(y, tf.float32)
  m_ = tf.cast(m, tf.float32)
  segment_ids = tf.zeros_like(x)

  seq_len = tf.shape(x)[1]
  l = tf.cast(l, tf.int32)

  input_mask = tf.cast(tf.sequence_mask(l, maxlen=seq_len), tf.float32)
  input_mask = tf.stop_gradient(input_mask)

  config = BertConfig.from_json_file(CONFIG_FILE)
  hidden_size = config.hidden_size

  with tf.variable_scope('', reuse=reuse):
    with tf.variable_scope('preprocessing'):
      pos_embeddings = tf.get_variable('pos_embedding', shape=[100, hidden_size],
                                       initializer=tf.glorot_normal_initializer())
      pos_feature = tf.nn.embedding_lookup(pos_embeddings, pos)

      dep_embeddings = tf.get_variable('dep_embedding', shape=[100, hidden_size],
                                       initializer=tf.glorot_normal_initializer())
      dep_feature = tf.nn.embedding_lookup(dep_embeddings, dep)

      path_embeddings = tf.get_variable('path_embedding', shape=[2000, hidden_size],
                                        initializer=tf.glorot_normal_initializer())
      path_feature = tf.nn.embedding_lookup(path_embeddings, path)

      lpath = tf.expand_dims(tf.cast(lpath, tf.float32), -1)
      lpath_feature = tf.layers.dense(lpath, units=hidden_size, activation=tf.nn.relu, name='lpath')

      cp = tf.expand_dims(tf.cast(cp, tf.float32), -1)
      cp_feature = tf.layers.dense(cp, units=hidden_size, activation=tf.nn.relu, name='cp')

      cue = tf.expand_dims(tf.cast(c, tf.float32), -1)
      cue_feature = tf.layers.dense(cue, units=hidden_size, name='cue')

      other_features = (pos_feature + cp_feature + cue_feature + dep_feature + path_feature + lpath_feature) / 6

    model = BertModel(
      config=config,
      is_training=train,
      input_ids=x,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=True,
      scope='bert',
      other_features=other_features
    )

    output_layer = model.get_sequence_output()
    logits = tf.layers.dense(output_layer, units=1, activation=None, name='fc')
    logits = tf.squeeze(logits, -1)
    prediction = tf.cast(logits > 0, tf.int32)

    losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=logits)
    loss = tf.reduce_mean(losses * m_)

    if train:
      train_step = create_optimizer(loss, FLAGS.learning_rate, num_train_step, num_warmup_steps, False)
      return loss, prediction, train_step
    else:
      return loss, prediction,


def _parse_function(example_proto):
  contexts, features = tf.parse_single_sequence_example(
    example_proto,
    context_features={
      "length": tf.FixedLenFeature([], dtype=tf.int64),
    },
    sequence_features={
      "token_id": tf.FixedLenSequenceFeature([], dtype=tf.int64),
      'masks': tf.FixedLenSequenceFeature([], dtype=tf.int64),
      'span': tf.FixedLenSequenceFeature([], dtype=tf.int64),
      'cue': tf.FixedLenSequenceFeature([], dtype=tf.int64),
      "pos": tf.FixedLenSequenceFeature([], dtype=tf.int64),
      "dep": tf.FixedLenSequenceFeature([], dtype=tf.int64),
      "path": tf.FixedLenSequenceFeature([], dtype=tf.int64),
      "lpath": tf.FixedLenSequenceFeature([], dtype=tf.int64),
      "cp": tf.FixedLenSequenceFeature([], dtype=tf.int64),
    }
  )

  return features["token_id"], features["span"], \
         features["pos"], features["dep"], features["path"], \
         features["lpath"], features["cp"], features["cue"], \
         features["masks"], contexts["length"],


def build_iter_ops(filenames, train, reuse):
  batch_size = FLAGS.batch_size if train else 2

  dataset = tf.data.TFRecordDataset(filenames)
  dataset = dataset.map(_parse_function)
  if train:
    dataset = dataset.shuffle(buffer_size=batch_size)
  dataset = dataset.padded_batch(batch_size,
                                 ([tf.Dimension(None)],
                                  [tf.Dimension(None)],
                                  [tf.Dimension(None)],
                                  [tf.Dimension(None)],
                                  [tf.Dimension(None)],
                                  [tf.Dimension(None)],
                                  [tf.Dimension(None)],
                                  [tf.Dimension(None)],
                                  [tf.Dimension(None)],
                                  []))
  iterator = dataset.make_initializable_iterator()

  x, y, pos, dep, path, lpath, cp, c, m, l = iterator.get_next()
  ops = negation_model(x, y, pos, dep, path, lpath, cp, c, m, l, train=train, reuse=reuse)

  ops = list(ops)
  ops.append(m)
  ops.append(y)

  return iterator, ops


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


def run_a_epoch_train(sess, iterator, ops):
  sess.run(iterator.initializer)
  epoch_pred = []
  epoch_y = []
  epoch_mask = []

  epoch_loss = []
  while True:
    try:
      op_result = sess.run(ops)
      epoch_loss.append(op_result[0])
      epoch_pred += list(op_result[1])

      epoch_mask += list(op_result[-2])
      epoch_y += list(op_result[-1])
      # print('\r Number examples: %d, Average Losses: %.4f' % (len(epoch_loss), np.mean(epoch_loss)), end='', flush=True)

    except tf.errors.OutOfRangeError:
      break

  f1 = find_f1(epoch_pred, epoch_y, epoch_mask)
  pcs = find_pcs(epoch_pred, epoch_y, epoch_mask)

  return np.mean(epoch_loss), f1, pcs


def train():
  all_files = os.listdir(FLAGS.training_data_path)
  all_files = [item for item in all_files if item.endswith('short.tfr')]

  train_files = [FLAGS.training_data_path + item for item in all_files if item[8] != str(FLAGS.cv)]
  valid_files = [FLAGS.training_data_path + item for item in all_files if item[8] == str(FLAGS.cv)]

  print(train_files)
  print(valid_files)

  with tf.Graph().as_default():
    train_iter, train_op = build_iter_ops(train_files, train=True, reuse=False)
    valid_iter, valid_op = build_iter_ops(valid_files, train=False, reuse=True)

    bert_variables = tf.trainable_variables('bert')
    bert_loader = tf.train.Saver(bert_variables)

    useful_variables = [item for item in tf.global_variables() if 'adam_m' not in item.name]

    saver = tf.train.Saver(useful_variables)

    os.makedirs(FLAGS.save_model_dir, exist_ok=True)

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      bert_loader.restore(sess, os.path.join(PRETRAIN_MODEL_PATH , 'bert_model.ckpt'))

      table = []

      for epoch in range(FLAGS.num_train_epochs):
        train_loss, train_f1, train_pcs = run_a_epoch_train(sess, train_iter, train_op)
        print('\r Train', epoch, train_loss, train_f1, train_pcs)

        valid_loss, valid_f1, valid_pcs = run_a_epoch_train(sess, valid_iter, valid_op)
        print('\r Valid', epoch, valid_loss, valid_f1, valid_pcs)

        saver.save(sess, FLAGS.save_model_dir + 'last_model', write_meta_graph=False, write_state=False)

        table.append([epoch, train_loss, train_f1, train_pcs, valid_loss, valid_f1, valid_pcs])

      table = pd.DataFrame(table)
      table.columns = ['epoch', 'loss:train', 'f1:train', 'pcs:train', 'loss:test', 'f1:test', 'pcs:test']
      table.to_csv(FLAGS.save_model_dir + 'precess.csv', index=False)


def eval():
  all_files = os.listdir(FLAGS.training_data_path)
  all_files = [item for item in all_files if item.endswith('long.tfr')]
  valid_files = [FLAGS.training_data_path + item for item in all_files if item[8] == str(FLAGS.cv)]

  with tf.Graph().as_default():
    tf.set_random_seed(FLAGS.seed)
    valid_iter, valid_op = build_iter_ops(valid_files, train=False, reuse=False)

    saver = tf.train.Saver()

    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      saver.restore(sess, FLAGS.save_model_dir + 'last_model')

      valid_loss, valid_f1, valid_pcs = run_a_epoch_train(sess, valid_iter, valid_op)
      print('\r Valid', valid_loss, valid_f1, valid_pcs)


def main(_):
  if FLAGS.training:
    train()
  else:
    eval()


if __name__ == '__main__':
  tf.app.run()
