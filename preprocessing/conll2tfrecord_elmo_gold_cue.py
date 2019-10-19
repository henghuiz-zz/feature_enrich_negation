import os
import numpy as np
import tensorflow as tf
from collections import Counter
import tensorflow_hub as hub
import json

DATA_PATH = 'data/'
DATASET_NAME = 'biology_abstract' # clinical_reports or biology_abstract
TASK = 'speculation' # speculation or negation

GLOVE_PATH = '/home/henghuiz/HugeData/word_vector/glove.840B.300d.vec'


class ELMO:
  def __init__(self):
    elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)
    self.tokens = tf.placeholder(dtype=tf.string, shape=[None, None], name='input_tokens')
    self.l = tf.placeholder(dtype=tf.int32, shape=[None], name='length')

    token_embedding = elmo(inputs={"tokens": self.tokens, "sequence_len": self.l}, signature="tokens", as_dict=True)

    self.we = token_embedding['word_emb']
    self.lm1 = token_embedding['lstm_outputs1']
    self.lm2 = token_embedding['lstm_outputs2']

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    self.session = tf.Session(config=config)
    self.session.run(tf.global_variables_initializer())
    self.session.run(tf.tables_initializer())

  def get_embeddings(self, x, l):
    we, lm1, lm2 = self.session.run([self.we, self.lm1, self.lm2], feed_dict={self.tokens: x, self.l: l})
    return we, lm1, lm2


def parse_text(text):
  sentences = text.split('\n\n')

  all_pos = Counter()
  all_dep = Counter()
  all_path = Counter()
  all_lpath = Counter()
  all_vocab = Counter()
  all_cp = Counter()

  max_length = 0

  # parse one sentences
  for sentence in sentences:
    token_sequence = []

    for token in sentence.split('\n'):
      if len(token) >= 8:
        token = token.split('\t')
        token_sequence.append(token)
    all_vocab.update([item[0].lower() for item in token_sequence])
    all_pos.update([item[2] for item in token_sequence])
    all_dep.update([item[3] for item in token_sequence])
    all_path.update([item[4] for item in token_sequence])
    all_lpath.update([item[5] for item in token_sequence])
    all_cp.update([item[6] for item in token_sequence])

    max_length = max(max_length, len(token_sequence))

  return all_pos, all_dep, all_path, all_lpath, all_cp, max_length, all_vocab


def write_tf_records(text, output_filename, all_we, all_lm1, all_lm2, pos2id, sem2id, root2id, token2id):
  writer = tf.python_io.TFRecordWriter(output_filename)
  sentences = text.split('\n\n')
  for sent_id, sentence in enumerate(sentences):
    token_sequence = []
    token_ids = []

    for token in sentence.split('\n'):
      if len(token) >= 8:
        token = token.split('\t')
        token_sequence.append(token)

    tokens = [item[0] for item in token_sequence]
    for token in tokens:
      if token in token2id.keys():
        token_ids.append(token2id[token])
      else:
        token_ids.append(0)

    cues = [int(item[7]) for item in token_sequence]
    pos = [pos2id[item[2]] for item in token_sequence]
    dep = [sem2id[item[3]] for item in token_sequence]
    path = [root2id[item[4]] for item in token_sequence]
    lpath = [int(item[5]) for item in token_sequence]
    cp = [int(item[6]) for item in token_sequence]
    span = [int(item[8]) for item in token_sequence]

    if len(tokens) > 0:
      we = all_we[sent_id]
      lm1 = all_lm1[sent_id]
      lm2 = all_lm2[sent_id]
      l = len(tokens)

      embeddings = np.zeros([l, 1024, 3])
      embeddings[:, :512, 0] = we[:l, :]
      embeddings[:, 512:, 0] = we[:l, :]
      embeddings[:, :, 1] = lm1[:l, :]
      embeddings[:, :, 2] = lm2[:l, :]

      context = tf.train.Features(feature={  # Non-serial data uses Feature
        "length": tf.train.Feature(int64_list=tf.train.Int64List(value=[len(tokens)])),
      })

      token_features = [tf.train.Feature(float_list=tf.train.FloatList(value=embedding.reshape(-1))) for embedding
                        in
                        embeddings]
      cue_features = [tf.train.Feature(int64_list=tf.train.Int64List(value=[cue])) for cue in cues]
      pos_features = [tf.train.Feature(int64_list=tf.train.Int64List(value=[pos_])) for pos_ in pos]
      dep_features = [tf.train.Feature(int64_list=tf.train.Int64List(value=[dep_])) for dep_ in dep]
      path_features = [tf.train.Feature(int64_list=tf.train.Int64List(value=[path_])) for path_ in path]
      lpath_features = [tf.train.Feature(int64_list=tf.train.Int64List(value=[lpath_])) for lpath_ in lpath]
      cp_features = [tf.train.Feature(int64_list=tf.train.Int64List(value=[cp_])) for cp_ in cp]
      span_features = [tf.train.Feature(int64_list=tf.train.Int64List(value=[span_])) for span_ in span]
      token_id_features = [tf.train.Feature(int64_list=tf.train.Int64List(value=[id_])) for id_ in token_ids]

      feature_list = {
        'token': tf.train.FeatureList(feature=token_features),
        'token_id': tf.train.FeatureList(feature=token_id_features),
        'cue': tf.train.FeatureList(feature=cue_features),
        'pos': tf.train.FeatureList(feature=pos_features),
        'dep': tf.train.FeatureList(feature=dep_features),
        'path': tf.train.FeatureList(feature=path_features),
        'lpath': tf.train.FeatureList(feature=lpath_features),
        'cp': tf.train.FeatureList(feature=cp_features),
        'span': tf.train.FeatureList(feature=span_features),
      }

      feature_lists = tf.train.FeatureLists(feature_list=feature_list)
      ex = tf.train.SequenceExample(feature_lists=feature_lists, context=context)
      writer.write(ex.SerializeToString())

  writer.close()


def find_text_list(text, max_len):
  sentences = text.split('\n\n')

  text_list = []
  length_list = []

  for sentence in sentences:
    token_sequence = []

    for token in sentence.split('\n'):
      if len(token) >= 8:
        token = token.split('\t')
        token_sequence.append(token[0])
    if len(token_sequence) > 0:
      length_list.append(len(token_sequence))
      for _ in range(max_len - len(token_sequence)):
        token_sequence.append('')

      text_list.append(token_sequence)

  return text_list, length_list


def data_iter(x, l):
  current_id = 0
  num_instance = len(x)

  while current_id < num_instance:
    yield x[current_id:current_id + 10], l[current_id:current_id + 10]
    current_id += 10


def find_embeddings(text_list, length_list, elmo):
  all_we = []
  all_lm1 = []
  all_lm2 = []

  for x, l in data_iter(text_list, length_list):
    we, lm1, lm2 = elmo.get_embeddings(x, l)
    all_we += list(we)
    all_lm1 += list(lm1)
    all_lm2 += list(lm2)

  all_we = np.array(all_we)
  all_lm1 = np.array(all_lm1)
  all_lm2 = np.array(all_lm2)

  return all_we, all_lm1, all_lm2


def main():
  data_path = DATA_PATH + 'conll/gold_cue/'+TASK+'_'+DATASET_NAME+'/'
  output_path = DATA_PATH + 'tfrecords_elmo/gold_cue/'+TASK+'_'+DATASET_NAME+'/'

  if not os.path.isdir(output_path):
    os.makedirs(output_path)

  filenames = [item for item in os.listdir(data_path)]

  all_pos = Counter()
  all_dep = Counter()
  all_path = Counter()
  all_vocab = Counter()
  all_depth = Counter()
  all_cp = Counter()
  max_len = 0

  for filename in filenames:
    full_path = data_path + filename
    text = open(full_path, 'r').read()
    pos, sem, root, depth, cp, max_len_ins, vocab_ins = parse_text(text)
    all_pos += pos
    all_dep += sem
    all_path += root
    all_depth += depth
    all_vocab += vocab_ins
    all_cp += cp

    max_len = max(max_len, max_len_ins)

  all_vocab = all_vocab.most_common(len(all_vocab))
  valid_vocabulary = [item[0] for item in all_vocab if item[1] >= 5]
  token2id = {token: token_id + 1 for token_id, token in enumerate(valid_vocabulary)}

  all_pos = [item[0] for item in all_pos.most_common(len(all_pos))]
  all_dep = [item[0] for item in all_dep.most_common(len(all_dep))]
  all_path = [item[0] for item in all_path.most_common(len(all_path))]

  pos2id = {item: key for key, item in enumerate(all_pos)}
  dep2id = {item: key for key, item in enumerate(all_dep)}
  path2id = {item: key for key, item in enumerate(all_path)}

  print(len(all_pos), len(all_dep), len(all_path), max_len)

  elmo = ELMO()

  json.dump({'pos':pos2id, 'dep': dep2id, 'path': path2id}, open(output_path + '/meta.json', 'w'), indent=True)

  for filename in filenames:
    full_path = data_path + filename
    text = open(full_path, 'r').read()

    x, l = find_text_list(text, max_len)
    we, lm1, lm2 = find_embeddings(x, l, elmo)
    output_file = output_path + filename[:-5] + 'tfr'
    print(output_file)
    write_tf_records(text, output_file, we, lm1, lm2, pos2id, dep2id, path2id, token2id)


if __name__ == '__main__':
  main()
