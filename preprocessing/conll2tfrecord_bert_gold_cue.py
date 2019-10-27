import os
import sys
import json
import argparse
import numpy as np
import tensorflow as tf
from collections import Counter
from bert.modeling import BertConfig, BertModel
from bert.tokenization import FullTokenizer

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data_path',
                    type=str,
                    default='data/',
                    help='path for the data folder')
parser.add_argument('--dataset_name',
                    type=str,
                    default='biology_abstract',
                    help='clinical_reports or biology_abstract')
parser.add_argument('--task',
                    type=str,
                    default='speculation',
                    help='speculation or negation')
parser.add_argument('--pretrain_models_path',
                    type=str,
                    default='/home/henghuiz/word_vector/bert/',
                    help='place you put your pretrained bert model')
parser.add_argument('--bert_model_name',
                    type=str,
                    default='uncased_L-24_H-1024_A-16',
                    help='name of the pretrained BERT model')
args = parser.parse_args()


class BERTModel:
  def __init__(self):
    bert_pretrained_dir = args.pretrain_models_path + args.bert_model_name
    self.do_lower_case = args.bert_model_name.startswith('uncased')
    self.vocab_file = os.path.join(bert_pretrained_dir, 'vocab.txt')
    self.config_file = os.path.join(bert_pretrained_dir, 'bert_config.json')
    self.tokenizer = FullTokenizer(vocab_file=self.vocab_file,
                                   do_lower_case=self.do_lower_case)

    self.input_id = tf.placeholder(tf.int64, [None, None], 'input_ids')
    self.input_mask = tf.placeholder(tf.int64, [None, None], 'input_mask')
    self.segment_ids = tf.placeholder(tf.int64, [None, None], 'segment_ids')

    bert_config = BertConfig.from_json_file(self.config_file)
    model = BertModel(config=bert_config,
                      is_training=False,
                      input_ids=self.input_id,
                      input_mask=self.input_mask,
                      token_type_ids=self.segment_ids,
                      use_one_hot_embeddings=True,
                      scope='bert')
    self.output_layer = model.get_sequence_output()
    self.embedding_layer = model.get_embedding_output()

    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self.session = tf.Session(config=config)
    saver.restore(self.session, bert_pretrained_dir + '/bert_model.ckpt')

  def tokenize(self, token_list, attributes_list):
    num_attributes = len(attributes_list)
    output_list = [[] for _ in range(num_attributes)]
    token_ids = []
    masks = []

    token_ids.append("[CLS]")
    for token_id, token in enumerate(token_list):
      new_tokens = self.tokenizer.tokenize(token)
      token_ids.extend(new_tokens)

      for att_id in range(num_attributes):
        l_ = [
            attributes_list[att_id][token_id] for _ in range(len(new_tokens))
        ]
        output_list[att_id].extend(l_)

      m = [0 for _ in range(len(new_tokens))]
      m[0] = 1
      masks.extend(m)

    token_ids.append("[SEP]")

    token_ids = self.tokenizer.convert_tokens_to_ids(token_ids)
    last_layer, embedding = self.get_embeddings(token_ids)

    if len(last_layer) != len(output_list[0]):
      print(token_list)
      print(token_ids)
      for list_i in output_list:
        print(list_i)

    assert len(last_layer) == len(output_list[0])

    return last_layer, embedding, token_ids[1:-1], output_list, masks

  def get_embeddings(self, token_ids):
    input_mask = [[1] * len(token_ids)]
    segment_ids = [[0] * len(token_ids)]
    input_id = [token_ids]

    outputs, emb = self.session.run(
        [self.output_layer, self.embedding_layer],
        feed_dict={
            self.input_mask: input_mask,
            self.segment_ids: segment_ids,
            self.input_id: input_id
        })

    return outputs[0][1:-1], emb[0][1:-1]

  def tokenize_sentence(self, token_list):
    token_ids = []

    token_ids.append("[CLS]")
    for token_id, token in enumerate(token_list):
      new_tokens = self.tokenizer.tokenize(token)
      token_ids.extend(new_tokens)
    token_ids.append("[SEP]")

    token_ids = self.tokenizer.convert_tokens_to_ids(token_ids)
    return token_ids[1:-1]


def parse_text(text, bert_model):
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

    token_list = [item[0].lower() for item in token_sequence]
    all_vocab.update(bert_model.tokenize_sentence(token_list))
    all_pos.update([item[2] for item in token_sequence])
    all_dep.update([item[3] for item in token_sequence])
    all_path.update([item[4] for item in token_sequence])
    all_lpath.update([item[5] for item in token_sequence])
    all_cp.update([item[6] for item in token_sequence])

    max_length = max(max_length, len(token_sequence))

  return all_pos, all_dep, all_path, all_lpath, all_cp, max_length, all_vocab


def write_tf_records(text, output_filename, bert_model, pos2id, sem2id,
                     root2id, token2id):
  writer = tf.python_io.TFRecordWriter(output_filename)
  sentences = text.split('\n\n')

  for sentence in sentences:
    token_sequence = []
    token_ids = []

    for token in sentence.split('\n'):
      if len(token) >= 8:
        token = token.split('\t')
        token_sequence.append(token)

    tokens = [item[0] for item in token_sequence]
    cues = [int(item[7]) for item in token_sequence]
    pos = [pos2id[item[2]] for item in token_sequence]
    dep = [sem2id[item[3]] for item in token_sequence]
    path = [root2id[item[4]] for item in token_sequence]
    lpath = [int(item[5]) for item in token_sequence]
    cp = [int(item[6]) for item in token_sequence]
    span = [int(item[8]) for item in token_sequence]

    if len(tokens) > 0:
      embeddings, char_emb, new_token_ids, others, masks = bert_model.tokenize(
          tokens, [cues, pos, dep, path, lpath, cp, span])
      cues, pos, dep, path, lpath, cp, span = others

      context = tf.train.Features(feature={  # Non-serial data uses Feature
        "length": tf.train.Feature(int64_list=tf.train.Int64List(value=[len(cues)])),
      })

      for token in new_token_ids:
        if token in token2id.keys():
          token_ids.append(token2id[token])
        else:
          token_ids.append(0)

      token_features = [
          tf.train.Feature(float_list=tf.train.FloatList(
              value=embedding.reshape(-1))) for embedding in embeddings
      ]
      char_token_features = [
          tf.train.Feature(float_list=tf.train.FloatList(
              value=embedding.reshape(-1))) for embedding in embeddings
      ]
      cue_features = [
          tf.train.Feature(int64_list=tf.train.Int64List(value=[cue]))
          for cue in cues
      ]
      pos_features = [
          tf.train.Feature(int64_list=tf.train.Int64List(value=[pos_]))
          for pos_ in pos
      ]
      dep_features = [
          tf.train.Feature(int64_list=tf.train.Int64List(value=[dep_]))
          for dep_ in dep
      ]
      path_features = [
          tf.train.Feature(int64_list=tf.train.Int64List(value=[path_]))
          for path_ in path
      ]
      lpath_features = [
          tf.train.Feature(int64_list=tf.train.Int64List(value=[lpath_]))
          for lpath_ in lpath
      ]
      cp_features = [
          tf.train.Feature(int64_list=tf.train.Int64List(value=[cp_]))
          for cp_ in cp
      ]
      span_features = [
          tf.train.Feature(int64_list=tf.train.Int64List(value=[span_]))
          for span_ in span
      ]
      twe_features = [
          tf.train.Feature(int64_list=tf.train.Int64List(value=[tid_]))
          for tid_ in token_ids
      ]
      mask_features = [
          tf.train.Feature(int64_list=tf.train.Int64List(value=[m_]))
          for m_ in masks
      ]

      feature_list = {
          'token': tf.train.FeatureList(feature=token_features),
          'embedding': tf.train.FeatureList(feature=char_token_features),
          'token_id': tf.train.FeatureList(feature=twe_features),
          'cue': tf.train.FeatureList(feature=cue_features),
          'pos': tf.train.FeatureList(feature=pos_features),
          'dep': tf.train.FeatureList(feature=dep_features),
          'path': tf.train.FeatureList(feature=path_features),
          'lpath': tf.train.FeatureList(feature=lpath_features),
          'cp': tf.train.FeatureList(feature=cp_features),
          'span': tf.train.FeatureList(feature=span_features),
          'masks': tf.train.FeatureList(feature=mask_features),
      }

      feature_lists = tf.train.FeatureLists(feature_list=feature_list)
      ex = tf.train.SequenceExample(feature_lists=feature_lists,
                                    context=context)
      writer.write(ex.SerializeToString())

  writer.close()


def data_iter(x, l):
  current_id = 0
  num_instance = len(x)

  while current_id < num_instance:
    yield x[current_id:current_id + 10], l[current_id:current_id + 10]
    current_id += 10


def main():
  data_path = args.data_path + 'conll/gold_cue/' + args.task + '_' + args.dataset_name + '/'
  output_path = args.data_path + 'tfrecords_bert/gold_cue/' + args.task + '_' + args.dataset_name + '/'

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
  bert_model = BERTModel()

  for filename in filenames:
    full_path = data_path + filename
    text = open(full_path, 'r').read()
    pos, sem, root, depth, cp, max_len_ins, vocab_ins = parse_text(
        text, bert_model)
    all_pos += pos
    all_dep += sem
    all_path += root
    all_depth += depth
    all_vocab += vocab_ins
    all_cp += cp

    max_len = max(max_len, max_len_ins)

  all_vocab = all_vocab.most_common(len(all_vocab))
  valid_vocabulary = [item[0] for item in all_vocab if item[1] >= 5]
  token2id = {
      token: token_id + 1
      for token_id, token in enumerate(valid_vocabulary)
  }

  all_pos = [item[0] for item in all_pos.most_common(len(all_pos))]
  all_dep = [item[0] for item in all_dep.most_common(len(all_dep))]
  all_path = [item[0] for item in all_path.most_common(len(all_path))]

  pos2id = {item: key for key, item in enumerate(all_pos)}
  dep2id = {item: key for key, item in enumerate(all_dep)}
  path2id = {item: key for key, item in enumerate(all_path)}

  print(len(all_pos), len(all_dep), len(all_path), max_len)

  json.dump({
      'pos': pos2id,
      'dep': dep2id,
      'path': path2id
  },
            open(output_path + '/meta.json', 'w'),
            indent=True)

  for filename in filenames:
    full_path = data_path + filename
    text = open(full_path, 'r').read()
    output_file = output_path + filename[:-5] + 'tfr'
    print(output_file)

    write_tf_records(text, output_file, bert_model, pos2id, dep2id, path2id,
                     token2id)


if __name__ == '__main__':
  main()
