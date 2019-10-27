import os
import tensorflow as tf
from bert.tokenization import FullTokenizer
from collections import Counter
import json
import argparse

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

DO_LOWER_CASE = args.bert_model_name.startswith('uncased')
VOCAB_FILE = os.path.join(args.pretrain_models_path, args.bert_model_name,
                          'vocab.txt')


def mapping(cata_dict, key):
  if key in cata_dict.keys():
    return cata_dict[key]
  else:
    return 0


def parse_text(text):
  sentences = text.split('\n\n')

  all_pos = Counter()
  all_dep = Counter()
  all_path = Counter()
  all_vocab = Counter()

  tokenizer = FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE)

  for sentence in sentences:
    token_sequence = []

    for token in sentence.split('\n'):
      if len(token) >= 8:
        token = token.split('\t')
        token_sequence.append(token)

    subwords = sum([tokenizer.tokenize(item[0]) for item in token_sequence],
                   [])
    all_vocab.update(subwords)
    all_pos.update([item[2] for item in token_sequence])
    all_dep.update([item[3] for item in token_sequence])
    all_path.update([item[4] for item in token_sequence])

  return all_pos, all_dep, all_path, all_vocab


def build_dataset(conll_file,
                  tfrecod_file,
                  pos2id,
                  dep2id,
                  path2id,
                  truncate=False):
  max_len = 0

  tokenizer = FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=DO_LOWER_CASE)

  with open(conll_file, 'r') as reader:
    text = reader.read().strip()
  sentences = text.split('\n\n')

  tf_writer = tf.python_io.TFRecordWriter(tfrecod_file)
  for sent in sentences:
    subword_list = ["[CLS]"]
    span_list = [0]
    mask_list = [0]
    cue_list = [0]

    pos_list = [0]
    dep_list = [0]
    path_list = [0]
    lpath_list = [-1]
    cp_list = [-1]

    subword_id_list = tokenizer.convert_tokens_to_ids(["[CLS]"])

    for token in sent.split('\n'):
      if len(token) >= 8:
        token = token.split('\t')

        token_ = token[0]
        subword = tokenizer.tokenize(token_)

        span = [int(token[8]) for _ in range(len(subword))]
        cue = [int(token[7]) for _ in range(len(subword))]

        pos = [int(mapping(pos2id, token[2])) for _ in range(len(subword))]
        dep = [int(mapping(dep2id, token[3])) for _ in range(len(subword))]
        path = [int(mapping(path2id, token[4])) for _ in range(len(subword))]
        lpath = [int(token[5]) for _ in range(len(subword))]
        cp = [int(token[6]) for _ in range(len(subword))]

        mask = [0 for _ in range(len(subword))]
        mask[0] = 1

        sub_id = tokenizer.convert_tokens_to_ids(subword)

        subword_list.extend(subword)
        mask_list.extend(mask)
        subword_id_list.extend(sub_id)

        pos_list.extend(pos)
        dep_list.extend(dep)
        path_list.extend(path)
        lpath_list.extend(lpath)
        cp_list.extend(cp)

        cue_list.extend(cue)
        span_list.extend(span)

    subword_list.append("[SEP]")
    span_list.append(0)
    cue_list.append(0)
    mask_list.append(0)
    subword_id_list.extend(tokenizer.convert_tokens_to_ids(["[SEP]"]))

    pos_list.append(0)
    dep_list.append(0)
    path_list.append(0)
    lpath_list.append(-1)
    cp_list.append(-1)

    assert len(subword_list) == len(span_list) == len(mask_list) == len(
        subword_id_list)

    max_len = max(max_len, len(subword_id_list))

    if len(subword_list) > 2:
      if (not truncate) or (len(subword_id_list) <= 64):
        # write tfrecord
        token_id = [
            tf.train.Feature(int64_list=tf.train.Int64List(value=[t_]))
            for t_ in subword_id_list
        ]
        mask = [
            tf.train.Feature(int64_list=tf.train.Int64List(value=[m_]))
            for m_ in mask_list
        ]
        span = [
            tf.train.Feature(int64_list=tf.train.Int64List(value=[s_]))
            for s_ in span_list
        ]
        cue = [
            tf.train.Feature(int64_list=tf.train.Int64List(value=[c_]))
            for c_ in cue_list
        ]

        pos_features = [
            tf.train.Feature(int64_list=tf.train.Int64List(value=[pos_]))
            for pos_ in pos_list
        ]
        dep_features = [
            tf.train.Feature(int64_list=tf.train.Int64List(value=[dep_]))
            for dep_ in dep_list
        ]
        path_features = [
            tf.train.Feature(int64_list=tf.train.Int64List(value=[path_]))
            for path_ in path_list
        ]
        lpath_features = [
            tf.train.Feature(int64_list=tf.train.Int64List(value=[lpath_]))
            for lpath_ in lpath_list
        ]
        cp_features = [
            tf.train.Feature(int64_list=tf.train.Int64List(value=[cp_]))
            for cp_ in cp_list
        ]

        feature_list = {
            'token_id': tf.train.FeatureList(feature=token_id),
            'span': tf.train.FeatureList(feature=span),
            'masks': tf.train.FeatureList(feature=mask),
            'cue': tf.train.FeatureList(feature=cue),
            'pos': tf.train.FeatureList(feature=pos_features),
            'dep': tf.train.FeatureList(feature=dep_features),
            'path': tf.train.FeatureList(feature=path_features),
            'lpath': tf.train.FeatureList(feature=lpath_features),
            'cp': tf.train.FeatureList(feature=cp_features),
        }

        context = tf.train.Features(
            feature={
                "length":
                tf.train.Feature(int64_list=tf.train.Int64List(
                    value=[len(subword_id_list)])),
            })

        feature_lists = tf.train.FeatureLists(feature_list=feature_list)
        ex = tf.train.SequenceExample(feature_lists=feature_lists,
                                      context=context)
        tf_writer.write(ex.SerializeToString())

  tf_writer.close()


def main():
  data_path = args.data_path + 'conll/gold_cue/' + args.task + '_' + args.dataset_name + '/'
  output_path = args.data_path + 'tfrecords_bert_ft/gold_cue/' + args.task + '_' + args.dataset_name + '/'

  if not os.path.isdir(output_path):
    os.makedirs(output_path)

  filenames = [item for item in os.listdir(data_path)]

  all_pos = Counter()
  all_path = Counter()
  all_dep = Counter()

  for file in filenames:
    text = open(data_path + file, 'r').read()
    pos, dep, path, _ = parse_text(text)

    all_pos += pos
    all_dep += dep
    all_path += path

  all_pos = [item[0] for item in all_pos.most_common(len(all_pos))]
  all_dep = [item[0] for item in all_dep.most_common(len(all_dep))]
  select_len = len([item for item in all_path.keys() if all_path[item] >= 5])
  all_path = [item[0] for item in all_path.most_common(select_len)]

  pos2id = {item: key + 1 for key, item in enumerate(all_pos)}
  dep2id = {item: key + 1 for key, item in enumerate(all_dep)}
  path2id = {item: key + 1 for key, item in enumerate(all_path)}

  json.dump({
      'pos': pos2id,
      'dep': dep2id,
      'path': path2id
  },
            open(output_path + '/meta.json', 'w'),
            indent=True)

  for file in filenames:
    build_dataset(data_path + file, output_path + file[:-6] + '_long.tfr',
                  pos2id, dep2id, path2id)

    build_dataset(data_path + file,
                  output_path + file[:-6] + '_short.tfr',
                  pos2id,
                  dep2id,
                  path2id,
                  truncate=True)


if __name__ == '__main__':
  main()
