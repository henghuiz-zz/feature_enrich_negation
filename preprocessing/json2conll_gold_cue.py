import spacy
import json
import os
import numpy as np
from tqdm import tqdm
from benepar.spacy_plugin import BeneparComponent
from sklearn.model_selection import KFold

DATA_PATH = 'data/'
DATASET_NAME = 'biology_abstract' # clinical_reports or biology_abstract
TASK = 'speculation' # speculation or negation


def get_start_and_end_offset_of_token_from_spacy(token):
  """
  Given a spacy token object return it's start and end indicies
  :param token: spacy token object
  :return: start and end of the span of the token
  """
  start = token.idx
  end = start + len(token)
  return start, end


def get_sentences_and_tokens_from_spacy(text, spacy_nlp, cue_list, scope):
  """
  Returns the document spacy object and a list of sentence objects, where a sentence is a list of tokens.
  Each token is a dictionary where each key represents a given tokens' syntactic feature.
  @:param text: String that contains the whole text
  @:param spacy_nlp: Spacy Language model object
  @:param cue_list: list of cue span
  @:param scope: scope span
  :return: a list of token object
  """

  document = spacy_nlp(text)

  sentence_tokens = []
  cue_id = []

  for idx, token in enumerate(document):
    token_dict = {}
    token_dict['start'], token_dict['end'] = get_start_and_end_offset_of_token_from_spacy(token)
    token_dict['text'] = text[token_dict['start']:token_dict['end']]
    token_dict['pos'] = token.tag_
    path_to_the_root = ""
    length_path = 0
    for token_a in token.ancestors:
      if length_path < 4:
        path_to_the_root += "_" + token_a.pos_
      length_path += 1

    token_dict['dep'] = token.dep_
    token_dict['path'] = path_to_the_root
    token_dict['l_path'] = length_path

    if token_dict['text'].strip() in ['\n', '\t', ' ', '']:
      continue
    # Make sure that the token text does not contain any space
    if len(token_dict['text'].split(' ')) != 1:
      print((
        "WARNING: the text of the token contains space character, replaced with hyphen\n\t{0}\n\t{1}".format(
          token_dict['text'], token_dict['text'].replace(' ', '-'))))
      token_dict['text'] = token_dict['text'].replace(' ', '-')

    is_cue = 0
    for cue_ins in cue_list:
      if (token_dict['start'] >= cue_ins[0]) and (token_dict['end'] <= cue_ins[1]):
        is_cue = 1

    token_dict['is_cue'] = is_cue
    token_dict['is_scope'] = int((token_dict['start'] >= scope[0]) and (token_dict['end'] <= scope[1]))

    if is_cue:
      cue_id.append(idx)

    sentence_tokens.append(token_dict)

  if len(cue_id) == 0:
    print(text, cue_list)

  all_distance = []

  for cue_ins in cue_id:
    dis_negation = 10000 * np.ones(len(document), dtype=int)
    current_span = document[cue_ins]
    dis_negation[cue_ins] = 0
    current_dis = 0

    while np.sum(dis_negation == 10000) > 0:
      current_span = current_span._.parent
      current_dis += 1
      if current_span is None:  # sentence parsing error
        for i in range(len(dis_negation)):
          dis_negation[i] = min(dis_negation[i], current_dis)
      else:
        for token in current_span:
          dis_negation[token.i] = min(dis_negation[token.i], current_dis)

    all_distance.append(dis_negation)

  all_distance = np.min(all_distance, axis=0)

  for dict_ins, cp in zip(sentence_tokens, all_distance):
    dict_ins['cp_path'] = cp

  return sentence_tokens, document


def read_json(json_path):
  all_documents = json.load(open(json_path, 'r'))
  negation_examples = []
  for doc_id in all_documents.keys():
    doc = all_documents[doc_id]
    for sentence in doc:
      negation_dict = {}
      for neg_ins in sentence[TASK]:
        negation_dict.setdefault(neg_ins[0], [])
        negation_dict[neg_ins[0]].append((neg_ins[1], neg_ins[2]))

      for neg_key in negation_dict.keys():
        cue_span = negation_dict[neg_key]
        scope_span = sentence['scope'][neg_key]
        text = sentence['text']
        text_id = doc_id + '/' + neg_key

        negation_examples.append([text, text_id, cue_span, scope_span])

  return negation_examples


def prepare_line(token_ins, doc_id):
  all_text = token_ins['text']
  all_text += '\t'
  all_text += doc_id
  all_text += '\t'

  remaining_item = [token_ins['pos'], token_ins['dep'], token_ins['path'],
                    token_ins['l_path'], token_ins['cp_path'],
                    token_ins['is_cue'], token_ins['is_scope']]

  all_text += '\t'.join([str(item) for item in remaining_item])

  return all_text


def main():
  nlp = spacy.load('en')
  nlp.add_pipe(BeneparComponent('benepar_en2_large'))

  all_negation_examples = read_json(DATA_PATH + 'json/'+DATASET_NAME+'.json')
  write_path = DATA_PATH + 'conll/gold_cue/'+TASK+'_'+DATASET_NAME+'/'
  if not os.path.isdir(write_path):
    os.makedirs(write_path)

  cv = KFold(n_splits=10, shuffle=True, random_state=0)
  split_id = 0
  for _, valid_id in cv.split(all_negation_examples, all_negation_examples):
    valid_examples = [all_negation_examples[i] for i in valid_id]
    with open(write_path + 'train_cv' + str(split_id) + '.conll', 'w') as writer:
      for example in tqdm(valid_examples):
        token_list, sentence = get_sentences_and_tokens_from_spacy(example[0], nlp, example[2], example[3])

        for token in token_list:
          write_string = prepare_line(token, example[1])
          writer.write(write_string + '\n')

        writer.write('\n')

    split_id += 1


if __name__ == '__main__':
  main()
