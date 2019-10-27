import bs4
import json
import html
import os
import argparse
from bs4 import BeautifulSoup

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data_path', type=str, default='data/', help='path for the data folder')
args = parser.parse_args()


def get_span_and_tag(sentence, previous_tag=None):
  if previous_tag is None:
    previous_tag = []
  all_span_tag = []
  for span in sentence.children:
    if isinstance(span, bs4.element.NavigableString):
      all_span_tag.append([html.unescape(span.string), previous_tag])
    else:
      this_tag = span.attrs.copy()
      this_tag['name'] = span.name
      results = get_span_and_tag(span, previous_tag + [this_tag])
      all_span_tag += results

  return all_span_tag


def return_span(all_span_tag):
  all_scope = {}
  all_neg_cue = []
  all_spe_cue = []

  all_text = ''

  for span_instance in all_span_tag:
    begin = len(all_text)
    if begin == 0:
      all_text += span_instance[0].lstrip()
    else:
      all_text += span_instance[0]

    end = len(all_text)

    for span_type in span_instance[1]:
      if span_type['name'] == 'xcope':
        if span_type['id'] in all_scope.keys():
          all_scope[span_type['id']][1] = end
        else:
          all_scope[span_type['id']] = [begin, end]
      else:
        assert span_type['name'] == 'cue'
        if span_type['type'] == 'negation':
          all_neg_cue.append([span_type['ref'], begin, end])
        else:
          assert span_type['type'] == 'speculation'
          all_spe_cue.append([span_type['ref'], begin, end])

  all_text = all_text.rstrip()
  return all_text, all_scope, all_spe_cue, all_neg_cue


def parse_clinical():
  clinical_xml = open(args.data_path + 'raw/clinical_modified.xml', 'r', encoding='UTF-8').read()
  soup = BeautifulSoup(clinical_xml, "lxml")

  all_record = soup.find_all(type="Medical_record")

  all_record_json = {}

  for record in all_record:
    record_id = record.find(type='CMC_DOCID').text
    all_sentence = record.find_all('sentence')

    record_obj = []
    for sentence in all_sentence:
      span_list = get_span_and_tag(sentence)
      all_text, all_scope, all_spe_cue, all_neg_cue = return_span(span_list)
      record_obj.append(
        {'text': all_text, 'scope': all_scope, 'speculation': all_spe_cue, 'negation': all_neg_cue}
      )

    all_record_json[record_id] = record_obj

  json.dump(all_record_json, open(args.data_path + 'json/clinical_reports.json', 'w', encoding='UTF-8'), indent=True)


def parse_bio():
  clinical_xml = open(args.data_path + 'raw/abstracts_pmid_modified.xml', 'r', encoding='UTF-8').read()
  soup = BeautifulSoup(clinical_xml, "lxml")

  all_record = soup.find_all(type="Biological_abstract")

  all_record_json = {}

  for record in all_record:
    record_id = record.find(type='PMID').text
    all_sentence = record.find_all('sentence')

    record_obj = []
    for sentence in all_sentence:
      span_list = get_span_and_tag(sentence)
      all_text, all_scope, all_spe_cue, all_neg_cue = return_span(span_list)
      record_obj.append(
        {'text': all_text, 'scope': all_scope, 'speculation': all_spe_cue, 'negation': all_neg_cue}
      )

    all_record_json[record_id] = record_obj

  json.dump(all_record_json, open(args.data_path + 'json/biology_abstract.json', 'w', encoding='UTF-8'), indent=True)


if __name__ == '__main__':
  if not os.path.isdir(args.data_path + 'json/'):
    os.makedirs(args.data_path + 'json/')
  parse_clinical()
  parse_bio()
