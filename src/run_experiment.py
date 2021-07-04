import os
import time
import traceback
import csv
import numpy
import glob
import json
import re
from html.parser import HTMLParser
from html.entities import entitydefs
import pandas as pd
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError as XmlParseError

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
import re
import pandas as pd 
from bs4 import BeautifulSoup

from utils.code_extractor import *

from elasticsearch import Elasticsearch
INDEX_NAME = "so_top5_java_threads_no_paren"
es_client = Elasticsearch([{'host': "elasticsearch", 'port': 9200}])

# from spacy import Tokenizer
from pprint import PrettyPrinter as PP
pp = PP(depth=2)
pprint = pp.pprint

import sys, os.path
sys.path.append(os.path.abspath('/app/data/'))

class ApiMention:
    def __init__(self, api, label, sent_no, start_pos, end_pos):
        self.api = api
        self.label = label
        self.sent_no = sent_no
        self.start_pos = start_pos
        self.end_pos = end_pos
    
    def __repr__(self):
        return f"{(self.api, self.label, self.sent_no, self.start_pos, self.end_pos)}"
    
    def to_dict(self):
        return {
            'api': self.api,
            'label': self.label,
            'sent_no': self.sent_no,
            'start_pos': self.start_pos,
            'end_pos': self.end_pos
        }


class CodeBlock:
    def __init__(self, code, start_at_line, end_at_line):
        self.code = self.remove_api_tag(code)
        self.start = start_at_line
        self.end = end_at_line
        self.neighbor_text = None
    
    def remove_api_tag(self, code):
        pattern = '<API label="(.*?)">(.*?)</API>'
        code = re.sub(pattern,  "\g<2>", code)
        return code
    
    def get_line_number_in_block(self, line_number_in_content):
        if line_number_in_content > self.start and line_number_in_content < self.end:
            return line_number_in_content - self.start - 1
        else:
            return -1
    
    def get_code_line_in_block(self, api_mention):
        line_i = self.get_line_number_in_block(api_mention.sent_no)
        if line_i == -1:
            return ""
        else:
            return self.get_code_wo_tag().split("\n")[line_i]
        
    def get_code_wo_tag(self):
        return "\n".join(self.code.split("\n")[1:-1])
    
    def add_neighbor_text(self, neighbor_text):
        self.neighbor_text = neighbor_text
    
    def __repr__(self):
        return "\n".join([f"{line_i:03d}: {line}" for line_i, line in enumerate(self.get_code_wo_tag().split("\n"))])
    
    def get(self):
        return self.get_code_wo_tag()

    
class ListCodeBlock:
    def __init__(self, code_blocks):
        self.list = code_blocks
    
    def get_code_block_of_mention(self, api_mention):
        for blk_i, code_block in enumerate(self.list):
            line_in_code = code_block.get_line_number_in_block(api_mention.sent_no)
            if line_in_code != -1:
                return True, blk_i, line_in_code
        return False, -1, -1

def find_text(content):
    texts = []
    start = 0
    end = -1
    for line_idx, line in enumerate(content.split("\n")):
        if line.strip() == "<pre><code>":
            end = line_idx - 1
        elif line.strip() == "</code></pre>":
            start = line_idx + 1
        if (start != -1) and (end != -1):
            if start == 0 and end == 0:
                pass
#             texts.append(CodeBlock("\n".join(content.split("\n")[start:end+1]), start, end))
            else:
                texts.append("\n".join([line for line in content.split("\n")[start:end+1] if (line.strip() != "" and line.strip() != "==========")]))
            start = -1
            end = -1
    text_end = []
    content_reverse = content.split("\n")
#     [line for line in content.split("\n") if (line.strip() != "" and line.strip != "==========")]
    content_reverse.reverse()
#     print(content_reverse)
    return_content = None
    for line_idx, line in enumerate(content_reverse):
        if line.strip() == "</code></pre>":
            if line_idx == 0:
                return texts
            else:
                return_content = content_reverse[:line_idx-1]
                return_content.reverse()
                break
    if return_content is not None:
        return_content = [line for line in return_content if (line.strip() != "" and line.strip() != "==========")]
        return texts+return_content
    else:
        return texts

def find_code_block(content):
    blocks = []
    start = -1
    end = -1
    for line_idx, line in enumerate(content.split("\n")):
        if line.strip() == "<pre><code>":
            start = line_idx
        elif line.strip() == "</code></pre>":
            end = line_idx
        if (start != -1) and (end != -1):
            blocks.append(CodeBlock("\n".join(content.split("\n")[start:end+1]), start, end))
            start = -1
            end = -1
    return blocks



data_labeling_dir = "/app/data/so_threads/**"
tags_dir = "/app/data/tags.json"
title_dir = "/app/data/title.json"
api_method_candidates_dir = "/app/data/api_method_candidates.json"
# files = glob.glob(data_labeling_dir+"/*")
# files = [f for f in files if "notjava" not in f]
# print(len(files))
# file_dict = []
# for _file in files:
#     with open(_file, "r") as fp:
#         content_raw = fp.read()
#     content = "\n".join(content_raw.split("\n")[1:])
#     texts = find_text(content)
#     out_file = "/app/output/data_top5_libs/share_for_labeling/for_apireal/" + os.sep.join(_file.split(os.sep)[-1:])
#     os.makedirs(os.path.dirname(out_file), exist_ok=True)
#     with open(out_file, "w+") as fp:
#         for text in texts:
#             print(text, file=fp)
#     _filename = _file.split(".")[-1]
#     file_dict.append({
#         'id': _file.split(os.sep)[-1].split(".")[0],
#         'raw': _file,
#         'text': out_file
#     })

files = glob.glob(data_labeling_dir+"/*")
file_dict = []
with open(tags_dir, "r") as fp:
    tags_dict = json.load(fp)
with open(title_dir, "r") as fp:
    title_dict = json.load(fp)
with open(api_method_candidates_dir, "r") as fp:
    api_cands = json.load(fp)

# for _file in files:
#     with open(_file, "r") as fp:
#         content_raw = fp.read()
#     content = "\n".join(content_raw.split("\n")[1:])


def read_txt(filename):
    with open(filename, "r") as fp:  
        content = fp.read()
    return content

def tokenize(text):
    count_vec = CountVectorizer(lowercase=False)
    content_vocabs = count_vec.fit([text]).vocabulary_
    tokens= (list(content_vocabs.keys()))
    return tokens, content_vocabs

def update_dicts(dict_old, dict_new):
    from copy import deepcopy
    updated_dict = deepcopy(dict_old)
    for key, list_value in dict_new.items():
        if key not in updated_dict:
            updated_dict[key] = []
        for item in list_value:
            if item not in updated_dict[key]:
                updated_dict[key].append(item)
    return updated_dict

class Thread:
    def __init__(self, content, title, tags):
        thread_id = content.split("\n")[0].split("/")[-1]
        content= content.split("\n")[1:]
        links = []
        new_content = []
        for sentence in content:
            if sentence[:11] == "<pre><code>":
                new_content.append(sentence[:11])
                new_content.append(sentence[11:])
            else:
                new_content.append(sentence)
        content = "\n".join(new_content)
        code_blocks = find_code_block(content)
        list_code_block = ListCodeBlock(code_blocks)
        texts = find_text(content)
        pattern = '<API label="(.*?)">(.*?)</API>'
        list_mention = []
        for sent_no, sentence in enumerate(content.split("\n")):
            match = re.search(pattern, sentence)
            while match is not None:
                s = match.start()
                matching_tag = match.group(0)
                label =  match.group(1)
                api = match.group(2)
                sentence = re.sub(re.escape(matching_tag), api, sentence, 1)
                list_mention.append(ApiMention(api, label, sent_no, s, s+len(api)))
                match = re.search(pattern, sentence)
#                 import time
#                 time.sleep(0.5)
        mention_codeblk_mapping = {}
        for mention_indice, mention in enumerate(list_mention):
            res, code_blk_indice, code_line = list_code_block.get_code_block_of_mention(mention)
            if res:
                mention_codeblk_mapping[mention_indice] = code_blk_indice

        for mention_i, code_blk_i in mention_codeblk_mapping.items():
            code_block = list_code_block.list[code_blk_i]
            api_mention = list_mention[mention_i]

        self.thread_id = thread_id
        self.texts = texts
        self.list_code_block = list_code_block
        self.list_mention = list_mention
        self.mention_codeblk_mapping = mention_codeblk_mapping
        self.tags = tags
        self.title = title
        
    def remove_api_tag(self, text):
        pattern = '<API label="(.*?)">(.*?)</API>'
        text = re.sub(pattern,  "\g<2>", text)
        return text
    
    def get_text(self):
        return "\n".join([line for line in self.texts if line.strip() != ""])
    
    def get_text_wo_label(self):
#         def proc_line(line):
#             soup = BeautifulSoup(line, 'html.parser')
#             return soup.text
        return "\n".join([self.remove_api_tag(line) for line in self.texts if line.strip() != ""])
        
    def get_code(self):
        return "\n".join([blk.get() for blk in self.list_code_block.list])
    
    def get_tags(self):
        # tags = es_client.get(id=self.thread_id, index=INDEX_NAME)['_source']['tags']
        return self.tags
    
    def get_title(self):
        # title = es_client.get(id=self.thread_id, index=INDEX_NAME)['_source']['title']
        return self.title
    
    def get_variation_dict(self):
        variations = {}
        var_type_dict, fn_var_dict, import_dict = extract_code(self.get_code())
        for key in import_dict.keys():
            parts = key.split(".")
            if parts[-1] == "*":
                continue
            variations[parts[-1]] = key
        
        for var, value in var_type_dict.items():
            _type = list(value.keys())[0]
            variations[var] = _type
        return variations
            
                
            
        return variations
    def extract_dp(self):
        try:
#             print(self.get_code())
            
            var_type_dict, fn_var_dict, import_dict = extract_code(self.get_code())
            dep_tracing_dict = resolve_imports(import_dict)
            fn_var_dict_trans = {}
            for fn, _vars in fn_var_dict.items():
                list_vars_calling_fn = [_var for _var in _vars.keys()]
                fn_var_dict_trans[fn] = list_vars_calling_fn
            single_type_method, multi_type_method = determine_var_package(var_type_dict, fn_var_dict)

            fn_type_dict = {}
            for key, value in single_type_method.items():
                if key not in multi_type_method:
                    multi_type_method[key] = []
                if value not in multi_type_method[key]:
                    multi_type_method[key].append(value)

            fn_type_dict = multi_type_method
            dependencies = update_dicts({}, fn_type_dict)
            dependencies = update_dicts(dependencies, dep_tracing_dict)
            return dependencies
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise Exception(f"Error with Thread {self.thread_id}")
            
    def get_api_mention_text(self):
        text = self.get_text()
        list_mentions_in_text = []
        for line_i, text_line in enumerate(text.split("\n")):
            soup = BeautifulSoup(text_line, 'html.parser')
            for api in soup.findAll("api"):
                if api['label'].strip() != "":
                    api_mention = {
                        'name': api.text,
                        'label': api['label'],
                        'line': soup.text,
                        'line_i': line_i
                    }
                    list_mentions_in_text.append(api_mention)
        return list_mentions_in_text
    
def get_text_scope(mentions, thread):
    text_scope = thread.get_text_wo_label()
    
    text_scope_lines = text_scope.split("\n")
    for mention in mentions:
        line = text_scope_lines[mention['line_i']]
        line = line.replace(mention['name'], " ", 1)
        text_scope_lines[mention['line_i']] = line
    return "\n".join(text_scope_lines)

def type_to_pred_1(mention, thread, text_scope):
    DEBUG_SCOPE= False
    if DEBUG_SCOPE: print(mention)
    fn_name = mention['name'].split(".")[-1]
    fn_name_caller = ".".join(mention['name'].split(".")[:-1])
    tags = thread.tags
    thread_content = thread.get_code() + "\n" + thread.get_text_wo_label() + " ".join(tags)
#     thread_content = thread.get_text_wo_label()
#     """
    text_scope = thread.get_title() + " " +thread.get_text_wo_label()
    if DEBUG_SCOPE:
        print("[Modified text scope]")
        print(text_scope)
        print('-------------')

    thread_tokens, thread_vocabs = tokenize(thread_content)
    text_scope_tokens, _= tokenize(text_scope)
    candidates = deepcopy(api_cands[fn_name])
    filtered_cands = []
    for fqn, lib in candidates.items(): # 4)
        for tag in tags:
            if tag in lib:
                filtered_cands.append(fqn)
                break
    candidates = filtered_cands
    score_dict = {} 
    has_type_one = {}
    for can_i, can in enumerate(candidates): 
        score_dict[can] = 0
        has_type_one[can] = False
        for domain in mention['domains']: # 5)
            can_parts = can.split(".") # increase point
            can_class =can_parts[-2]
            if domain in can:
#                 continue
                
                if len(domain.split(".")) > 1:
                    if domain.split(".")[-1] != fn_name:
                        focus_str = domain.split(".")[-2]
                    else:
                        focus_str = domain.split(".")[-1]
                    
                else:
                    focus_str = domain.split(".")[-1]

                if focus_str == can_class:
                    score_dict[can] += 1

    for can in candidates:
        class_name = can.split(".")[-2]
        if fn_name_caller != "":
            
            if fn_name_caller == class_name: #2 6)
                score_dict[can] += 1
                has_type_one[can] = True
                

        if class_name in text_scope_tokens:
            score_dict[can] += 1

            
    score_list = [(api, score) for api, score in score_dict.items()]
    
    score_list = sorted(score_list, key=lambda api: api[1], reverse=True)

    if len(score_list) == 0:
        return "None", 0, False
    if score_list[0][1] == 0:
        return "None", 0, False
    
    return score_list[0][0], score_list[0][1], has_type_one[score_list[0][0]]


def eval_mentions(text_mentions):
    print("Nrof mentions:", len(text_mentions))
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    count_case_in_cand = 0
    count_case_not_in_cand = 0
    tp_cases = []
    fp_cases = []
    fn_cases = []
    tn_cases = []
    for mention in text_mentions:
        pred = mention['pred']
        label = mention['label']
        
        if label == "None":
            count_case_not_in_cand += 1
        else:
            count_case_in_cand += 1
            
        if pred == 'None':
            if label == "None":
                tn += 1
                tn_cases.append(mention)
            else:
                fn += 1
                fn_cases.append(mention)
        else:
            if pred == label:
                tp += 1
                tp_cases.append(mention)
                
            else:
                fp += 1
                fp_cases.append(mention)
        if DEBUG:
            display(mention)
    print("None cases:", count_case_not_in_cand)
    print("cases in API list:", count_case_in_cand)
    
    if tp+fp == 0:
        prec = 0
    else:
        prec = tp/(tp+fp)
    if tp +fn == 0:
        recall = 0
    else:
        recall = tp/(tp+fn)
    print("precison: ", prec)
    print("recall: ", recall)
    print(tp)
    if prec+recall == 0:
        f1 = -1
    else:
        f1 = 2*prec*recall/(prec+recall)
    print("F1-score: ", f1)
    print(f"tp: {tp}, fp: {fp}, fn: {fn}, tn: {tn}")
    return fp_cases, tp_cases, fn_cases, tn_cases

DEBUG = False
# Start
my_mc = {}
file_count = 0
list_all_predicted_mentions = []
for file_temp in files:
    # if "notjava" in file_temp:
    #     continue
    # if "remove" in file_temp:
    #     continue
    file_count += 1
    thread_id =  file_temp.split(os.sep)[-1].split(".")[0]
    a_thread = Thread(read_txt(file_temp), title_dict[thread_id], tags_dict[thread_id])
    if DEBUG: print(a_thread.thread_id)
    
    variations = a_thread.get_variation_dict()
    if DEBUG:
        print("variations")
        # display(variations)
        
    dep_dict = a_thread.extract_dp()
    text_mentions = a_thread.get_api_mention_text()
    my_mc[a_thread.thread_id] = len(text_mentions)
    
    new_text_mentions = []
    for m_idx, m in enumerate(text_mentions):
        mention = deepcopy(m)
        mention['domains'] = []
        list_domains = []
        simple_m_name = m['name'].split(".")[-1]
        prefix = ".".join(m['name'].split(".")[:-1])
        if prefix != "":
            if prefix in dep_dict:
                prefix_domains = dep_dict[prefix]
                for domain in prefix_domains:
                    list_domains.append(domain)
                    if domain in dep_dict:
                        list_prefix_domains += dep_dict[domain]
                    
        elif m['name'] in dep_dict:
            method_related_domains = dep_dict[m['name']]
            for domain in method_related_domains:
                list_domains.append(domain)
                if domain in dep_dict:
                    list_domains += dep_dict[domain]
        mention['domains'] = list(set(list_domains))
        mention['thread'] = a_thread.thread_id
        new_text_mentions.append(mention)


    text_scope = get_text_scope(new_text_mentions, a_thread)
    for mention in new_text_mentions:
        mention['pred'], mention['score'], has_type_one = type_to_pred_1(mention, a_thread, text_scope) # Approach 1: Use only import variations
        if has_type_one:
            mention['has_type_one'] = True

    list_all_predicted_mentions += new_text_mentions

eval_mentions(list_all_predicted_mentions)