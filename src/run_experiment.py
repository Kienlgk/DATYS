import os
import traceback
import glob
import json
import re
from bs4 import BeautifulSoup

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

from utils.code_extractor import *
from Thread import Thread
from Metrics import eval_mentions

import sys
sys.path.append(os.path.abspath('/app/data/'))

data_labeling_dir = "/app/data/so_threads/"
tags_dir = "/app/data/tags.json"
title_dir = "/app/data/title.json"
api_method_candidates_dir = "/app/data/api_method_candidates.json"

with open(tags_dir, "r") as fp:
    tags_dict = json.load(fp)
with open(title_dir, "r") as fp:
    title_dict = json.load(fp)
with open(api_method_candidates_dir, "r") as fp:
    api_cands = json.load(fp)
        

def read_txt(filename):
    with open(filename, "r") as fp:  
        content = fp.read()
    return content

def tokenize(text):
    count_vec = CountVectorizer(lowercase=False)
    content_vocabs = count_vec.fit([text]).vocabulary_
    tokens= (list(content_vocabs.keys()))
    return tokens, content_vocabs

def get_text_scope(mentions, thread):
    text_scope = thread.get_text_wo_label()
    
    text_scope_lines = text_scope.split("\n")
    for mention in mentions:
        line = text_scope_lines[mention['line_i']]
        line = line.replace(mention['name'], " ", 1)
        text_scope_lines[mention['line_i']] = line
    return "\n".join(text_scope_lines)

def type_scoping(mention, thread, text_scope):
    fn_name = mention['name'].split(".")[-1]
    fn_name_caller = ".".join(mention['name'].split(".")[:-1])
    tags = thread.tags
    text_scope = thread.get_title() + " " +thread.get_text_wo_label() + " ".join(tags)
    text_scope_tokens, _= tokenize(text_scope)
    candidates = deepcopy(api_cands[fn_name])
    filtered_cands = []
    for fqn, lib in candidates.items():
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
        for pos_type in mention['p_types']:
            can_parts = can.split(".")
            can_class =can_parts[-2]
            if pos_type in can:
                if len(pos_type.split(".")) > 1:
                    if pos_type.split(".")[-1] != fn_name:
                        focus_str = pos_type.split(".")[-2]
                    else:
                        focus_str = pos_type.split(".")[-1]
                    
                else:
                    focus_str = pos_type.split(".")[-1]

                if focus_str == can_class:
                    score_dict[can] += 1

    for can in candidates:
        class_name = can.split(".")[-2]
        if fn_name_caller != "":
            if fn_name_caller == class_name:
                score_dict[can] += 1

        if class_name in text_scope_tokens:
            score_dict[can] += 1

    score_list = [(api, score) for api, score in score_dict.items()]
    score_list = sorted(score_list, key=lambda api: api[1], reverse=True)

    if len(score_list) == 0 or score_list[0][1] == 0:
        prediction = "None"
        score = 0
    else:
        prediction = score_list[0][0]
        score = score_list[0][1]
    
    return prediction, score




def run_experiment():
    files = glob.glob(data_labeling_dir+"/*")
    file_dict = []
    
    # Start
    list_all_predicted_mentions = []
    for file_temp in files:
        thread_id =  file_temp.split(os.sep)[-1].split(".")[0]
        a_thread = Thread(read_txt(file_temp), title_dict[thread_id], tags_dict[thread_id])
        
        possible_type_list = a_thread.get_possible_type_dict()
            
        p_type_dict = a_thread.extract_possible_types()
        text_mentions = a_thread.get_api_mention_text()
        
        new_text_mentions = []
        for m_idx, m in enumerate(text_mentions):
            mention = deepcopy(m)
            mention['p_types'] = []
            list_p_types = []
            simple_m_name = m['name'].split(".")[-1]
            prefix = ".".join(m['name'].split(".")[:-1])
            if prefix != "":
                if prefix in p_type_dict:
                    p_types_of_prefix = p_type_dict[prefix]
                    for p_type in p_types_of_prefix:
                        list_p_types.append(p_type)
                        
            elif m['name'] in p_type_dict:
                method_related_p_types = p_type_dict[m['name']]
                for p_type in method_related_p_types:
                    list_p_types.append(p_type)
                    if p_type in p_type_dict:
                        list_p_types += p_type_dict[p_type]
            mention['p_types'] = list(set(list_p_types))
            mention['thread'] = a_thread.thread_id
            new_text_mentions.append(mention)


        text_scope = get_text_scope(new_text_mentions, a_thread)
        for mention in new_text_mentions:
            mention['pred'], mention['score'] = type_scoping(mention, a_thread, text_scope)

        list_all_predicted_mentions += new_text_mentions

    eval_mentions(list_all_predicted_mentions)

if __name__ == "__main__":
    print("Start running DATYS ...")
    run_experiment()