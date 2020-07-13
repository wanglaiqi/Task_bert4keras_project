#!/usr/bin/env python
# coding:utf-8
"""
Task:the logic about tools
Date:2020.05.18
Author:Laiqi
"""
import yaml
import json

# tool_01:load info of yaml file
def load_yaml_file(filename):
    # read config file
    with open(filename, 'r') as file:
        data = file.read()
        config_dict = yaml.load(data, Loader=yaml.FullLoader)
    # return the data
    return config_dict

# tool_02 load json dataset from file
def load_json_file(filename):
    with open(filename, 'r', encoding="utf-8") as file:
        data_list = []
        for line in file.readlines():
            temp_dict = json.loads(line)
            # add data into data_list
            data_list.append(temp_dict)
    # return the result
    return data_list

# # tool_03:save json data into file
# def save_data(data_list):
#     # diplay character is chinese
#     # data_list = json.dumps(data_list, ensure_ascii=False)
#     with open(data_path + "train_new_sub.json", "w") as file:
#         # traverse the data_list
#         for data in data_list:
#             data = json.dumps(data, ensure_ascii=False)
#             file.write(data)
#             file.write("\n")

# tool_03: load the data format of conll
def load_conll_file(filename):
    """
    Returns:
        [([word1, word2, word3, word4], [label1, label2, label3, label4]),
        ([word5, word6, word7, wordd8], [label5, label6, label7, label8])]
    """
    dataset = []
    with open(filename, "r") as f:
        words, tags = [], []
        # for each line of the file correspond to one word and tag
        for line in f:
            if line != "\n":
                # line = line.strip()
                word, tag = line.split(" ")
                word = word.strip()
                tag = tag.strip()
                try:
                    if len(word) > 0 and len(tag) > 0:
                        word, tag = str(word), str(tag)
                        words.append(word)
                        tags.append(tag)
                except Exception as e:
                    print("an exception was raise! skipping a word")
            else:
                if len(words) > 0:
                    assert len(words) == len(tags)
                    dataset.append((words, tags))
                    words, tags = [], []
    # return the result
    return dataset

# tool_04: get span labels
def get_span_labels(sentence_tags, inv_label_mapping=None):
    """
    Desc:get from token_level labels to list of entities, it doesnot matter tagging scheme is BMES or BIO or BIOUS
    Returns: a list of entities [(start, end, labels), (start, end, labels)]
    """
    if inv_label_mapping:
        sentence_tags = [inv_label_mapping[i] for i in sentence_tags]
    span_labels = []
    last = "O"
    start = -1
    # traverse the sentence tags
    for i, tag in enumerate(sentence_tags):
        pos, _ = (None, "O") if tag == "O" else tag.split("-")
        if (pos == "S" or pos == "B" or tag == "O") and last != "O":
            span_labels.append((start, i - 1, last.split("-")[-1]))
        if pos == "B" or pos == "S" or last == "O":
            start = i
        last = tag
    if sentence_tags[-1] != "O":
        span_labels.append((start, len(sentence_tags) -1 , sentence_tags[-1].split("-"[-1])))
    # return the result
    return span_labels

