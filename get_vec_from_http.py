#!/usr/bin/python
# -*- coding:utf-8 -*-

import requests
import json

url = "http://127.0.0.1:7777/vector"
headers = {
    "Content-Type": "application/json"
}

dict_input = dict()
dict_input["text"] = "自然语言 处理 后台"
dict_input["validLength"] = -1

ret = requests.post(url=url, json=dict_input, headers=headers)
dict_ret = json.loads(ret.text)
raw_text = dict_ret["rawText"]
raw_valid_length = dict_ret["rawValidLength"]
text = dict_ret["text"]
valid_length = dict_ret["validLength"]
vector = dict_ret["vector"]
print("-----------")
print("原始输入文本：" + raw_text)
print("原始输入长度：" + str(raw_valid_length))
print("实际处理文本：" + text)
print("实际处理长度：" + str(valid_length))
print("ALBERT 向量 %d 维：\n" % len(vector) + str(vector))
print("-----------")
