#!/usr/bin/python
# -*- coding:utf-8 -*-

from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

config_path = 'model/albert_tiny_zh_google/albert_config_tiny_g.json'
checkpoint_path = 'model/albert_tiny_zh_google/albert_model.ckpt'
dict_path = 'model/albert_tiny_zh_google/vocab.txt'

tokenizer = Tokenizer(dict_path, do_lower_case=True)
model = build_transformer_model(config_path, checkpoint_path, model='albert',
                                with_pool=True)  # 建立模型，加载权重

token_ids, segment_ids = tokenizer.encode(u'你好世界')
print(token_ids, segment_ids)

print('\n ===== predicting =====\n')
vec1 = model.predict([np.array([token_ids]), np.array([segment_ids])])
print(vec1)
