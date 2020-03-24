#!/usr/bin/python
# -*- coding:utf-8 -*-
# 生成TensorFlow pb模型文件供Java工程使用

from bert4keras.models import build_transformer_model
import tensorflow as tf
from keras import backend as K
from tensorflow.python.framework import graph_util, graph_io
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def export_graph(model, export_path, output_name):
    input_names = model.input_names

    if not tf.gfile.Exists(export_path):
        tf.gfile.MakeDirs(export_path)

    with K.get_session() as sess:
        init_graph = sess.graph
        with init_graph.as_default():
            out_nodes = []

            for i in range(len(model.outputs)):
                out_nodes.append("output_" + str(i + 1))
                tf.identity(model.output[i], "output_" + str(i + 1))

            init_graph = sess.graph.as_graph_def()
            main_graph = graph_util.convert_variables_to_constants(sess, init_graph, out_nodes)
            graph_io.write_graph(main_graph, export_path, name='%s.pb' % output_name, as_text=False)

    return input_names, out_nodes


if __name__ == '__main__':
    config_path = 'model/albert_tiny_zh_google/albert_config_tiny_g.json'
    checkpoint_path = 'model/albert_tiny_zh_google/albert_model.ckpt'
    dict_path = 'model/albert_tiny_zh_google/vocab.txt'
    output_path = "output/"
    model = build_transformer_model(config_path, checkpoint_path, model='albert', with_pool=True)  # 建立模型，加载权重
    inputs, outputs = export_graph(model, output_path, "albert_tiny_zh_google")
    print(inputs)
    print(outputs)
