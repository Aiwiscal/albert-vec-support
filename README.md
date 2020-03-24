## ALBERT Vector Service with Spring Boot(https://github.com/Aiwiscal/albert-vec) <br> Python辅助代码

### requirements:
Keras==2.2.4 <br>
requests==2.18.4 <br>
tensorflow==1.12.0 <br>
numpy==1.14.0 <br>
bert4keras==0.5.8 <br>

### 代码作用
    - bert4keras_extract_feature.py:使用bert4keras载入albert预训练权重，实现向量预测
    - freeze_TFmodel.py:生成TensorFlow pb文件，便于Java环境中的应用
    - get_vec_from_http.py: 启动albert-vec服务后，使用http请求获得文本的albert向量表示

### 参考
bert4keras: https://github.com/bojone/bert4keras <br>
