# zju-cv-hw1
2020浙江大学软件学院计算机视觉导论课程第一次作业

炼丹近3天，最终确定K-means均值簇数n_clusters=70.

### 一、检索图片

这里为了方便，检索的图片放在了 zju-cv-hw1/dataset/test 文件夹下

```
python retrieve.py --name 带检索图片的部分路径名你（类别/图片名称.ppm）
```

example:

```
python retrieve.py --name bark/img1.ppm
```

### 二、评估检索性能

```
python train_eval.py
```

结果：

```bash
Test Acc: 0.9
Precision, Recall and F1-Score...
              precision    recall  f1-score   support

        bark     1.0000    1.0000    1.0000         1
       bikes     0.0000    0.0000    0.0000         1
        boat     1.0000    1.0000    1.0000         1
        graf     0.5000    1.0000    0.6667         1
      leuven     1.0000    1.0000    1.0000         1
       trees     1.0000    1.0000    1.0000         1
         ubc     1.0000    1.0000    1.0000         2
        wall     1.0000    1.0000    1.0000         2

    accuracy                         0.9000        10
   macro avg     0.8125    0.8750    0.8333        10
weighted avg     0.8500    0.9000    0.8667        10

Confusion Matrix...
[[1 0 0 0 0 0 0 0]
 [0 0 0 1 0 0 0 0]
 [0 0 1 0 0 0 0 0]
 [0 0 0 1 0 0 0 0]
 [0 0 0 0 1 0 0 0]
 [0 0 0 0 0 1 0 0]
 [0 0 0 0 0 0 2 0]
 [0 0 0 0 0 0 0 2]]

```