import paddle
def Accuracy_evaluation(log_probs,labels):
    # 2,49,20
    index = paddle.argmax(log_probs,axis=-1)
    equal_labels = paddle.equal(labels, index)
    # 计算等于 index 的数量，并除以 labels 的长度来得到比例
    acc = (paddle.sum(equal_labels.astype(paddle.int32)) / len(labels)).item()


    return acc * 100