import datetime
import os
import time
from argparse import ArgumentParser
import numpy as np
import paddle,logging
from tqdm import trange, tqdm
import paddle.nn.functional as F

from AccuracyEvaluation import Accuracy_evaluation
from Model.PPLCNet_x1_0.PPLCNet_x1_0 import PPLCNET_x1_5
from Dataset.DataSet import StructureLoader
# from Model.YoloBody import YoloBody
from recoveryModel import load_model_optimizer


def setargs():
    # 用于解析命令行参数并生成帮助文档 定义程序所需要的命令行参数，以及它们的类型、默认值、描述等信息
    parser = ArgumentParser()
    parser.add_argument("--trainPath", default="train",type=str, help="Image path")  # 图片路径
    parser.add_argument("--valPath", default="val",type=str, help="Image path")  # 图片路径
    parser.add_argument("--testPath", default="test",type=str, help="Image path")  # 图片路径
    parser.add_argument("--mode", default="train", choices=["train", "validate", "seqdesign", "multiscan"])
    parser.add_argument("--lr", default=0.01, type=float, help="Model learning rate")
    parser.add_argument("--pictureSize", default=224,type=int,help="convert image size")  # 图片大小
    parser.add_argument("--maxEpochs", default=100,type=int,help="Maximum training batch")  # 训练批次大小
    parser.add_argument("--log_dir", default='log', help="path to log into")  # 日志路径
    parser.add_argument("--path", default="checkpoint", help="path to checkpoint to restore")  # 要创建的目录
    parser.add_argument("--checkpoint", default=None, help="path to checkpoint to restore")  # 要恢复的检查点的路径
    parser.add_argument("--device", default="cpu", type=str,
                        help="Name of the device.")  #
    parser.add_argument('--seed', type=int, default=1111, help='random seed for reproducibility')
    parser.add_argument('--batch_size', type=int, default=20, help='batch size')
    parser.add_argument('--ifGray', type=bool, default=False, help='Is the image grayscale')
    parser.add_argument('--class_num', type=int, default=2, help='The quantity that needs to be classified')
    parser.add_argument('--dropout', type=float, default=0.2, help='Random dropout probability')
    parser.set_defaults(verbose=False)  # 将名为 verbose 的参数的默认值设置为 False。
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':

    # 加载 logging 实例
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # 创建一个文件处理器，并设置级别为INFO
    file_handler = logging.FileHandler('train.log')
    file_handler.setLevel(logging.INFO)
    # 创建一个流处理器（控制台输出），并设置级别为INFO
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    # 将文件处理器和流处理器添加到logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # 配置选项
    opt = setargs()

    # 如果不存在 需要创建 checkpoint文件夹
    if not os.path.exists(opt.path):
        os.makedirs(opt.path)

    # 加载数据集
    train_loader = StructureLoader(dataPath=opt.dataPath,pictureSize=opt.pictureSize,ifGray=opt.ifGray,batch_size=opt.batch_size)
    val_loader = StructureLoader(dataPath=opt.dataPath,pictureSize=opt.pictureSize,ifGray=opt.ifGray,batch_size=opt.batch_size,shuffle=False)
    test_loader = StructureLoader(dataPath=opt.dataPath,pictureSize=opt.pictureSize,ifGray=opt.ifGray,batch_size=opt.batch_size,shuffle=False)

    # 选择设备
    paddle.set_device(opt.device)
    model = PPLCNET_x1_5(class_num=opt.class_num)
    # YoloBody(len(test_loader.dataset[0]), 'x')
    model.train()

    # 选择优化器
    optimizer = paddle.optimizer.Adam(learning_rate=opt.lr,parameters=model.parameters())
    # 学习率优化
    scheduler = paddle.optimizer.lr.ReduceOnPlateau(learning_rate=optimizer.get_lr() ,mode='min', factor=0.1, patience=5, verbose=True)
    # paddle 写法不一样，Optimizer 实例持有 Scheduler 实例
    optimizer.set_lr_scheduler(scheduler)

    # 看是否有checkpoint
    start_epoch = 0
    if opt.checkpoint is not None:
        start_epoch = load_model_optimizer(opt.checkpoint, model,opt.device)
        start_epoch += 1
        logger.info(start_epoch)

    start_train = datetime.datetime.now()
    logger.info("The current start time for running is: " + start_train.strftime("%Y-%m-%d %H:%M:%S"))
    start_train = time.time()
    epoch_losses_train, epoch_losses_test,epoch_losses_val = [], [],[]
    epoch_checkpoints = []
    epochbar = tqdm(total=opt.maxEpochs)
    for epoch in trange(start_epoch, opt.maxEpochs):

        train_sum, train_weights,train_acc,times = 0., 0.,0.,0.
        model.train()
        # 初始化进度条
        pbar = tqdm(total=int(np.ceil(len(train_loader)/train_loader.batch_size)))
        for batch_idx, batch in enumerate(train_loader):

            start_batch = datetime.datetime.now()
            logging.info("The current start time for running is:" + start_batch.strftime(
                "%Y-%m-%d %H:%M:%S") + "epoch：" + str(epoch) + "batch_idx：" + str(batch_idx))
            start_batch = time.time()

            # 制作正确的标签 # 将两幅图合成一幅
            labels = []
            for i,data in enumerate(batch):
                labels.append(data[2])
                # data.pop()
                batch[i] = np.concatenate([data[0],data[1]],axis=2)

            # 将图片数据转移到 opt.device 上
            batch = paddle.to_tensor(batch).to(device=opt.device,dtype='float32')

            # 应用dropout
            batch = F.dropout(batch, p=opt.dropout)
            # 将优化器的梯度设置为零，这意味着梯度将不更新，
            # 因为没有新的数据到来
            optimizer.clear_grad()
            # 进入模型
            predicts = model(batch)

            labels = paddle.to_tensor(labels)
            # 计算损失，取一个批次样本损失的平均值 (交叉熵损失函数)
            loss = F.cross_entropy(predicts, labels)
            log_probs = F.log_softmax(predicts, axis=-1)

            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            elapsed_featurize = time.time() - start_batch
            # 计算准确率
            acc = Accuracy_evaluation(log_probs,labels)
            # 累计损失
            train_sum += paddle.sum(loss).cpu().data.numpy()
            train_acc += acc
            times += 1
            # 更新进度条
            pbar.update(1)
            # 更新进度条，并设置后缀显示loss和acc
            pbar.set_postfix(train_loss=np.exp(train_sum/times), train_acc=str(train_acc/times)+"%")

        model.eval()
        with paddle.no_grad():
            val_sum, val_weights,val_acc,val_times = 0., 0.,0.,0.
            # 初始化进度条
            pbar = tqdm(total=int(np.ceil(len(val_loader) / val_loader.batch_size)))
            for batch_idx, batch in enumerate(val_loader):
                # 制作标签
                # 制作正确的标签
                labels = []
                for i, data in enumerate(batch):
                    labels.append(data[2])
                    # data.pop()
                    batch[i] = np.concatenate([data[0], data[1]], axis=2)

                # 将图片数据转移到 opt.device 上
                batch = paddle.to_tensor(batch).to(device=opt.device,dtype='float32')
                # forward()
                predicts = model(batch)
                labels = paddle.to_tensor(labels)
                # 计算损失，取一个批次样本损失的平均值 (交叉熵损失函数)
                loss = F.cross_entropy(predicts, labels)
                log_probs = F.log_softmax(predicts, axis=-1)
                # 计算准确率
                acc = Accuracy_evaluation(log_probs, labels)
                # Accumulate
                val_sum += paddle.sum(loss).cpu().data.numpy()
                val_acc += acc
                val_times += 1
                # 更新进度条
                pbar.update(1)
                # 更新进度条，并设置后缀显示loss和acc
                pbar.set_postfix(val_loss=np.exp(val_sum / val_times), val_acc=str(val_acc / val_times) + "%")

        train_loss = train_sum / times
        train_perplexity = np.exp(train_loss)
        val_loss = val_sum / val_times
        val_perplexity = np.exp(val_loss)

        logger.info("epoch: " + str(epoch) + " train acc" + ": " + str(train_acc/times))
        logger.info("epoch: " + str(epoch) + " train loss" + ": " + str(train_perplexity))
        logger.info("epoch: " + str(epoch) + " val acc" + ": " + str(val_acc/val_times))
        logger.info("epoch: " + str(epoch) + " val loss" + ": " + str(val_perplexity))

        scheduler.step(val_perplexity)

        # Save the model
        checkpoint_filename = os.path.join(opt.path , 'epoch{}.pt'.format(epoch + 1))
        paddle.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }, checkpoint_filename)