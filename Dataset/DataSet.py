# 加载相关库
import logging
import os,random,paddle,gzip,json,glob
from paddle.vision.transforms import Resize
import numpy as np
from PIL import Image


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def read_picture(img_path,transform,ifGray):
    # 读取图片
    img = Image.open(img_path)
    img_np = transform(img)
    # 转换颜色通道顺序 # 针对灰度图片
    if ifGray:
        # 灰度化图片
        img_gray = img.convert('L')
    # if img_np.ndim == 2:
        img_np = img_gray[:, :, np.newaxis]
    # 转换为NumPy数组
    img_np = np.array(img_np)
    # Paddle期望输入数据的颜色通道顺序为CHW（Channel, Height, Width），
    # PIL和NumPy数组默认的顺序是HWC（Height, Width, Channel）
    img_np = img_np.transpose((2, 0, 1))  # 从HWC转换为CHW

    return img_np

# 定义数据集读取器 {picture_category}:{0 or 1}
# dataPath 图片路径
# pictureSize 图片大小
def load_data(dataPath="../train",pictureSize=224,ifGray=False):
    # 图像转换大小
    transform = Resize(size=[pictureSize,pictureSize])

    # 获取文件夹中所有的.jpg和.png图片路径
    pictureFiles = glob.glob(os.path.join(dataPath + "/Files/", '*.jpg')) + \
                   glob.glob(os.path.join(dataPath + "/Files/", '*.png'))
    # 获得分类照片
    CategoryImages = glob.glob(os.path.join(dataPath + "/Retrieval/", '*.jpg')) + \
                     glob.glob(os.path.join(dataPath + "/Files/", '*.png'))
    # 读取照片，转换成列表 (已经灰度化和调整大小)
    categoryFiles = []
    categoryNames = []
    for categoryFile in CategoryImages:
        # 获得最后一部分的图片名字
        lastIndex = categoryFile.rfind('\\')
        suffix = categoryFile.rfind('.')
        categoryName = categoryFile[lastIndex + 1:suffix]
        categoryFiles.append(categoryFile)
        categoryNames.append(categoryName)

    data = []
    for pictureFile in pictureFiles:
        # 获得最后一部分的图片名字
        lastIndex = pictureFile.rfind('/')
        suffix = pictureFile.rfind('.')
        pictureName = pictureFile[lastIndex + 1:suffix]
        # 去文件夹中找对应的图片
        jsonName = dataPath+'/Annotations/'+pictureName + '.json'
        with open(jsonName, 'r') as f:
            jsonData = f.readlines()
             # jsonData = json.load(f)
        for categoryName,categoryFile in zip(categoryNames,categoryFiles):
            # 单项数据
            singleItem = []
            singleItem.append(read_picture(pictureFile, transform,ifGray))
            singleItem.append(read_picture(categoryFile, transform,ifGray))
            jsonCategoryName = jsonData[2][jsonData[2].find("\"")+1 : jsonData[2].rfind("\"")]
            if categoryName == jsonCategoryName:
                singleItem.append(1)
            else:
                singleItem.append(0)
            data.append(singleItem)
    # 书写日志
    logger.info(f"{dataPath} Number of pictureFile {len(data)}")
    return data

# 定义DataLoader
class StructureLoader():
    def __init__(self, dataPath="../train",pictureSize=224,batch_size = 25,ifGray=False,shuffle=True):
        self.dataset = load_data(dataPath,pictureSize,ifGray=ifGray)
        self.shuffle = shuffle
        self.batch_size = batch_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __iter__(self,):
        # 遍历数据集并产生批次
        for i in range(0, len(self.dataset), self.batch_size):
            batch = self.dataset[i:i + self.batch_size]
            if self.shuffle:
                np.random.shuffle(batch)
            yield batch
