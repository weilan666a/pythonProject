import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image

# 设置图片文件夹路径和新的文件夹路径
src_dir = '/path/to/your/images'
train_dir = 'train'
val_dir = 'val'
test_dir = 'test'

# 创建新的文件夹
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 获取所有图片文件
images = [os.path.join(src_dir, f) for f in os.listdir(src_dir) if f.endswith(('.jpg', '.png'))]

# 打乱图片顺序
np.random.shuffle(images)

# 切分数据集
train_images, temp_images = train_test_split(images, test_size=0.2, random_state=42)
val_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)

# 定义复制图片的函数
def copy_images(image_list, dest_dir):
    for image_path in image_list:
        shutil.copy(image_path, dest_dir)

# 复制图片到新的文件夹
copy_images(train_images, train_dir)
copy_images(val_images, val_dir)
copy_images(test_images, test_dir)

print(f"Copied {len(train_images)} images to {train_dir}")
print(f"Copied {len(val_images)} images to {val_dir}")
print(f"Copied {len(test_images)} images to {test_dir}")