# -*- coding: utf-8 -*-
import time

import ezdxf

start_time = time.time()
dxf_path = "test.dxf"
try:
    dxf = ezdxf.readfile(dxf_path)
except Exception as e:
    print(f"DXF路径：{dxf_path} 读取DXF异常:{e}")
    raise e
finally:
    print(f"读取时间:{time.time() - start_time}")

