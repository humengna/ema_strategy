# -*- coding: utf-8 -*-
"""结果落盘的容错处理。

回测动辄跑几十分钟,不能因为目标 CSV 被 Excel 占着(Windows 上 [Errno 13]
Permission denied)就把全部计算成果丢掉。这里的写入永不抛异常:
目标路径写不了就自动换一个带时间戳的名字,并告诉用户换到了哪。

配套的调用约定:**先打印报告,再写文件**。报告是主要产出,
不该排在一个可能失败的 IO 之后。
"""
from __future__ import annotations

import os
import sys
import time

import pandas as pd


def safe_to_csv(df: pd.DataFrame, path: str, index: bool = False,
                encoding: str = "utf-8-sig") -> str | None:
    """写 CSV;目标被占用或无权限时自动换名重试。

    返回实际写入的路径;彻底失败返回 None(不抛异常)。
    """
    try:
        df.to_csv(path, index=index, encoding=encoding)
        return path
    except (PermissionError, OSError) as exc:
        stem, ext = os.path.splitext(path)
        alt = f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}{ext or '.csv'}"
        print(f"  [!] 写入 {path} 失败({type(exc).__name__}: {exc})", file=sys.stderr)
        print(f"      该文件可能正被 Excel 等程序占用;改写到 {alt}", file=sys.stderr)
        try:
            df.to_csv(alt, index=index, encoding=encoding)
            return alt
        except (PermissionError, OSError) as exc2:
            print(f"  [!] 备用路径同样失败:{type(exc2).__name__}: {exc2}", file=sys.stderr)
            print(f"      结果未能落盘,但上方报告已完整输出。", file=sys.stderr)
            return None
