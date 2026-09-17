# -*- coding: utf-8 -*-
"""安全回收物理内存：对全部进程调用 EmptyWorkingSet（不杀进程，仅将工作集写回页文件）"""
import ctypes
import psutil

psapi = ctypes.WinDLL("psapi.dll")
kernel32 = ctypes.WinDLL("kernel32.dll")
PROCESS_SET_QUOTA = 0x0100
PROCESS_QUERY_INFORMATION = 0x0400

计数 = 0
for p in psutil.process_iter():
    try:
        h = kernel32.OpenProcess(PROCESS_SET_QUOTA | PROCESS_QUERY_INFORMATION, False, p.pid)
        if h:
            if psapi.EmptyWorkingSet(h):
                计数 += 1
            kernel32.CloseHandle(h)
    except Exception:
        pass

import gc
gc.collect()
m = psutil.virtual_memory()
print(f"已回收 {计数} 个进程工作集，可用内存：{round(m.available/1e9,2)} GB", flush=True)
