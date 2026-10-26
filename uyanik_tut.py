"""Gece işleri için bilgisayarı uyku moduna girmekten alıkoyar (sistem ayarı değiştirmez; süreç bitince biter).

Kullanım: python uyanik_tut.py [saat]   (varsayılan 16)
"""
import ctypes
import sys
import time

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
end = time.time() + float(sys.argv[1] if len(sys.argv) > 1 else 16) * 3600
while time.time() < end:
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
    time.sleep(60)
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
