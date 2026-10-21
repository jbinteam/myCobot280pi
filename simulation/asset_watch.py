"""Nonblocking Linux asset-change revisions; unavailable watchers fail closed.

Directory watches catch in-place writes, timestamp changes, atomic replacement
and symlink replacement. If watching fails or events overflow, callers must
rehash content instead of reusing cached fingerprints.
"""
import atexit
import ctypes
import os
from pathlib import Path
import struct
import sys


class AssetWatch:
    MASK=0x2|0x4|0x8|0x40|0x80|0x100|0x200|0x400|0x800
    OVERFLOW=0x4000
    IGNORED=0x8000

    def __init__(self):
        self.fd=None;self.revision_id=0;self.directories={}
        if not sys.platform.startswith('linux'):return
        try:
            self.libc=ctypes.CDLL(None,use_errno=True)
            self.libc.inotify_init1.argtypes=[ctypes.c_int]
            self.libc.inotify_init1.restype=ctypes.c_int
            self.libc.inotify_add_watch.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_uint32]
            self.libc.inotify_add_watch.restype=ctypes.c_int
            fd=self.libc.inotify_init1(os.O_NONBLOCK|os.O_CLOEXEC)
            if fd<0:return
            self.fd=fd;atexit.register(self.close)
        except (OSError,AttributeError):self.close()

    def close(self):
        if self.fd is not None:
            os.close(self.fd);self.fd=None
        self.directories.clear()

    def revision(self,paths):
        if self.fd is None:return None
        try:
            for path in {Path(p).absolute().parent for p in paths}:
                stat=path.stat();identity=(stat.st_dev,stat.st_ino)
                previous=self.directories.get(path)
                if previous is not None and previous[1]==identity:continue
                wd=self.libc.inotify_add_watch(self.fd,os.fsencode(path),self.MASK)
                if wd<0:
                    self.close();return None
                self.directories[path]=(wd,identity)
            while True:
                try:data=os.read(self.fd,65536)
                except BlockingIOError:break
                if not data:
                    self.close();return None
                offset=0
                while offset<len(data):
                    wd,mask,_,length=struct.unpack_from('iIII',data,offset)
                    offset+=16+length
                    if mask&self.OVERFLOW:
                        self.close();return None
                    if mask&self.IGNORED:
                        self.directories={path:value for path,value in self.directories.items() if value[0]!=wd}
                    self.revision_id+=1
            return self.revision_id
        except (OSError,ValueError,struct.error):
            self.close();return None
