import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import requests
from studio_downloads import ensure_download, safetensor_valid
from studio_storage import atomic_json

class Response:
    def __init__(self, data, status=200, headers=None):
        self.data=data; self.status_code=status; self.headers=headers or {'Content-Length':str(len(data))}
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def raise_for_status(self):
        if self.status_code>=400: raise requests.HTTPError('test')
    def iter_content(self,size): yield self.data

class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.dest=Path(self.tmp.name)/'model.safetensors'
        header=json.dumps({'weight':{'dtype':'F32','shape':[1],'data_offsets':[0,4]}}).encode()
        self.data=len(header).to_bytes(8,'little')+header+b'1234'
        self.sha=hashlib.sha256(self.data).hexdigest()
    def test_verified_download_and_cache(self):
        with patch('studio_downloads.requests.get',return_value=Response(self.data)) as get:
            ensure_download('https://example.com/model',self.dest,sha256=self.sha)
            ensure_download('https://example.com/model',self.dest,sha256=self.sha)
            self.assertEqual(get.call_count,1); self.assertTrue(safetensor_valid(self.dest))
    def test_resume_range(self):
        partial=self.dest.with_suffix('.safetensors.part'); partial.write_bytes(self.data[:10])
        atomic_json(partial.with_suffix('.part.json'),{'url':'https://example.com/model'})
        with patch('studio_downloads.requests.get',return_value=Response(self.data[10:],206,{'Content-Range':f'bytes 10-{len(self.data)-1}/{len(self.data)}'})) as get:
            ensure_download('https://example.com/model',self.dest,sha256=self.sha)
            self.assertEqual(get.call_args.kwargs['headers']['Range'],'bytes=10-')
            self.assertEqual(self.dest.read_bytes(),self.data)
    def test_bad_hash_never_promoted(self):
        with patch('studio_downloads.requests.get',return_value=Response(self.data)):
            with self.assertRaises(RuntimeError): ensure_download('https://example.com/model',self.dest,sha256='0'*64,retries=1)
        self.assertFalse(self.dest.exists())
    def test_cancel_never_promoted(self):
        with self.assertRaises(InterruptedError): ensure_download('https://example.com/model',self.dest,cancelled=lambda:True)
        self.assertFalse(self.dest.exists())
    def test_invalid_tensor_layout(self):
        self.dest.write_bytes(b'not a tensor'); self.assertFalse(safetensor_valid(self.dest))
