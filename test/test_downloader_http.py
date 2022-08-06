#!/usr/bin/env python3

# Allow direct execution
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import http.server
import re
import threading

from test.helper import http_server_port, try_rm
from test.test_networking import RequestHandlerTestBase, with_request_handlers
from yt_dlp.downloader.http import HttpFD
from yt_dlp.utils import encodeFilename

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


TEST_SIZE = 10 * 1024


class HTTPTestRequestHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def send_content_range(self, total=None):
        range_header = self.headers.get('Range')
        start = end = None
        if range_header:
            mobj = re.search(r'^bytes=(\d+)-(\d+)', range_header)
            if mobj:
                start = int(mobj.group(1))
                end = int(mobj.group(2))
        valid_range = start is not None and end is not None
        if valid_range:
            content_range = 'bytes %d-%d' % (start, end)
            if total:
                content_range += '/%d' % total
            self.send_header('Content-Range', content_range)
        return (end - start + 1) if valid_range else total

    def serve(self, range=True, content_length=True):
        self.send_response(200)
        self.send_header('Content-Type', 'video/mp4')
        size = TEST_SIZE
        if range:
            size = self.send_content_range(TEST_SIZE)
        if content_length:
            self.send_header('Content-Length', size)
        self.end_headers()
        self.wfile.write(b'#' * size)

    def do_GET(self):
        if self.path == '/regular':
            self.serve()
        elif self.path == '/no-content-length':
            self.serve(content_length=False)
        elif self.path == '/no-range':
            self.serve(range=False)
        elif self.path == '/no-range-no-content-length':
            self.serve(range=False, content_length=False)
        else:
            assert False


class FakeLogger:
    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        pass


class TestHttpFD(RequestHandlerTestBase, unittest.TestCase):
    def setUp(self):
        self.httpd = http.server.HTTPServer(
            ('127.0.0.1', 0), HTTPTestRequestHandler)
        self.port = http_server_port(self.httpd)
        self.server_thread = threading.Thread(target=self.httpd.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()

    def download(self, params, ep):
        params['logger'] = FakeLogger()
        ydl = self.make_ydl(params, fake=False)
        downloader = HttpFD(ydl, params)
        filename = 'testfile.mp4'
        try_rm(encodeFilename(filename))
        self.assertTrue(downloader.real_download(filename, {
            'url': 'http://127.0.0.1:%d/%s' % (self.port, ep),
        }), ep)
        self.assertEqual(os.path.getsize(encodeFilename(filename)), TEST_SIZE, ep)
        try_rm(encodeFilename(filename))

    def download_all(self, params):
        for ep in ('regular', 'no-content-length', 'no-range', 'no-range-no-content-length'):
            self.download(params, ep)

    @with_request_handlers()
    def test_regular(self):
        self.download_all({})

    @with_request_handlers()
    def test_chunked(self):
        self.download_all({
            'http_chunk_size': 1000,
        })


if __name__ == '__main__':
    unittest.main()
