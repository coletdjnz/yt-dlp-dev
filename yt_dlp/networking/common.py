from __future__ import unicode_literals
from __future__ import annotations

import collections
import email.policy
import http.cookiejar
import inspect
import io
import ssl
import sys
import time
import typing
import urllib.parse
from email.message import Message
from http import HTTPStatus
import urllib.request
import urllib.response
from typing import Union, Type

from ..compat import compat_cookiejar, compat_str

from ..utils import (
    extract_basic_auth,
    escape_url,
    sanitize_url,
    write_string,
    std_headers,
    update_url_query,
    bug_reports_message, TransportError, YoutubeDLError, RequestError, CaseInsensitiveDict
)

from .utils import random_user_agent

if typing.TYPE_CHECKING:
    from ..YoutubeDL import YoutubeDL


class Request:
    """
    Request class to define a request to be made.
    A wrapper for urllib.request.Request with improvements for yt-dlp,
    while retaining required backwards-compat functions used in yt-dlp.
    """
    def __init__(
            self, url, data=None, headers=None, proxies=None, compression=True, method=None, timeout=None):
        """
        @param proxies: proxy dict mapping to use for the request and any redirects
        @param compression: whether to include content-encoding header on request (i.e. disable/enable compression).
        """
        url, basic_auth_header = extract_basic_auth(escape_url(sanitize_url(url)))
        self.__request_store = urllib.request.Request(url, data=data, method=method)
        self._headers: CaseInsensitiveDict = CaseInsensitiveDict(headers)
        self.timeout = timeout

        # TODO: add support for passing different types of auth into a YDlRequest, and don't add the headers.
        if basic_auth_header:
            self.headers['Authorization'] = basic_auth_header

        self.proxies = proxies or {}
        self.compression = compression

    @property
    def url(self):
        return self.__request_store.full_url

    @url.setter
    def url(self, url):
        self.__request_store.full_url = url

    @property
    def data(self):
        return self.__request_store.data

    @data.setter
    def data(self, data):
        self.__request_store.data = data

    @property
    def headers(self):
        return self._headers

    @headers.setter
    def headers(self, new_headers):
        if not isinstance(new_headers, CaseInsensitiveDict):
            raise TypeError('headers must be CaseInsensitiveDict')
        self._headers = new_headers

    @property
    def method(self):
        return self.__request_store.get_method()

    def copy(self):
        return self.__class__(
            self.url, self.data, self.headers.copy(), self.proxies.copy(), self.compression, self.method)

    def add_header(self, key, value):
        self._headers[key] = value

    def get_header(self, key, default=None):
        return self._headers.get(key, default)

    def get_full_url(self):
        return self.url

    def get_method(self):
        return self.method

    @property
    def type(self):
        """URI scheme"""
        return self.__request_store.type

    @property
    def host(self):
        return self.__request_store.host


class HEADRequest(Request):
    @property
    def method(self):
        return 'HEAD'


class PUTRequest(Request):
    @property
    def method(self):
        return 'PUT'


def update_YDLRequest(req: Request, url=None, data=None, headers=None, query=None):
    """
    Replaces the old update_Request.
    TODO: do we want to replace this with a better method?
    """
    req = req.copy()
    req.data = data or req.data
    req.headers.update(headers or {})
    req.url = update_url_query(url or req.url, query or {})
    return req


class HTTPResponse(io.IOBase):
    """
    Adapter interface for HTTP responses
    """
    REDIRECT_STATUS_CODES = [301, 302, 303, 307, 308]

    def __init__(
            self, raw,
            headers: typing.Mapping[str, str],
            url: str,
            status: int = 200,
            reason: typing.Optional[str] = None):
        """
        @param raw: Original response
        @headers: response headers
        @status: Response HTTP status code
        @reason: HTTP status reason
        """
        self.raw = raw
        self.headers: Message = Message(policy=email.policy.HTTP)
        for name, value in (headers or {}).items():
            self.headers.add_header(name, value)
        self.code = self.status = status
        self.reason = reason
        self.url = url
        if not reason:
            try:
                self.reason = HTTPStatus(status).phrase
            except ValueError:
                pass

    # compat
    def getcode(self):
        return self.status

    # compat
    def getstatus(self):
        return self.status

    def geturl(self):
        return self.url

    def get_redirect_url(self):
        return self.getheader('location') if self.status in self.REDIRECT_STATUS_CODES else None

    def getheaders(self):
        return self.headers

    def getheader(self, name, default=None):
        return self.headers.get(name, default)

    def info(self):
        return self.headers

    def readable(self):
        return True

    def read(self, amt: int = None):
        return self.raw.read(amt)

    def tell(self) -> int:
        return self.raw.tell()

    def close(self):
        self.raw.close()
        return super().close()


class RequestHandler:
    """
    Bare-bones request handler.
    Use this for defining custom protocols for extractors.
    """
    SUPPORTED_SCHEMES: list = None

    @classmethod
    def _is_supported_scheme(cls, request: Request):
        return urllib.parse.urlparse(request.url).scheme.lower() in cls.SUPPORTED_SCHEMES or []

    def can_handle(self, request: Request) -> bool:
        """Validate if handler is suitable for given request. Can override in subclasses."""
        return self._is_supported_scheme(request)

    def handle(self, request: Request):
        """Method to handle given request. Redefine in subclasses"""


class BackendRH(RequestHandler):
    """Network Backend adapter class
    Responsible for handling requests.
    """

    def __init__(self, ydl: YoutubeDL):
        self.ydl = ydl
        self.cookiejar = self.ydl.cookiejar

    # TODO: rework
    def to_screen(self, *args, **kwargs):
        self.ydl.to_stdout(*args, **kwargs)

    def to_stderr(self, message):
        self.ydl.to_stderr(message)

    def report_warning(self, *args, **kwargs):
        self.ydl.report_warning(*args, **kwargs)

    def report_error(self, *args, **kwargs):
        self.ydl.report_error(*args, **kwargs)

    def write_debug(self, *args, **kwargs):
        self.ydl.write_debug(*args, **kwargs)

    def make_sslcontext(self, **kwargs):
        """
        Make a new SSLContext configured for this backend.
        Note: _make_sslcontext must be implemented
        """
        context = self._make_sslcontext(
            verify=not self.ydl.params.get('nocheckcertificate'), **kwargs)
        if not context:
            return context
        if self.ydl.params.get('legacyserverconnect'):
            context.options |= 4  # SSL_OP_LEGACY_SERVER_CONNECT
        return context

    def _make_sslcontext(self, verify: bool, **kwargs) -> ssl.SSLContext:
        """Generate a backend-specific SSLContext. Redefine in subclasses"""


class RHManager:

    def __init__(self, ydl: YoutubeDL):
        self.handlers = []
        self.ydl: YoutubeDL = ydl
        self.proxies: dict = self.get_default_proxies()

    def get_default_proxies(self) -> dict:
        proxies = urllib.request.getproxies() or {}
        # compat. Set HTTPS_PROXY to __noproxy__ to revert
        if 'http' in proxies and 'https' not in proxies:
            proxies['https'] = proxies['http']
        conf_proxy = self.ydl.params.get('proxy')
        if conf_proxy:
            # compat. We should ideally use all proxy here
            proxies.update({'http': conf_proxy, 'https': conf_proxy})
        return proxies

    def add_handler(self, handler: RequestHandler):
        if handler not in self.handlers:
            self.handlers.append(handler)

    def remove_handler(self, handler: Union[RequestHandler, Type[RequestHandler]]):
        """
        Remove RequestHandler(s)
        @param handler: Handler object or handler type. Specifying handler type will remove all handlers of that type.
        """
        if inspect.isclass(handler):
            finder = lambda x: isinstance(x, handler)
        else:
            finder = lambda x: x is handler
        self.handlers = [x for x in self.handlers if not finder(handler)]

    def urlopen(self, req: Union[Request, str, urllib.request.Request]) -> HTTPResponse:
        """
        Passes a request onto a suitable RequestHandler
        """
        if len(self.handlers) == 0:
            raise YoutubeDLError('No request handlers configured')
        if isinstance(req, str):
            req = Request(req)
        elif isinstance(req, urllib.request.Request):
            # compat
            req = Request(
                req.get_full_url(), data=req.data, method=req.get_method(),
                headers=CaseInsensitiveDict(req.headers, req.unredirected_hdrs))

        assert isinstance(req, Request)

        req = req.copy()
        req.headers = CaseInsensitiveDict(self.ydl.params.get('http_headers', {}), req.headers)

        if req.headers.get('Youtubedl-no-compression'):
            req.compression = False
            del req.headers['Youtubedl-no-compression']

        # Proxy preference: header req proxy > req proxies > ydl opt proxies > env proxies
        req.proxies = {**(self.proxies or {}), **(req.proxies or {})}
        req_proxy = req.headers.get('Ytdl-request-proxy')
        if req_proxy:
            del req.headers['Ytdl-request-proxy']
            req.proxies.update({'http': req_proxy, 'https': req_proxy})
        for k, v in req.proxies.items():
            if v == '__noproxy__':  # compat
                req.proxies[k] = None
        req.timeout = float(req.timeout or self.ydl.params.get('socket_timeout') or 20)  # do not accept 0

        for handler in reversed(self.handlers):
            if not handler.can_handle(req):
                continue
            try:
                if self.ydl.params.get('debug_printtraffic'):
                    self.ydl.to_stdout(f'Forwarding request to {type(handler).__name__} request handler')
                res = handler.handle(req)
            except Exception as e:
                if not isinstance(e, YoutubeDLError):
                    self.ydl.report_warning(f'Unexpected error from request handler: {type(e).__name__}: {e}' + bug_reports_message())

                if isinstance(e, RequestError):
                    e.handler = handler
                raise

            if not res:
                self.ydl.report_warning(f'{type(handler).__name__} request handler returned nothing for response' + bug_reports_message())
                continue
            assert isinstance(res, HTTPResponse)
            return res
        raise YoutubeDLError('No request handlers configured that could handle this request.')


class YoutubeDLCookieJar(compat_cookiejar.MozillaCookieJar):
    """
    See [1] for cookie file format.
    1. https://curl.haxx.se/docs/http-cookies.html
    """
    _HTTPONLY_PREFIX = '#HttpOnly_'
    _ENTRY_LEN = 7
    _HEADER = '''# Netscape HTTP Cookie File
# This file is generated by yt-dlp.  Do not edit.
'''
    _CookieFileEntry = collections.namedtuple(
        'CookieFileEntry',
        ('domain_name', 'include_subdomains', 'path', 'https_only', 'expires_at', 'name', 'value'))

    def save(self, filename=None, ignore_discard=False, ignore_expires=False):
        """
        Save cookies to a file.
        Most of the code is taken from CPython 3.8 and slightly adapted
        to support cookie files with UTF-8 in both python 2 and 3.
        """
        if filename is None:
            if self.filename is not None:
                filename = self.filename
            else:
                raise ValueError(compat_cookiejar.MISSING_FILENAME_TEXT)

        # Store session cookies with `expires` set to 0 instead of an empty
        # string
        for cookie in self:
            if cookie.expires is None:
                cookie.expires = 0

        with io.open(filename, 'w', encoding='utf-8') as f:
            f.write(self._HEADER)
            now = time.time()
            for cookie in self:
                if not ignore_discard and cookie.discard:
                    continue
                if not ignore_expires and cookie.is_expired(now):
                    continue
                if cookie.secure:
                    secure = 'TRUE'
                else:
                    secure = 'FALSE'
                if cookie.domain.startswith('.'):
                    initial_dot = 'TRUE'
                else:
                    initial_dot = 'FALSE'
                if cookie.expires is not None:
                    expires = compat_str(cookie.expires)
                else:
                    expires = ''
                if cookie.value is None:
                    # cookies.txt regards 'Set-Cookie: foo' as a cookie
                    # with no name, whereas http.cookiejar regards it as a
                    # cookie with no value.
                    name = ''
                    value = cookie.name
                else:
                    name = cookie.name
                    value = cookie.value
                f.write(
                    '\t'.join([cookie.domain, initial_dot, cookie.path,
                               secure, expires, name, value]) + '\n')

    def load(self, filename=None, ignore_discard=False, ignore_expires=False):
        """Load cookies from a file."""
        if filename is None:
            if self.filename is not None:
                filename = self.filename
            else:
                raise ValueError(compat_cookiejar.MISSING_FILENAME_TEXT)

        def prepare_line(line):
            if line.startswith(self._HTTPONLY_PREFIX):
                line = line[len(self._HTTPONLY_PREFIX):]
            # comments and empty lines are fine
            if line.startswith('#') or not line.strip():
                return line
            cookie_list = line.split('\t')
            if len(cookie_list) != self._ENTRY_LEN:
                raise compat_cookiejar.LoadError('invalid length %d' % len(cookie_list))
            cookie = self._CookieFileEntry(*cookie_list)
            if cookie.expires_at and not cookie.expires_at.isdigit():
                raise compat_cookiejar.LoadError('invalid expires at %s' % cookie.expires_at)
            return line

        cf = io.StringIO()
        with io.open(filename, encoding='utf-8') as f:
            for line in f:
                try:
                    cf.write(prepare_line(line))
                except compat_cookiejar.LoadError as e:
                    write_string(
                        'WARNING: skipping cookie file entry due to %s: %r\n'
                        % (e, line), sys.stderr)
                    continue
        cf.seek(0)
        self._really_load(cf, filename, ignore_discard, ignore_expires)
        # Session cookies are denoted by either `expires` field set to
        # an empty string or 0. MozillaCookieJar only recognizes the former
        # (see [1]). So we need force the latter to be recognized as session
        # cookies on our own.
        # Session cookies may be important for cookies-based authentication,
        # e.g. usually, when user does not check 'Remember me' check box while
        # logging in on a site, some important cookies are stored as session
        # cookies so that not recognizing them will result in failed login.
        # 1. https://bugs.python.org/issue17164
        for cookie in self:
            # Treat `expires=0` cookies as session cookies
            if cookie.expires == 0:
                cookie.expires = None
                cookie.discard = True


# Use make_std_headers() to get a copy of these
_std_headers = CaseInsensitiveDict({
    'User-Agent': random_user_agent(),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-us,en;q=0.5',
    'Sec-Fetch-Mode': 'navigate',
})


# Get a copy of std headers, while also retaining backwards compat with utils.std_headers
def make_std_headers():
    return CaseInsensitiveDict(_std_headers, std_headers)
