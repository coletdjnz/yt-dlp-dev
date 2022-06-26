from __future__ import annotations

import io
import itertools
import ssl
import typing
import urllib.parse
import urllib.request
import urllib.response
from email.message import Message
from http import HTTPStatus
from typing import Union

from .. import utils

try:
    from urllib.request import _parse_proxy
except ImportError:
    _parse_proxy = None

from ..utils import (
    CaseInsensitiveDict,
    RequestError,
    SSLError,
    UnsupportedRequest,
    YoutubeDLError,
    bug_reports_message,
    escape_url,
    extract_basic_auth,
    remove_start,
    sanitize_url,
    update_url_query,
)

if typing.TYPE_CHECKING:
    from ..YoutubeDL import YoutubeDL


class Request:
    """
    Represents a request to be made.
    Partially backwards-compatible with urllib.request.Request.

    @param url: url to send. Will be sanitized and auth will be extracted as basic auth if present.
    @param data: payload data to send.
    @param headers: headers to send.
    @param proxies: proxy dict mapping of proto:proxy to use for the request and any redirects.
    @param query: URL query parameters to update the url with.
    @param method: HTTP method to use. If no method specified, will use POST if payload data is present else GET
    @param compression: whether to include content-encoding header on request.
    @param redirect: whether to follow redirects for this request.
    @param timeout: socket timeout value for this request.
    """

    def __init__(
            self,
            url: str,
            data=None,
            headers: typing.Mapping = None,
            proxies: dict = None,
            query: dict = None,
            method: str = None,
            compression: bool = True,
            redirect: bool = True,
            timeout: Union[float, int] = None):

        url, basic_auth_header = extract_basic_auth(escape_url(sanitize_url(url)))

        if query:
            url = update_url_query(url, query)
        # rely on urllib Request's url parsing
        self.__request_store = urllib.request.Request(url)
        self.method = method
        self._headers = CaseInsensitiveDict(headers)
        self._data = None
        self.data = data
        self.timeout = timeout
        self.redirect = redirect

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
        return self._data

    @data.setter
    def data(self, data):
        # https://docs.python.org/3/library/urllib.request.html#urllib.request.Request.data
        if data != self._data:
            self._data = data
            if 'content-length' in self.headers:
                del self.headers['content-length']

    @property
    def headers(self) -> CaseInsensitiveDict:
        return self._headers

    @headers.setter
    def headers(self, new_headers: CaseInsensitiveDict):
        if not isinstance(new_headers, CaseInsensitiveDict):
            raise TypeError('headers must be a CaseInsensitiveDict')
        self._headers = new_headers

    @property
    def method(self):
        return self.__method or ('POST' if self.data is not None else 'GET')

    @method.setter
    def method(self, method: str):
        self.__method = method

    def copy(self):
        return type(self)(
            url=self.url, data=self.data, headers=self.headers.copy(), timeout=self.timeout,
            proxies=self.proxies.copy(), compression=self.compression, method=self.method, redirect=self.redirect)

    def add_header(self, key, value):
        self._headers[key] = value

    def get_header(self, key, default=None):
        return self._headers.get(key, default)

    @property
    def type(self):
        """URI scheme"""
        return self.__request_store.type

    @property
    def host(self):
        return self.__request_store.host

    # The following methods are for compatability reasons and are deprecated
    @property
    def fullurl(self):
        """Deprecated, use Request.url"""
        return self.url

    @fullurl.setter
    def fullurl(self, url):
        """Deprecated, use Request.url"""
        self.url = url

    def get_full_url(self):
        """Deprecated, use Request.url"""
        return self.url

    def get_method(self):
        """Deprecated, use Request.method"""
        return self.method

    def has_header(self, name):
        """Deprecated, use `name in Request.headers`"""
        return name in self.headers


class HEADRequest(Request):
    @property
    def method(self):
        return 'HEAD'


class PUTRequest(Request):
    @property
    def method(self):
        return 'PUT'


class Response(io.IOBase):
    """
    Abstract base class for HTTP response adapters.

    Interface partially backwards-compatible with addinfourl and http.client.HTTPResponse.

    @param raw: Original response.
    @param url: URL that this is a response of.
    @param headers: response headers.
    @param status: Response HTTP status code. Default is 200 OK.
    @param reason: HTTP status reason. Will use built-in reasons based on status code if not provided.
    """
    REDIRECT_STATUS_CODES = [301, 302, 303, 307, 308]

    def __init__(
            self, raw,
            url: str,
            headers: typing.Mapping[str, str],
            status: int = 200,
            reason: typing.Optional[str] = None):

        self.raw = raw
        self.headers: Message = Message()
        for name, value in (headers or {}).items():
            self.headers.add_header(name, value)
        self.status = status
        self.reason = reason
        self.url = url
        if not reason:
            try:
                self.reason = HTTPStatus(status).phrase
            except ValueError:
                pass

    def get_redirect_url(self):
        return self.headers.get('location') if self.status in self.REDIRECT_STATUS_CODES else None

    def readable(self):
        return True

    def read(self, amt: int = None):
        return self.raw.read(amt)

    def tell(self) -> int:
        return self.raw.tell()

    def close(self):
        self.raw.close()
        return super().close()

    # The following methods are for compatability reasons and are deprecated
    @property
    def code(self):
        """Deprecated, use HTTPResponse.status"""
        return self.status

    def getstatus(self):
        """Deprecated, use HTTPResponse.status"""
        return self.status

    def geturl(self):
        """Deprecated, use HTTPResponse.url"""
        return self.url

    def info(self):
        """Deprecated, use HTTPResponse.headers"""
        return self.headers


class RequestHandler:

    SUPPORTED_SCHEMES: list = None

    def __init__(self, ydl: YoutubeDL):
        self._set_ydl(ydl)
        self.cookiejar = self.ydl.cookiejar

    def _set_ydl(self, ydl):
        # TODO: this sucks
        self.ydl = ydl

        for func in (
            'deprecation_warning',
            'report_warning',
            'to_stderr',
            'write_debug',
            'to_debugtraffic'
        ):
            if not hasattr(self, func):
                setattr(self, func, getattr(ydl, func))

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
            # Allow use of weaker ciphers in Python 3.10+. See https://bugs.python.org/issue43998
            context.set_ciphers('DEFAULT')

        client_certfile = self.ydl.params.get('client_certificate')
        if client_certfile:
            try:
                context.load_cert_chain(
                    client_certfile, keyfile=self.ydl.params.get('client_certificate_key'),
                    password=self.ydl.params.get('client_certificate_password'))
            except ssl.SSLError:
                raise YoutubeDLError('Unable to load client certificate')

        return context

    def _make_sslcontext(self, verify: bool, **kwargs) -> ssl.SSLContext:
        """Generate a backend-specific SSLContext. Redefine in subclasses"""
        raise NotImplementedError

    def _check_scheme(self, request: Request):
        scheme = urllib.parse.urlparse(request.url).scheme.lower()
        if scheme == 'file':  # no other handler should handle this request
            raise RequestError('file:// scheme is explicitly disabled in yt-dlp for security reasons')

        if scheme not in self.SUPPORTED_SCHEMES:
            raise UnsupportedRequest(f'{scheme} scheme is not supported')

    def prepare_request(self, request: Request):
        self._check_scheme(request)
        request.headers = CaseInsensitiveDict(self.ydl.params.get('http_headers', {}), request.headers)
        if request.headers.get('Youtubedl-no-compression'):
            request.compression = False
            del request.headers['Youtubedl-no-compression']

        # Proxy preference: header req proxy > req proxies > ydl opt proxies > env proxies
        request.proxies = {**(self.ydl.proxies or {}), **(request.proxies or {})}
        req_proxy = request.headers.get('Ytdl-request-proxy')
        if req_proxy:
            del request.headers['Ytdl-request-proxy']
            request.proxies.update({'http': req_proxy, 'https': req_proxy})
        for proxy_key, proxy_url in request.proxies.items():
            if proxy_url == '__noproxy__':  # compat
                request.proxies[proxy_key] = None
                continue

            if proxy_url is not None and _parse_proxy is not None:
                # Ensure proxies without a scheme are http.
                proxy_scheme = _parse_proxy(proxy_url)[0]
                if proxy_scheme is None:
                    request.proxies[proxy_key] = 'http://' + remove_start(proxy_url, '//')

        request.timeout = float(request.timeout or self.ydl.params.get('socket_timeout') or 20)  # do not accept 0
        return self._prepare_request(request)

    def _prepare_request(self, request: Request):
        """Prepare a request for this handler. Redefine in subclasses."""
        return request

    def handle(self, request: Request):
        try:
            request = self.prepare_request(request)
            return self._real_handle(request)
        except RequestError as e:
            e.handler = self
            raise

    def _real_handle(self, request: Request):
        """Handle a request. Redefine in subclasses."""
        raise NotImplementedError

    def close(self):
        pass

    @utils.classproperty
    def NAME(cls):
        return cls.__name__


class RequestDirector:

    def __init__(self, ydl):
        self._handlers = []
        self.ydl = ydl

    def close(self):
        for handler in self._handlers:
            handler.close()

    def add_handler(self, handler):
        if handler not in self._handlers and isinstance(handler, RequestHandler):
            self._handlers.append(handler)

    def remove_handler(self, handler):
        """
        Remove a RequestHandler from the broker.
        If a class is provided, all handlers of that class type are removed.
        """
        self._handlers = [h for h in self._handlers if not (isinstance(h, handler) or h is handler)]

    def get_handlers(self, handler=None):
        """Get all handlers for a particular class type"""
        return [h for h in self._handlers if isinstance(h, handler or RequestHandler)]

    def replace_handler(self, handler):
        self.remove_handler(handler)
        self.add_handler(handler)

    def send(self, request: Union[Request, str, urllib.request.Request]) -> Response:
        """
        Passes a request onto a suitable RequestHandler
        """
        if len(self._handlers) == 0:
            raise RequestError('No request handlers configured')
        if isinstance(request, str):
            request = Request(request)
        elif isinstance(request, urllib.request.Request):
            # compat
            request = Request(
                request.get_full_url(), data=request.data, method=request.get_method(),
                headers=CaseInsensitiveDict(request.headers, request.unredirected_hdrs))

        assert isinstance(request, Request)

        unexpected_errors = []
        unsupported_errors = []
        for handler in reversed(self._handlers):
            handler_req = request.copy()
            try:
                self.ydl.to_debugtraffic(f'Forwarding request to "{handler.NAME}" request handler')
                response = handler.handle(handler_req)

            except UnsupportedRequest as e:
                self.ydl.to_debugtraffic(
                    f'"{handler.NAME}" request handler cannot handle this request, trying another handler... (cause: {type(e).__name__}:{e})')
                unsupported_errors.append(e)
                continue

            except Exception as e:
                if isinstance(e, RequestError):
                    raise
                # something went very wrong, try fallback to next handler
                self.ydl.report_error(
                    f'Unexpected error from "{handler.NAME}" request handler: {e}' + bug_reports_message(),
                    is_error=False)
                unexpected_errors.append(e)
                continue

            if not response:
                self.ydl.report_warning(
                    f'{handler.NAME} request handler returned nothing for response, trying another handler...' + bug_reports_message())
                continue

            assert isinstance(response, Response)
            return response

        # no handler was able to handle the request, try print some useful info
        # FIXME: this is a bit ugly
        err_handler_map = {}
        for err in unsupported_errors:
            err_handler_map.setdefault(err.msg, []).append(err.handler.NAME)

        reasons = [f'{msg} ({", ".join(handlers)})' for msg, handlers in err_handler_map.items()]
        if unexpected_errors:
            reasons.append(f'{len(unexpected_errors)} unexpected error(s)')

        err_str = 'Unable to handle request'
        if reasons:
            err_str += f', possible reason(s): ' + ', '.join(reasons)

        raise RequestError(err_str)
