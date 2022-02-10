#!/usr/bin/env python3
# coding: utf-8

from __future__ import unicode_literals

import base64
import binascii
import calendar
import codecs
import collections
import contextlib
import ctypes
import datetime
import email.utils
import email.header
import errno
import functools
import hashlib
import hmac
import importlib.util
import itertools
import json
import locale
import math
import operator
import os
import platform
import random
import re
import io
import socket
import subprocess
import sys
import tempfile
import xml.etree.ElementTree
import mimetypes

from .compat import (
    compat_HTMLParseError,
    compat_HTMLParser,
    compat_chr,
    compat_etree_fromstring,
    compat_expanduser,
    compat_html_entities,
    compat_html_entities_html5,
    compat_os_name,
    compat_parse_qs,
    compat_shlex_split,
    compat_shlex_quote,
    compat_str,
    compat_struct_pack,
    compat_struct_unpack,
    compat_urllib_parse,
    compat_urllib_parse_urlencode,
    compat_urllib_parse_urlparse,
    compat_urllib_parse_urlunparse,
    compat_urllib_parse_quote,
    compat_urllib_parse_quote_plus,
    compat_urllib_request,
    compat_urlparse,
)

std_headers = {}

USER_AGENTS = {
    'Safari': 'Mozilla/5.0 (X11; Linux x86_64; rv:10.0) AppleWebKit/533.20.25 (KHTML, like Gecko) Version/5.0.4 Safari/533.20.27',
}


NO_DEFAULT = object()

ENGLISH_MONTH_NAMES = [
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December']

MONTH_NAMES = {
    'en': ENGLISH_MONTH_NAMES,
    'fr': [
        'janvier', 'février', 'mars', 'avril', 'mai', 'juin',
        'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre'],
}

KNOWN_EXTENSIONS = (
    'mp4', 'm4a', 'm4p', 'm4b', 'm4r', 'm4v', 'aac',
    'flv', 'f4v', 'f4a', 'f4b',
    'webm', 'ogg', 'ogv', 'oga', 'ogx', 'spx', 'opus',
    'mkv', 'mka', 'mk3d',
    'avi', 'divx',
    'mov',
    'asf', 'wmv', 'wma',
    '3gp', '3g2',
    'mp3',
    'flac',
    'ape',
    'wav',
    'f4f', 'f4m', 'm3u8', 'smil')

# needed for sanitizing filenames in restricted mode
ACCENT_CHARS = dict(zip('ÂÃÄÀÁÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖŐØŒÙÚÛÜŰÝÞßàáâãäåæçèéêëìíîïðñòóôõöőøœùúûüűýþÿ',
                        itertools.chain('AAAAAA', ['AE'], 'CEEEEIIIIDNOOOOOOO', ['OE'], 'UUUUUY', ['TH', 'ss'],
                                        'aaaaaa', ['ae'], 'ceeeeiiiionooooooo', ['oe'], 'uuuuuy', ['th'], 'y')))

DATE_FORMATS = (
    '%d %B %Y',
    '%d %b %Y',
    '%B %d %Y',
    '%B %dst %Y',
    '%B %dnd %Y',
    '%B %drd %Y',
    '%B %dth %Y',
    '%b %d %Y',
    '%b %dst %Y',
    '%b %dnd %Y',
    '%b %drd %Y',
    '%b %dth %Y',
    '%b %dst %Y %I:%M',
    '%b %dnd %Y %I:%M',
    '%b %drd %Y %I:%M',
    '%b %dth %Y %I:%M',
    '%Y %m %d',
    '%Y-%m-%d',
    '%Y.%m.%d.',
    '%Y/%m/%d',
    '%Y/%m/%d %H:%M',
    '%Y/%m/%d %H:%M:%S',
    '%Y%m%d%H%M',
    '%Y%m%d%H%M%S',
    '%Y%m%d',
    '%Y-%m-%d %H:%M',
    '%Y-%m-%d %H:%M:%S',
    '%Y-%m-%d %H:%M:%S.%f',
    '%Y-%m-%d %H:%M:%S:%f',
    '%d.%m.%Y %H:%M',
    '%d.%m.%Y %H.%M',
    '%Y-%m-%dT%H:%M:%SZ',
    '%Y-%m-%dT%H:%M:%S.%fZ',
    '%Y-%m-%dT%H:%M:%S.%f0Z',
    '%Y-%m-%dT%H:%M:%S',
    '%Y-%m-%dT%H:%M:%S.%f',
    '%Y-%m-%dT%H:%M',
    '%b %d %Y at %H:%M',
    '%b %d %Y at %H:%M:%S',
    '%B %d %Y at %H:%M',
    '%B %d %Y at %H:%M:%S',
    '%H:%M %d-%b-%Y',
)

DATE_FORMATS_DAY_FIRST = list(DATE_FORMATS)
DATE_FORMATS_DAY_FIRST.extend([
    '%d-%m-%Y',
    '%d.%m.%Y',
    '%d.%m.%y',
    '%d/%m/%Y',
    '%d/%m/%y',
    '%d/%m/%Y %H:%M:%S',
])

DATE_FORMATS_MONTH_FIRST = list(DATE_FORMATS)
DATE_FORMATS_MONTH_FIRST.extend([
    '%m-%d-%Y',
    '%m.%d.%Y',
    '%m/%d/%Y',
    '%m/%d/%y',
    '%m/%d/%Y %H:%M:%S',
])

PACKED_CODES_RE = r"}\('(.+)',(\d+),(\d+),'([^']+)'\.split\('\|'\)"
JSON_LD_RE = r'(?is)<script[^>]+type=(["\']?)application/ld\+json\1[^>]*>(?P<json_ld>.+?)</script>'


def preferredencoding():
    """Get preferred encoding.

    Returns the best encoding scheme for the system, based on
    locale.getpreferredencoding() and some further tweaks.
    """
    try:
        pref = locale.getpreferredencoding()
        'TEST'.encode(pref)
    except Exception:
        pref = 'UTF-8'

    return pref


def write_json_file(obj, fn):
    """ Encode obj as JSON and write it to fn, atomically if possible """

    fn = encodeFilename(fn)
    path_basename = os.path.basename
    path_dirname = os.path.dirname

    tf = tempfile.NamedTemporaryFile(
        suffix='.tmp',
        prefix=path_basename(fn) + '.',
        dir=path_dirname(fn),
        delete=False,
        mode='w',
        encoding='utf-8'
    )

    try:
        with tf:
            json.dump(obj, tf, ensure_ascii=False)
        if sys.platform == 'win32':
            # Need to remove existing file on Windows, else os.rename raises
            # WindowsError or FileExistsError.
            try:
                os.unlink(fn)
            except OSError:
                pass
        try:
            mask = os.umask(0)
            os.umask(mask)
            os.chmod(tf.name, 0o666 & ~mask)
        except OSError:
            pass
        os.rename(tf.name, fn)
    except Exception:
        try:
            os.remove(tf.name)
        except OSError:
            pass
        raise


def find_xpath_attr(node, xpath, key, val=None):
    """ Find the xpath xpath[@key=val] """
    assert re.match(r'^[a-zA-Z_-]+$', key)
    expr = xpath + ('[@%s]' % key if val is None else "[@%s='%s']" % (key, val))
    return node.find(expr)

# On python2.6 the xml.etree.ElementTree.Element methods don't support
# the namespace parameter


def xpath_with_ns(path, ns_map):
    components = [c.split(':') for c in path.split('/')]
    replaced = []
    for c in components:
        if len(c) == 1:
            replaced.append(c[0])
        else:
            ns, tag = c
            replaced.append('{%s}%s' % (ns_map[ns], tag))
    return '/'.join(replaced)


def xpath_element(node, xpath, name=None, fatal=False, default=NO_DEFAULT):
    from .exceptions import ExtractorError
    def _find_xpath(xpath):
        return node.find(xpath)

    if isinstance(xpath, (str, compat_str)):
        n = _find_xpath(xpath)
    else:
        for xp in xpath:
            n = _find_xpath(xp)
            if n is not None:
                break

    if n is None:
        if default is not NO_DEFAULT:
            return default
        elif fatal:
            name = xpath if name is None else name
            raise ExtractorError('Could not find XML element %s' % name)
        else:
            return None
    return n


def xpath_text(node, xpath, name=None, fatal=False, default=NO_DEFAULT):
    from .exceptions import ExtractorError
    n = xpath_element(node, xpath, name, fatal=fatal, default=default)
    if n is None or n == default:
        return n
    if n.text is None:
        if default is not NO_DEFAULT:
            return default
        elif fatal:
            name = xpath if name is None else name
            raise ExtractorError('Could not find XML element\'s text %s' % name)
        else:
            return None
    return n.text


def xpath_attr(node, xpath, key, name=None, fatal=False, default=NO_DEFAULT):
    from .exceptions import ExtractorError
    n = find_xpath_attr(node, xpath, key)
    if n is None:
        if default is not NO_DEFAULT:
            return default
        elif fatal:
            name = '%s[@%s]' % (xpath, key) if name is None else name
            raise ExtractorError('Could not find XML attribute %s' % name)
        else:
            return None
    return n.attrib[key]


def get_element_by_id(id, html):
    """Return the content of the tag with the specified ID in the passed HTML document"""
    return get_element_by_attribute('id', id, html)


def get_element_html_by_id(id, html):
    """Return the html of the tag with the specified ID in the passed HTML document"""
    return get_element_html_by_attribute('id', id, html)


def get_element_by_class(class_name, html):
    """Return the content of the first tag with the specified class in the passed HTML document"""
    retval = get_elements_by_class(class_name, html)
    return retval[0] if retval else None


def get_element_html_by_class(class_name, html):
    """Return the html of the first tag with the specified class in the passed HTML document"""
    retval = get_elements_html_by_class(class_name, html)
    return retval[0] if retval else None


def get_element_by_attribute(attribute, value, html, escape_value=True):
    retval = get_elements_by_attribute(attribute, value, html, escape_value)
    return retval[0] if retval else None


def get_element_html_by_attribute(attribute, value, html, escape_value=True):
    retval = get_elements_html_by_attribute(attribute, value, html, escape_value)
    return retval[0] if retval else None


def get_elements_by_class(class_name, html):
    """Return the content of all tags with the specified class in the passed HTML document as a list"""
    return get_elements_by_attribute(
        'class', r'[^\'"]*\b%s\b[^\'"]*' % re.escape(class_name),
        html, escape_value=False)


def get_elements_html_by_class(class_name, html):
    """Return the html of all tags with the specified class in the passed HTML document as a list"""
    return get_elements_html_by_attribute(
        'class', r'[^\'"]*\b%s\b[^\'"]*' % re.escape(class_name),
        html, escape_value=False)


def get_elements_by_attribute(*args, **kwargs):
    """Return the content of the tag with the specified attribute in the passed HTML document"""
    return [content for content, _ in get_elements_text_and_html_by_attribute(*args, **kwargs)]


def get_elements_html_by_attribute(*args, **kwargs):
    """Return the html of the tag with the specified attribute in the passed HTML document"""
    return [whole for _, whole in get_elements_text_and_html_by_attribute(*args, **kwargs)]


def get_elements_text_and_html_by_attribute(attribute, value, html, escape_value=True):
    """
    Return the text (content) and the html (whole) of the tag with the specified
    attribute in the passed HTML document
    """

    value_quote_optional = '' if re.match(r'''[\s"'`=<>]''', value) else '?'

    value = re.escape(value) if escape_value else value

    partial_element_re = r'''(?x)
        <(?P<tag>[a-zA-Z0-9:._-]+)
         (?:\s(?:[^>"']|"[^"]*"|'[^']*')*)?
         \s%(attribute)s\s*=\s*(?P<_q>['"]%(vqo)s)(?-x:%(value)s)(?P=_q)
        ''' % {'attribute': re.escape(attribute), 'value': value, 'vqo': value_quote_optional}

    for m in re.finditer(partial_element_re, html):
        content, whole = get_element_text_and_html_by_tag(m.group('tag'), html[m.start():])

        yield (
            unescapeHTML(re.sub(r'^(?P<q>["\'])(?P<content>.*)(?P=q)$', r'\g<content>', content, flags=re.DOTALL)),
            whole
        )


class HTMLBreakOnClosingTagParser(compat_HTMLParser):
    """
    HTML parser which raises HTMLBreakOnClosingTagException upon reaching the
    closing tag for the first opening tag it has encountered, and can be used
    as a context manager
    """

    class HTMLBreakOnClosingTagException(Exception):
        pass

    def __init__(self):
        self.tagstack = collections.deque()
        compat_HTMLParser.__init__(self)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        # handle_endtag does not return upon raising HTMLBreakOnClosingTagException,
        # so data remains buffered; we no longer have any interest in it, thus
        # override this method to discard it
        pass

    def handle_starttag(self, tag, _):
        self.tagstack.append(tag)

    def handle_endtag(self, tag):
        if not self.tagstack:
            raise compat_HTMLParseError('no tags in the stack')
        while self.tagstack:
            inner_tag = self.tagstack.pop()
            if inner_tag == tag:
                break
        else:
            raise compat_HTMLParseError(f'matching opening tag for closing {tag} tag not found')
        if not self.tagstack:
            raise self.HTMLBreakOnClosingTagException()


def get_element_text_and_html_by_tag(tag, html):
    """
    For the first element with the specified tag in the passed HTML document
    return its' content (text) and the whole element (html)
    """
    def find_or_raise(haystack, needle, exc):
        try:
            return haystack.index(needle)
        except ValueError:
            raise exc
    closing_tag = f'</{tag}>'
    whole_start = find_or_raise(
        html, f'<{tag}', compat_HTMLParseError(f'opening {tag} tag not found'))
    content_start = find_or_raise(
        html[whole_start:], '>', compat_HTMLParseError(f'malformed opening {tag} tag'))
    content_start += whole_start + 1
    with HTMLBreakOnClosingTagParser() as parser:
        parser.feed(html[whole_start:content_start])
        if not parser.tagstack or parser.tagstack[0] != tag:
            raise compat_HTMLParseError(f'parser did not match opening {tag} tag')
        offset = content_start
        while offset < len(html):
            next_closing_tag_start = find_or_raise(
                html[offset:], closing_tag,
                compat_HTMLParseError(f'closing {tag} tag not found'))
            next_closing_tag_end = next_closing_tag_start + len(closing_tag)
            try:
                parser.feed(html[offset:offset + next_closing_tag_end])
                offset += next_closing_tag_end
            except HTMLBreakOnClosingTagParser.HTMLBreakOnClosingTagException:
                return html[content_start:offset + next_closing_tag_start], \
                    html[whole_start:offset + next_closing_tag_end]
        raise compat_HTMLParseError('unexpected end of html')


class HTMLAttributeParser(compat_HTMLParser):
    """Trivial HTML parser to gather the attributes for a single element"""

    def __init__(self):
        self.attrs = {}
        compat_HTMLParser.__init__(self)

    def handle_starttag(self, tag, attrs):
        self.attrs = dict(attrs)


class HTMLListAttrsParser(compat_HTMLParser):
    """HTML parser to gather the attributes for the elements of a list"""

    def __init__(self):
        compat_HTMLParser.__init__(self)
        self.items = []
        self._level = 0

    def handle_starttag(self, tag, attrs):
        if tag == 'li' and self._level == 0:
            self.items.append(dict(attrs))
        self._level += 1

    def handle_endtag(self, tag):
        self._level -= 1


def extract_attributes(html_element):
    """Given a string for an HTML element such as
    <el
         a="foo" B="bar" c="&98;az" d=boz
         empty= noval entity="&amp;"
         sq='"' dq="'"
    >
    Decode and return a dictionary of attributes.
    {
        'a': 'foo', 'b': 'bar', c: 'baz', d: 'boz',
        'empty': '', 'noval': None, 'entity': '&',
        'sq': '"', 'dq': '\''
    }.
    NB HTMLParser is stricter in Python 2.6 & 3.2 than in later versions,
    but the cases in the unit test will work for all of 2.6, 2.7, 3.2-3.5.
    """
    parser = HTMLAttributeParser()
    try:
        parser.feed(html_element)
        parser.close()
    # Older Python may throw HTMLParseError in case of malformed HTML
    except compat_HTMLParseError:
        pass
    return parser.attrs


def parse_list(webpage):
    """Given a string for an series of HTML <li> elements,
    return a dictionary of their attributes"""
    parser = HTMLListAttrsParser()
    parser.feed(webpage)
    parser.close()
    return parser.items


def clean_html(html):
    """Clean an HTML snippet into a readable string"""

    if html is None:  # Convenience for sanitizing descriptions etc.
        return html

    html = re.sub(r'\s+', ' ', html)
    html = re.sub(r'(?u)\s?<\s?br\s?/?\s?>\s?', '\n', html)
    html = re.sub(r'(?u)<\s?/\s?p\s?>\s?<\s?p[^>]*>', '\n', html)
    # Strip html tags
    html = re.sub('<.*?>', '', html)
    # Replace html entities
    html = unescapeHTML(html)
    return html.strip()


def sanitize_open(filename, open_mode):
    """Try to open the given filename, and slightly tweak it if this fails.
    Attempts to open the given filename. If this fails, it tries to change
    the filename slightly, step by step, until it's either able to open it
    or it fails and raises a final exception, like the standard open()
    function.
    It returns the tuple (stream, definitive_file_name).
    """
    try:
        if filename == '-':
            if sys.platform == 'win32':
                import msvcrt
                msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
            return (sys.stdout.buffer if hasattr(sys.stdout, 'buffer') else sys.stdout, filename)
        stream = locked_file(filename, open_mode, block=False).open()
        return (stream, filename)
    except (IOError, OSError) as err:
        if err.errno in (errno.EACCES,):
            raise

        # In case of error, try to remove win32 forbidden chars
        alt_filename = sanitize_path(filename)
        if alt_filename == filename:
            raise
        else:
            # An exception here should be caught in the caller
            stream = locked_file(filename, open_mode, block=False).open()
            return (stream, alt_filename)


def timeconvert(timestr):
    """Convert RFC 2822 defined time string into system timestamp"""
    timestamp = None
    timetuple = email.utils.parsedate_tz(timestr)
    if timetuple is not None:
        timestamp = email.utils.mktime_tz(timetuple)
    return timestamp


def sanitize_filename(s, restricted=False, is_id=False):
    """Sanitizes a string so it could be used as part of a filename.
    If restricted is set, use a stricter subset of allowed characters.
    Set is_id if this is not an arbitrary string, but an ID that should be kept
    if possible.
    """
    def replace_insane(char):
        if restricted and char in ACCENT_CHARS:
            return ACCENT_CHARS[char]
        elif not restricted and char == '\n':
            return ' '
        elif char == '?' or ord(char) < 32 or ord(char) == 127:
            return ''
        elif char == '"':
            return '' if restricted else '\''
        elif char == ':':
            return '_-' if restricted else ' -'
        elif char in '\\/|*<>':
            return '_'
        if restricted and (char in '!&\'()[]{}$;`^,#' or char.isspace()):
            return '_'
        if restricted and ord(char) > 127:
            return '_'
        return char

    if s == '':
        return ''
    # Handle timestamps
    s = re.sub(r'[0-9]+(?::[0-9]+)+', lambda m: m.group(0).replace(':', '_'), s)
    result = ''.join(map(replace_insane, s))
    if not is_id:
        while '__' in result:
            result = result.replace('__', '_')
        result = result.strip('_')
        # Common case of "Foreign band name - English song title"
        if restricted and result.startswith('-_'):
            result = result[2:]
        if result.startswith('-'):
            result = '_' + result[len('-'):]
        result = result.lstrip('.')
        if not result:
            result = '_'
    return result


def sanitize_path(s, force=False):
    """Sanitizes and normalizes path on Windows"""
    if sys.platform == 'win32':
        force = False
        drive_or_unc, _ = os.path.splitdrive(s)
    elif force:
        drive_or_unc = ''
    else:
        return s

    norm_path = os.path.normpath(remove_start(s, drive_or_unc)).split(os.path.sep)
    if drive_or_unc:
        norm_path.pop(0)
    sanitized_path = [
        path_part if path_part in ['.', '..'] else re.sub(r'(?:[/<>:"\|\\?\*]|[\s.]$)', '#', path_part)
        for path_part in norm_path]
    if drive_or_unc:
        sanitized_path.insert(0, drive_or_unc + os.path.sep)
    elif force and s[0] == os.path.sep:
        sanitized_path.insert(0, os.path.sep)
    return os.path.join(*sanitized_path)


def sanitize_url(url):
    # Prepend protocol-less URLs with `http:` scheme in order to mitigate
    # the number of unwanted failures due to missing protocol
    if url.startswith('//'):
        return 'http:%s' % url
    # Fix some common typos seen so far
    COMMON_TYPOS = (
        # https://github.com/ytdl-org/youtube-dl/issues/15649
        (r'^httpss://', r'https://'),
        # https://bx1.be/lives/direct-tv/
        (r'^rmtp([es]?)://', r'rtmp\1://'),
    )
    for mistake, fixup in COMMON_TYPOS:
        if re.match(mistake, url):
            return re.sub(mistake, fixup, url)
    return url


def extract_basic_auth(url):
    parts = compat_urlparse.urlsplit(url)
    if parts.username is None:
        return url, None
    url = compat_urlparse.urlunsplit(parts._replace(netloc=(
        parts.hostname if parts.port is None
        else '%s:%d' % (parts.hostname, parts.port))))
    auth_payload = base64.b64encode(
        ('%s:%s' % (parts.username, parts.password or '')).encode('utf-8'))
    return url, 'Basic ' + auth_payload.decode('utf-8')


def sanitized_Request(url, *args, **kwargs):
    url, auth_header = extract_basic_auth(escape_url(sanitize_url(url)))
    if auth_header is not None:
        headers = args[1] if len(args) >= 2 else kwargs.setdefault('headers', {})
        headers['Authorization'] = auth_header
    return compat_urllib_request.Request(url, *args, **kwargs)


def expand_path(s):
    """Expand shell variables and ~"""
    return os.path.expandvars(compat_expanduser(s))


def orderedSet(iterable):
    """ Remove all duplicates from the input iterable """
    res = []
    for el in iterable:
        if el not in res:
            res.append(el)
    return res


def _htmlentity_transform(entity_with_semicolon):
    """Transforms an HTML entity to a character."""
    entity = entity_with_semicolon[:-1]

    # Known non-numeric HTML entity
    if entity in compat_html_entities.name2codepoint:
        return compat_chr(compat_html_entities.name2codepoint[entity])

    # TODO: HTML5 allows entities without a semicolon. For example,
    # '&Eacuteric' should be decoded as 'Éric'.
    if entity_with_semicolon in compat_html_entities_html5:
        return compat_html_entities_html5[entity_with_semicolon]

    mobj = re.match(r'#(x[0-9a-fA-F]+|[0-9]+)', entity)
    if mobj is not None:
        numstr = mobj.group(1)
        if numstr.startswith('x'):
            base = 16
            numstr = '0%s' % numstr
        else:
            base = 10
        # See https://github.com/ytdl-org/youtube-dl/issues/7518
        try:
            return compat_chr(int(numstr, base))
        except ValueError:
            pass

    # Unknown entity in name, return its literal representation
    return '&%s;' % entity


def unescapeHTML(s):
    if s is None:
        return None
    assert type(s) == compat_str

    return re.sub(
        r'&([^&;]+;)', lambda m: _htmlentity_transform(m.group(1)), s)


def escapeHTML(text):
    return (
        text
        .replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
        .replace('"', '&quot;')
        .replace("'", '&#39;')
    )


def process_communicate_or_kill(p, *args, **kwargs):
    try:
        return p.communicate(*args, **kwargs)
    except BaseException:  # Including KeyboardInterrupt
        p.kill()
        p.wait()
        raise


class Popen(subprocess.Popen):
    if sys.platform == 'win32':
        _startupinfo = subprocess.STARTUPINFO()
        _startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    else:
        _startupinfo = None

    def __init__(self, *args, **kwargs):
        super(Popen, self).__init__(*args, **kwargs, startupinfo=self._startupinfo)

    def communicate_or_kill(self, *args, **kwargs):
        return process_communicate_or_kill(self, *args, **kwargs)


def get_subprocess_encoding():
    if sys.platform == 'win32' and sys.getwindowsversion()[0] >= 5:
        # For subprocess calls, encode with locale encoding
        # Refer to http://stackoverflow.com/a/9951851/35070
        encoding = preferredencoding()
    else:
        encoding = sys.getfilesystemencoding()
    if encoding is None:
        encoding = 'utf-8'
    return encoding


def encodeFilename(s, for_subprocess=False):
    """
    @param s The name of the file
    """

    assert type(s) == compat_str

    # Python 3 has a Unicode API
    return s


def decodeFilename(b, for_subprocess=False):
    return b


def encodeArgument(s):
    if not isinstance(s, compat_str):
        # Legacy code that uses byte strings
        # Uncomment the following line after fixing all post processors
        # assert False, 'Internal error: %r should be of type %r, is %r' % (s, compat_str, type(s))
        s = s.decode('ascii')
    return encodeFilename(s, True)


def decodeArgument(b):
    return decodeFilename(b, True)


def decodeOption(optval):
    if optval is None:
        return optval
    if isinstance(optval, bytes):
        optval = optval.decode(preferredencoding())

    assert isinstance(optval, compat_str)
    return optval


_timetuple = collections.namedtuple('Time', ('hours', 'minutes', 'seconds', 'milliseconds'))


def timetuple_from_msec(msec):
    secs, msec = divmod(msec, 1000)
    mins, secs = divmod(secs, 60)
    hrs, mins = divmod(mins, 60)
    return _timetuple(hrs, mins, secs, msec)


def formatSeconds(secs, delim=':', msec=False):
    time = timetuple_from_msec(secs * 1000)
    if time.hours:
        ret = '%d%s%02d%s%02d' % (time.hours, delim, time.minutes, delim, time.seconds)
    elif time.minutes:
        ret = '%d%s%02d' % (time.minutes, delim, time.seconds)
    else:
        ret = '%d' % time.seconds
    return '%s.%03d' % (ret, time.milliseconds) if msec else ret


def extract_timezone(date_str):
    m = re.search(
        r'''(?x)
            ^.{8,}?                                              # >=8 char non-TZ prefix, if present
            (?P<tz>Z|                                            # just the UTC Z, or
                (?:(?<=.\b\d{4}|\b\d{2}:\d\d)|                   # preceded by 4 digits or hh:mm or
                   (?<!.\b[a-zA-Z]{3}|[a-zA-Z]{4}|..\b\d\d))     # not preceded by 3 alpha word or >= 4 alpha or 2 digits
                   [ ]?                                          # optional space
                (?P<sign>\+|-)                                   # +/-
                (?P<hours>[0-9]{2}):?(?P<minutes>[0-9]{2})       # hh[:]mm
            $)
        ''', date_str)
    if not m:
        timezone = datetime.timedelta()
    else:
        date_str = date_str[:-len(m.group('tz'))]
        if not m.group('sign'):
            timezone = datetime.timedelta()
        else:
            sign = 1 if m.group('sign') == '+' else -1
            timezone = datetime.timedelta(
                hours=sign * int(m.group('hours')),
                minutes=sign * int(m.group('minutes')))
    return timezone, date_str


def parse_iso8601(date_str, delimiter='T', timezone=None):
    """ Return a UNIX timestamp from the given date """

    if date_str is None:
        return None

    date_str = re.sub(r'\.[0-9]+', '', date_str)

    if timezone is None:
        timezone, date_str = extract_timezone(date_str)

    try:
        date_format = '%Y-%m-%d{0}%H:%M:%S'.format(delimiter)
        dt = datetime.datetime.strptime(date_str, date_format) - timezone
        return calendar.timegm(dt.timetuple())
    except ValueError:
        pass


def date_formats(day_first=True):
    return DATE_FORMATS_DAY_FIRST if day_first else DATE_FORMATS_MONTH_FIRST


def unified_strdate(date_str, day_first=True):
    """Return a string with the date in the format YYYYMMDD"""

    if date_str is None:
        return None
    upload_date = None
    # Replace commas
    date_str = date_str.replace(',', ' ')
    # Remove AM/PM + timezone
    date_str = re.sub(r'(?i)\s*(?:AM|PM)(?:\s+[A-Z]+)?', '', date_str)
    _, date_str = extract_timezone(date_str)

    for expression in date_formats(day_first):
        try:
            upload_date = datetime.datetime.strptime(date_str, expression).strftime('%Y%m%d')
        except ValueError:
            pass
    if upload_date is None:
        timetuple = email.utils.parsedate_tz(date_str)
        if timetuple:
            try:
                upload_date = datetime.datetime(*timetuple[:6]).strftime('%Y%m%d')
            except ValueError:
                pass
    if upload_date is not None:
        return compat_str(upload_date)


def unified_timestamp(date_str, day_first=True):
    if date_str is None:
        return None

    date_str = re.sub(r'[,|]', '', date_str)

    pm_delta = 12 if re.search(r'(?i)PM', date_str) else 0
    timezone, date_str = extract_timezone(date_str)

    # Remove AM/PM + timezone
    date_str = re.sub(r'(?i)\s*(?:AM|PM)(?:\s+[A-Z]+)?', '', date_str)

    # Remove unrecognized timezones from ISO 8601 alike timestamps
    m = re.search(r'\d{1,2}:\d{1,2}(?:\.\d+)?(?P<tz>\s*[A-Z]+)$', date_str)
    if m:
        date_str = date_str[:-len(m.group('tz'))]

    # Python only supports microseconds, so remove nanoseconds
    m = re.search(r'^([0-9]{4,}-[0-9]{1,2}-[0-9]{1,2}T[0-9]{1,2}:[0-9]{1,2}:[0-9]{1,2}\.[0-9]{6})[0-9]+$', date_str)
    if m:
        date_str = m.group(1)

    for expression in date_formats(day_first):
        try:
            dt = datetime.datetime.strptime(date_str, expression) - timezone + datetime.timedelta(hours=pm_delta)
            return calendar.timegm(dt.timetuple())
        except ValueError:
            pass
    timetuple = email.utils.parsedate_tz(date_str)
    if timetuple:
        return calendar.timegm(timetuple) + pm_delta * 3600


def determine_ext(url, default_ext='unknown_video'):
    if url is None or '.' not in url:
        return default_ext
    guess = url.partition('?')[0].rpartition('.')[2]
    if re.match(r'^[A-Za-z0-9]+$', guess):
        return guess
    # Try extract ext from URLs like http://example.com/foo/bar.mp4/?download
    elif guess.rstrip('/') in KNOWN_EXTENSIONS:
        return guess.rstrip('/')
    else:
        return default_ext


def subtitles_filename(filename, sub_lang, sub_format, expected_real_ext=None):
    return replace_extension(filename, sub_lang + '.' + sub_format, expected_real_ext)


def datetime_from_str(date_str, precision='auto', format='%Y%m%d'):
    """
    Return a datetime object from a string in the format YYYYMMDD or
    (now|today|date)[+-][0-9](microsecond|second|minute|hour|day|week|month|year)(s)?

    format: string date format used to return datetime object from
    precision: round the time portion of a datetime object.
                auto|microsecond|second|minute|hour|day.
                auto: round to the unit provided in date_str (if applicable).
    """
    auto_precision = False
    if precision == 'auto':
        auto_precision = True
        precision = 'microsecond'
    today = datetime_round(datetime.datetime.utcnow(), precision)
    if date_str in ('now', 'today'):
        return today
    if date_str == 'yesterday':
        return today - datetime.timedelta(days=1)
    match = re.match(
        r'(?P<start>.+)(?P<sign>[+-])(?P<time>\d+)(?P<unit>microsecond|second|minute|hour|day|week|month|year)(s)?',
        date_str)
    if match is not None:
        start_time = datetime_from_str(match.group('start'), precision, format)
        time = int(match.group('time')) * (-1 if match.group('sign') == '-' else 1)
        unit = match.group('unit')
        if unit == 'month' or unit == 'year':
            new_date = datetime_add_months(start_time, time * 12 if unit == 'year' else time)
            unit = 'day'
        else:
            if unit == 'week':
                unit = 'day'
                time *= 7
            delta = datetime.timedelta(**{unit + 's': time})
            new_date = start_time + delta
        if auto_precision:
            return datetime_round(new_date, unit)
        return new_date

    return datetime_round(datetime.datetime.strptime(date_str, format), precision)


def date_from_str(date_str, format='%Y%m%d'):
    """
    Return a datetime object from a string in the format YYYYMMDD or
    (now|today|date)[+-][0-9](microsecond|second|minute|hour|day|week|month|year)(s)?

    format: string date format used to return datetime object from
    """
    return datetime_from_str(date_str, precision='microsecond', format=format).date()


def datetime_add_months(dt, months):
    """Increment/Decrement a datetime object by months."""
    month = dt.month + months - 1
    year = dt.year + month // 12
    month = month % 12 + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year, month, day)


def datetime_round(dt, precision='day'):
    """
    Round a datetime object's time to a specific precision
    """
    if precision == 'microsecond':
        return dt

    unit_seconds = {
        'day': 86400,
        'hour': 3600,
        'minute': 60,
        'second': 1,
    }
    roundto = lambda x, n: ((x + n / 2) // n) * n
    timestamp = calendar.timegm(dt.timetuple())
    return datetime.datetime.utcfromtimestamp(roundto(timestamp, unit_seconds[precision]))


def hyphenate_date(date_str):
    """
    Convert a date in 'YYYYMMDD' format to 'YYYY-MM-DD' format"""
    match = re.match(r'^(\d\d\d\d)(\d\d)(\d\d)$', date_str)
    if match is not None:
        return '-'.join(match.groups())
    else:
        return date_str


class DateRange(object):
    """Represents a time interval between two dates"""

    def __init__(self, start=None, end=None):
        """start and end must be strings in the format accepted by date"""
        if start is not None:
            self.start = date_from_str(start)
        else:
            self.start = datetime.datetime.min.date()
        if end is not None:
            self.end = date_from_str(end)
        else:
            self.end = datetime.datetime.max.date()
        if self.start > self.end:
            raise ValueError('Date range: "%s" , the start date must be before the end date' % self)

    @classmethod
    def day(cls, day):
        """Returns a range that only contains the given day"""
        return cls(day, day)

    def __contains__(self, date):
        """Check if the date is in the range"""
        if not isinstance(date, datetime.date):
            date = date_from_str(date)
        return self.start <= date <= self.end

    def __str__(self):
        return '%s - %s' % (self.start.isoformat(), self.end.isoformat())


def platform_name():
    """ Returns the platform name as a compat_str """
    res = platform.platform()
    if isinstance(res, bytes):
        res = res.decode(preferredencoding())

    assert isinstance(res, compat_str)
    return res


def get_windows_version():
    ''' Get Windows version. None if it's not running on Windows '''
    if compat_os_name == 'nt':
        return version_tuple(platform.win32_ver()[1])
    else:
        return None


def write_string(s, out=None, encoding=None):
    if out is None:
        out = sys.stderr
    assert type(s) == compat_str

    if 'b' in getattr(out, 'mode', ''):
        byt = s.encode(encoding or preferredencoding(), 'ignore')
        out.write(byt)
    elif hasattr(out, 'buffer'):
        enc = encoding or getattr(out, 'encoding', None) or preferredencoding()
        byt = s.encode(enc, 'ignore')
        out.buffer.write(byt)
    else:
        out.write(s)
    out.flush()


def bytes_to_intlist(bs):
    if not bs:
        return []
    if isinstance(bs[0], int):  # Python 3
        return list(bs)
    else:
        return [ord(c) for c in bs]


def intlist_to_bytes(xs):
    if not xs:
        return b''
    return compat_struct_pack('%dB' % len(xs), *xs)


# Cross-platform file locking
if sys.platform == 'win32':
    import ctypes.wintypes
    import msvcrt

    class OVERLAPPED(ctypes.Structure):
        _fields_ = [
            ('Internal', ctypes.wintypes.LPVOID),
            ('InternalHigh', ctypes.wintypes.LPVOID),
            ('Offset', ctypes.wintypes.DWORD),
            ('OffsetHigh', ctypes.wintypes.DWORD),
            ('hEvent', ctypes.wintypes.HANDLE),
        ]

    kernel32 = ctypes.windll.kernel32
    LockFileEx = kernel32.LockFileEx
    LockFileEx.argtypes = [
        ctypes.wintypes.HANDLE,     # hFile
        ctypes.wintypes.DWORD,      # dwFlags
        ctypes.wintypes.DWORD,      # dwReserved
        ctypes.wintypes.DWORD,      # nNumberOfBytesToLockLow
        ctypes.wintypes.DWORD,      # nNumberOfBytesToLockHigh
        ctypes.POINTER(OVERLAPPED)  # Overlapped
    ]
    LockFileEx.restype = ctypes.wintypes.BOOL
    UnlockFileEx = kernel32.UnlockFileEx
    UnlockFileEx.argtypes = [
        ctypes.wintypes.HANDLE,     # hFile
        ctypes.wintypes.DWORD,      # dwReserved
        ctypes.wintypes.DWORD,      # nNumberOfBytesToLockLow
        ctypes.wintypes.DWORD,      # nNumberOfBytesToLockHigh
        ctypes.POINTER(OVERLAPPED)  # Overlapped
    ]
    UnlockFileEx.restype = ctypes.wintypes.BOOL
    whole_low = 0xffffffff
    whole_high = 0x7fffffff

    def _lock_file(f, exclusive, block):  # todo: block unused on win32
        overlapped = OVERLAPPED()
        overlapped.Offset = 0
        overlapped.OffsetHigh = 0
        overlapped.hEvent = 0
        f._lock_file_overlapped_p = ctypes.pointer(overlapped)
        handle = msvcrt.get_osfhandle(f.fileno())
        if not LockFileEx(handle, 0x2 if exclusive else 0x0, 0,
                          whole_low, whole_high, f._lock_file_overlapped_p):
            raise OSError('Locking file failed: %r' % ctypes.FormatError())

    def _unlock_file(f):
        assert f._lock_file_overlapped_p
        handle = msvcrt.get_osfhandle(f.fileno())
        if not UnlockFileEx(handle, 0,
                            whole_low, whole_high, f._lock_file_overlapped_p):
            raise OSError('Unlocking file failed: %r' % ctypes.FormatError())

else:
    # Some platforms, such as Jython, is missing fcntl
    try:
        import fcntl

        def _lock_file(f, exclusive, block):
            fcntl.flock(f,
                        fcntl.LOCK_SH if not exclusive
                        else fcntl.LOCK_EX if block
                        else fcntl.LOCK_EX | fcntl.LOCK_NB)

        def _unlock_file(f):
            fcntl.flock(f, fcntl.LOCK_UN)

    except ImportError:
        UNSUPPORTED_MSG = 'file locking is not supported on this platform'

        def _lock_file(f, exclusive, block):
            raise IOError(UNSUPPORTED_MSG)

        def _unlock_file(f):
            raise IOError(UNSUPPORTED_MSG)


class locked_file(object):
    def __init__(self, filename, mode, block=True, encoding=None):
        assert mode in ['r', 'rb', 'a', 'ab', 'w', 'wb']
        self.f = io.open(filename, mode, encoding=encoding)
        self.mode = mode
        self.block = block

    def __enter__(self):
        exclusive = 'r' not in self.mode
        try:
            _lock_file(self.f, exclusive, self.block)
        except IOError:
            self.f.close()
            raise
        return self

    def __exit__(self, etype, value, traceback):
        try:
            _unlock_file(self.f)
        finally:
            self.f.close()

    def __iter__(self):
        return iter(self.f)

    def write(self, *args):
        return self.f.write(*args)

    def read(self, *args):
        return self.f.read(*args)

    def flush(self):
        self.f.flush()

    def open(self):
        return self.__enter__()

    def close(self, *args):
        self.__exit__(self, *args, value=False, traceback=False)


def get_filesystem_encoding():
    encoding = sys.getfilesystemencoding()
    return encoding if encoding is not None else 'utf-8'


def shell_quote(args):
    quoted_args = []
    encoding = get_filesystem_encoding()
    for a in args:
        if isinstance(a, bytes):
            # We may get a filename encoded with 'encodeFilename'
            a = a.decode(encoding)
        quoted_args.append(compat_shlex_quote(a))
    return ' '.join(quoted_args)


def smuggle_url(url, data):
    """ Pass additional data in a URL for internal use. """

    url, idata = unsmuggle_url(url, {})
    data.update(idata)
    sdata = compat_urllib_parse_urlencode(
        {'__youtubedl_smuggle': json.dumps(data)})
    return url + '#' + sdata


def unsmuggle_url(smug_url, default=None):
    if '#__youtubedl_smuggle' not in smug_url:
        return smug_url, default
    url, _, sdata = smug_url.rpartition('#')
    jsond = compat_parse_qs(sdata)['__youtubedl_smuggle'][0]
    data = json.loads(jsond)
    return url, data


def format_decimal_suffix(num, fmt='%d%s', *, factor=1000):
    """ Formats numbers with decimal sufixes like K, M, etc """
    num, factor = float_or_none(num), float(factor)
    if num is None:
        return None
    exponent = 0 if num == 0 else int(math.log(num, factor))
    suffix = ['', *'kMGTPEZY'][exponent]
    if factor == 1024:
        suffix = {'k': 'Ki', '': ''}.get(suffix, f'{suffix}i')
    converted = num / (factor ** exponent)
    return fmt % (converted, suffix)


def format_bytes(bytes):
    return format_decimal_suffix(bytes, '%.2f%sB', factor=1024) or 'N/A'


def lookup_unit_table(unit_table, s):
    units_re = '|'.join(re.escape(u) for u in unit_table)
    m = re.match(
        r'(?P<num>[0-9]+(?:[,.][0-9]*)?)\s*(?P<unit>%s)\b' % units_re, s)
    if not m:
        return None
    num_str = m.group('num').replace(',', '.')
    mult = unit_table[m.group('unit')]
    return int(float(num_str) * mult)


def parse_filesize(s):
    if s is None:
        return None

    # The lower-case forms are of course incorrect and unofficial,
    # but we support those too
    _UNIT_TABLE = {
        'B': 1,
        'b': 1,
        'bytes': 1,
        'KiB': 1024,
        'KB': 1000,
        'kB': 1024,
        'Kb': 1000,
        'kb': 1000,
        'kilobytes': 1000,
        'kibibytes': 1024,
        'MiB': 1024 ** 2,
        'MB': 1000 ** 2,
        'mB': 1024 ** 2,
        'Mb': 1000 ** 2,
        'mb': 1000 ** 2,
        'megabytes': 1000 ** 2,
        'mebibytes': 1024 ** 2,
        'GiB': 1024 ** 3,
        'GB': 1000 ** 3,
        'gB': 1024 ** 3,
        'Gb': 1000 ** 3,
        'gb': 1000 ** 3,
        'gigabytes': 1000 ** 3,
        'gibibytes': 1024 ** 3,
        'TiB': 1024 ** 4,
        'TB': 1000 ** 4,
        'tB': 1024 ** 4,
        'Tb': 1000 ** 4,
        'tb': 1000 ** 4,
        'terabytes': 1000 ** 4,
        'tebibytes': 1024 ** 4,
        'PiB': 1024 ** 5,
        'PB': 1000 ** 5,
        'pB': 1024 ** 5,
        'Pb': 1000 ** 5,
        'pb': 1000 ** 5,
        'petabytes': 1000 ** 5,
        'pebibytes': 1024 ** 5,
        'EiB': 1024 ** 6,
        'EB': 1000 ** 6,
        'eB': 1024 ** 6,
        'Eb': 1000 ** 6,
        'eb': 1000 ** 6,
        'exabytes': 1000 ** 6,
        'exbibytes': 1024 ** 6,
        'ZiB': 1024 ** 7,
        'ZB': 1000 ** 7,
        'zB': 1024 ** 7,
        'Zb': 1000 ** 7,
        'zb': 1000 ** 7,
        'zettabytes': 1000 ** 7,
        'zebibytes': 1024 ** 7,
        'YiB': 1024 ** 8,
        'YB': 1000 ** 8,
        'yB': 1024 ** 8,
        'Yb': 1000 ** 8,
        'yb': 1000 ** 8,
        'yottabytes': 1000 ** 8,
        'yobibytes': 1024 ** 8,
    }

    return lookup_unit_table(_UNIT_TABLE, s)


def parse_count(s):
    if s is None:
        return None

    s = re.sub(r'^[^\d]+\s', '', s).strip()

    if re.match(r'^[\d,.]+$', s):
        return str_to_int(s)

    _UNIT_TABLE = {
        'k': 1000,
        'K': 1000,
        'm': 1000 ** 2,
        'M': 1000 ** 2,
        'kk': 1000 ** 2,
        'KK': 1000 ** 2,
        'b': 1000 ** 3,
        'B': 1000 ** 3,
    }

    ret = lookup_unit_table(_UNIT_TABLE, s)
    if ret is not None:
        return ret

    mobj = re.match(r'([\d,.]+)(?:$|\s)', s)
    if mobj:
        return str_to_int(mobj.group(1))


def parse_resolution(s):
    if s is None:
        return {}

    mobj = re.search(r'(?<![a-zA-Z0-9])(?P<w>\d+)\s*[xX×,]\s*(?P<h>\d+)(?![a-zA-Z0-9])', s)
    if mobj:
        return {
            'width': int(mobj.group('w')),
            'height': int(mobj.group('h')),
        }

    mobj = re.search(r'(?<![a-zA-Z0-9])(\d+)[pPiI](?![a-zA-Z0-9])', s)
    if mobj:
        return {'height': int(mobj.group(1))}

    mobj = re.search(r'\b([48])[kK]\b', s)
    if mobj:
        return {'height': int(mobj.group(1)) * 540}

    return {}


def parse_bitrate(s):
    if not isinstance(s, compat_str):
        return
    mobj = re.search(r'\b(\d+)\s*kbps', s)
    if mobj:
        return int(mobj.group(1))


def month_by_name(name, lang='en'):
    """ Return the number of a month by (locale-independently) English name """

    month_names = MONTH_NAMES.get(lang, MONTH_NAMES['en'])

    try:
        return month_names.index(name) + 1
    except ValueError:
        return None


def month_by_abbreviation(abbrev):
    """ Return the number of a month by (locale-independently) English
        abbreviations """

    try:
        return [s[:3] for s in ENGLISH_MONTH_NAMES].index(abbrev) + 1
    except ValueError:
        return None


def fix_xml_ampersands(xml_str):
    """Replace all the '&' by '&amp;' in XML"""
    return re.sub(
        r'&(?!amp;|lt;|gt;|apos;|quot;|#x[0-9a-fA-F]{,4};|#[0-9]{,4};)',
        '&amp;',
        xml_str)


def setproctitle(title):
    assert isinstance(title, compat_str)

    # ctypes in Jython is not complete
    # http://bugs.jython.org/issue2148
    if sys.platform.startswith('java'):
        return

    try:
        libc = ctypes.cdll.LoadLibrary('libc.so.6')
    except OSError:
        return
    except TypeError:
        # LoadLibrary in Windows Python 2.7.13 only expects
        # a bytestring, but since unicode_literals turns
        # every string into a unicode string, it fails.
        return
    title_bytes = title.encode('utf-8')
    buf = ctypes.create_string_buffer(len(title_bytes))
    buf.value = title_bytes
    try:
        libc.prctl(15, buf, 0, 0, 0)
    except AttributeError:
        return  # Strange libc, just skip this


def remove_start(s, start):
    return s[len(start):] if s is not None and s.startswith(start) else s


def remove_end(s, end):
    return s[:-len(end)] if s is not None and s.endswith(end) else s


def remove_quotes(s):
    if s is None or len(s) < 2:
        return s
    for quote in ('"', "'", ):
        if s[0] == quote and s[-1] == quote:
            return s[1:-1]
    return s


def get_domain(url):
    domain = re.match(r'(?:https?:\/\/)?(?:www\.)?(?P<domain>[^\n\/]+\.[^\n\/]+)(?:\/(.*))?', url)
    return domain.group('domain') if domain else None


def url_basename(url):
    path = compat_urlparse.urlparse(url).path
    return path.strip('/').split('/')[-1]


def base_url(url):
    return re.match(r'https?://[^?#&]+/', url).group()


def urljoin(base, path):
    if isinstance(path, bytes):
        path = path.decode('utf-8')
    if not isinstance(path, compat_str) or not path:
        return None
    if re.match(r'^(?:[a-zA-Z][a-zA-Z0-9+-.]*:)?//', path):
        return path
    if isinstance(base, bytes):
        base = base.decode('utf-8')
    if not isinstance(base, compat_str) or not re.match(
            r'^(?:https?:)?//', base):
        return None
    return compat_urlparse.urljoin(base, path)


def int_or_none(v, scale=1, default=None, get_attr=None, invscale=1):
    if get_attr and v is not None:
        v = getattr(v, get_attr, None)
    try:
        return int(v) * invscale // scale
    except (ValueError, TypeError, OverflowError):
        return default


def str_or_none(v, default=None):
    return default if v is None else compat_str(v)


def str_to_int(int_str):
    """ A more relaxed version of int_or_none """
    if isinstance(int_str, int):
        return int_str
    elif isinstance(int_str, compat_str):
        int_str = re.sub(r'[,\.\+]', '', int_str)
        return int_or_none(int_str)


def float_or_none(v, scale=1, invscale=1, default=None):
    if v is None:
        return default
    try:
        return float(v) * invscale / scale
    except (ValueError, TypeError):
        return default


def bool_or_none(v, default=None):
    return v if isinstance(v, bool) else default


def strip_or_none(v, default=None):
    return v.strip() if isinstance(v, compat_str) else default


def url_or_none(url):
    if not url or not isinstance(url, compat_str):
        return None
    url = url.strip()
    return url if re.match(r'^(?:(?:https?|rt(?:m(?:pt?[es]?|fp)|sp[su]?)|mms|ftps?):)?//', url) else None


def strftime_or_none(timestamp, date_format, default=None):
    datetime_object = None
    try:
        if isinstance(timestamp, (int, float)):  # unix timestamp
            datetime_object = datetime.datetime.utcfromtimestamp(timestamp)
        elif isinstance(timestamp, compat_str):  # assume YYYYMMDD
            datetime_object = datetime.datetime.strptime(timestamp, '%Y%m%d')
        return datetime_object.strftime(date_format)
    except (ValueError, TypeError, AttributeError):
        return default


def parse_duration(s):
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not s:
        return None

    days, hours, mins, secs, ms = [None] * 5
    m = re.match(r'''(?x)
            (?P<before_secs>
                (?:(?:(?P<days>[0-9]+):)?(?P<hours>[0-9]+):)?(?P<mins>[0-9]+):)?
            (?P<secs>(?(before_secs)[0-9]{1,2}|[0-9]+))
            (?P<ms>[.:][0-9]+)?Z?$
        ''', s)
    if m:
        days, hours, mins, secs, ms = m.group('days', 'hours', 'mins', 'secs', 'ms')
    else:
        m = re.match(
            r'''(?ix)(?:P?
                (?:
                    [0-9]+\s*y(?:ears?)?\s*
                )?
                (?:
                    [0-9]+\s*m(?:onths?)?\s*
                )?
                (?:
                    [0-9]+\s*w(?:eeks?)?\s*
                )?
                (?:
                    (?P<days>[0-9]+)\s*d(?:ays?)?\s*
                )?
                T)?
                (?:
                    (?P<hours>[0-9]+)\s*h(?:ours?)?\s*
                )?
                (?:
                    (?P<mins>[0-9]+)\s*m(?:in(?:ute)?s?)?\s*
                )?
                (?:
                    (?P<secs>[0-9]+)(?P<ms>\.[0-9]+)?\s*s(?:ec(?:ond)?s?)?\s*
                )?Z?$''', s)
        if m:
            days, hours, mins, secs, ms = m.groups()
        else:
            m = re.match(r'(?i)(?:(?P<hours>[0-9.]+)\s*(?:hours?)|(?P<mins>[0-9.]+)\s*(?:mins?\.?|minutes?)\s*)Z?$', s)
            if m:
                hours, mins = m.groups()
            else:
                return None

    duration = 0
    if secs:
        duration += float(secs)
    if mins:
        duration += float(mins) * 60
    if hours:
        duration += float(hours) * 60 * 60
    if days:
        duration += float(days) * 24 * 60 * 60
    if ms:
        duration += float(ms.replace(':', '.'))
    return duration


def prepend_extension(filename, ext, expected_real_ext=None):
    name, real_ext = os.path.splitext(filename)
    return (
        '{0}.{1}{2}'.format(name, ext, real_ext)
        if not expected_real_ext or real_ext[1:] == expected_real_ext
        else '{0}.{1}'.format(filename, ext))


def replace_extension(filename, ext, expected_real_ext=None):
    name, real_ext = os.path.splitext(filename)
    return '{0}.{1}'.format(
        name if not expected_real_ext or real_ext[1:] == expected_real_ext else filename,
        ext)


def check_executable(exe, args=[]):
    """ Checks if the given binary is installed somewhere in PATH, and returns its name.
    args can be a list of arguments for a short output (like -version) """
    try:
        Popen([exe] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate_or_kill()
    except OSError:
        return False
    return exe


def _get_exe_version_output(exe, args):
    try:
        # STDIN should be redirected too. On UNIX-like systems, ffmpeg triggers
        # SIGTTOU if yt-dlp is run in the background.
        # See https://github.com/ytdl-org/youtube-dl/issues/955#issuecomment-209789656
        out, _ = Popen(
            [encodeArgument(exe)] + args, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT).communicate_or_kill()
    except OSError:
        return False
    if isinstance(out, bytes):  # Python 2.x
        out = out.decode('ascii', 'ignore')
    return out


def detect_exe_version(output, version_re=None, unrecognized='present'):
    assert isinstance(output, compat_str)
    if version_re is None:
        version_re = r'version\s+([-0-9._a-zA-Z]+)'
    m = re.search(version_re, output)
    if m:
        return m.group(1)
    else:
        return unrecognized


def get_exe_version(exe, args=['--version'],
                    version_re=None, unrecognized='present'):
    """ Returns the version of the specified executable,
    or False if the executable is not present """
    out = _get_exe_version_output(exe, args)
    return detect_exe_version(out, version_re, unrecognized) if out else False


class LazyList(collections.abc.Sequence):
    ''' Lazy immutable list from an iterable
    Note that slices of a LazyList are lists and not LazyList'''

    class IndexError(IndexError):
        pass

    def __init__(self, iterable, *, reverse=False, _cache=None):
        self.__iterable = iter(iterable)
        self.__cache = [] if _cache is None else _cache
        self.__reversed = reverse

    def __iter__(self):
        if self.__reversed:
            # We need to consume the entire iterable to iterate in reverse
            yield from self.exhaust()
            return
        yield from self.__cache
        for item in self.__iterable:
            self.__cache.append(item)
            yield item

    def __exhaust(self):
        self.__cache.extend(self.__iterable)
        # Discard the emptied iterable to make it pickle-able
        self.__iterable = []
        return self.__cache

    def exhaust(self):
        ''' Evaluate the entire iterable '''
        return self.__exhaust()[::-1 if self.__reversed else 1]

    @staticmethod
    def __reverse_index(x):
        return None if x is None else -(x + 1)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            if self.__reversed:
                idx = slice(self.__reverse_index(idx.start), self.__reverse_index(idx.stop), -(idx.step or 1))
            start, stop, step = idx.start, idx.stop, idx.step or 1
        elif isinstance(idx, int):
            if self.__reversed:
                idx = self.__reverse_index(idx)
            start, stop, step = idx, idx, 0
        else:
            raise TypeError('indices must be integers or slices')
        if ((start or 0) < 0 or (stop or 0) < 0
                or (start is None and step < 0)
                or (stop is None and step > 0)):
            # We need to consume the entire iterable to be able to slice from the end
            # Obviously, never use this with infinite iterables
            self.__exhaust()
            try:
                return self.__cache[idx]
            except IndexError as e:
                raise self.IndexError(e) from e
        n = max(start or 0, stop or 0) - len(self.__cache) + 1
        if n > 0:
            self.__cache.extend(itertools.islice(self.__iterable, n))
        try:
            return self.__cache[idx]
        except IndexError as e:
            raise self.IndexError(e) from e

    def __bool__(self):
        try:
            self[-1] if self.__reversed else self[0]
        except self.IndexError:
            return False
        return True

    def __len__(self):
        self.__exhaust()
        return len(self.__cache)

    def __reversed__(self):
        return type(self)(self.__iterable, reverse=not self.__reversed, _cache=self.__cache)

    def __copy__(self):
        return type(self)(self.__iterable, reverse=self.__reversed, _cache=self.__cache)

    def __repr__(self):
        # repr and str should mimic a list. So we exhaust the iterable
        return repr(self.exhaust())

    def __str__(self):
        return repr(self.exhaust())


class PagedList:

    class IndexError(IndexError):
        pass

    def __len__(self):
        # This is only useful for tests
        return len(self.getslice())

    def __init__(self, pagefunc, pagesize, use_cache=True):
        self._pagefunc = pagefunc
        self._pagesize = pagesize
        self._use_cache = use_cache
        self._cache = {}

    def getpage(self, pagenum):
        page_results = self._cache.get(pagenum)
        if page_results is None:
            page_results = list(self._pagefunc(pagenum))
        if self._use_cache:
            self._cache[pagenum] = page_results
        return page_results

    def getslice(self, start=0, end=None):
        return list(self._getslice(start, end))

    def _getslice(self, start, end):
        raise NotImplementedError('This method must be implemented by subclasses')

    def __getitem__(self, idx):
        # NOTE: cache must be enabled if this is used
        if not isinstance(idx, int) or idx < 0:
            raise TypeError('indices must be non-negative integers')
        entries = self.getslice(idx, idx + 1)
        if not entries:
            raise self.IndexError()
        return entries[0]


class OnDemandPagedList(PagedList):
    def _getslice(self, start, end):
        for pagenum in itertools.count(start // self._pagesize):
            firstid = pagenum * self._pagesize
            nextfirstid = pagenum * self._pagesize + self._pagesize
            if start >= nextfirstid:
                continue

            startv = (
                start % self._pagesize
                if firstid <= start < nextfirstid
                else 0)
            endv = (
                ((end - 1) % self._pagesize) + 1
                if (end is not None and firstid <= end <= nextfirstid)
                else None)

            page_results = self.getpage(pagenum)
            if startv != 0 or endv is not None:
                page_results = page_results[startv:endv]
            yield from page_results

            # A little optimization - if current page is not "full", ie. does
            # not contain page_size videos then we can assume that this page
            # is the last one - there are no more ids on further pages -
            # i.e. no need to query again.
            if len(page_results) + startv < self._pagesize:
                break

            # If we got the whole page, but the next page is not interesting,
            # break out early as well
            if end == nextfirstid:
                break


class InAdvancePagedList(PagedList):
    def __init__(self, pagefunc, pagecount, pagesize):
        self._pagecount = pagecount
        PagedList.__init__(self, pagefunc, pagesize, True)

    def _getslice(self, start, end):
        start_page = start // self._pagesize
        end_page = self._pagecount if end is None else min(self._pagecount, end // self._pagesize + 1)
        skip_elems = start - start_page * self._pagesize
        only_more = None if end is None else end - start
        for pagenum in range(start_page, end_page):
            page_results = self.getpage(pagenum)
            if skip_elems:
                page_results = page_results[skip_elems:]
                skip_elems = None
            if only_more is not None:
                if len(page_results) < only_more:
                    only_more -= len(page_results)
                else:
                    yield from page_results[:only_more]
                    break
            yield from page_results


def lowercase_escape(s):
    unicode_escape = codecs.getdecoder('unicode_escape')
    return re.sub(
        r'\\u[0-9a-fA-F]{4}',
        lambda m: unicode_escape(m.group(0))[0],
        s)


def escape_rfc3986(s):
    """Escape non-ASCII characters as suggested by RFC 3986"""
    return compat_urllib_parse.quote(s, b"%/;:@&=+$,!~*'()?#[]")


def escape_url(url):
    """Escape URL as suggested by RFC 3986"""
    url_parsed = compat_urllib_parse_urlparse(url)
    return url_parsed._replace(
        netloc=url_parsed.netloc.encode('idna').decode('ascii'),
        path=escape_rfc3986(url_parsed.path),
        params=escape_rfc3986(url_parsed.params),
        query=escape_rfc3986(url_parsed.query),
        fragment=escape_rfc3986(url_parsed.fragment)
    ).geturl()


def parse_qs(url):
    return compat_parse_qs(compat_urllib_parse_urlparse(url).query)


def read_batch_urls(batch_fd):
    def fixup(url):
        if not isinstance(url, compat_str):
            url = url.decode('utf-8', 'replace')
        BOM_UTF8 = ('\xef\xbb\xbf', '\ufeff')
        for bom in BOM_UTF8:
            if url.startswith(bom):
                url = url[len(bom):]
        url = url.lstrip()
        if not url or url.startswith(('#', ';', ']')):
            return False
        # "#" cannot be stripped out since it is part of the URI
        # However, it can be safely stipped out if follwing a whitespace
        return re.split(r'\s#', url, 1)[0].rstrip()

    with contextlib.closing(batch_fd) as fd:
        return [url for url in map(fixup, fd) if url]


def urlencode_postdata(*args, **kargs):
    return compat_urllib_parse_urlencode(*args, **kargs).encode('ascii')


def update_url_query(url, query):
    if not query:
        return url
    parsed_url = compat_urlparse.urlparse(url)
    qs = compat_parse_qs(parsed_url.query)
    qs.update(query)
    return compat_urlparse.urlunparse(parsed_url._replace(
        query=compat_urllib_parse_urlencode(qs, True)))


def _multipart_encode_impl(data, boundary):
    content_type = 'multipart/form-data; boundary=%s' % boundary

    out = b''
    for k, v in data.items():
        out += b'--' + boundary.encode('ascii') + b'\r\n'
        if isinstance(k, compat_str):
            k = k.encode('utf-8')
        if isinstance(v, compat_str):
            v = v.encode('utf-8')
        # RFC 2047 requires non-ASCII field names to be encoded, while RFC 7578
        # suggests sending UTF-8 directly. Firefox sends UTF-8, too
        content = b'Content-Disposition: form-data; name="' + k + b'"\r\n\r\n' + v + b'\r\n'
        if boundary.encode('ascii') in content:
            raise ValueError('Boundary overlaps with data')
        out += content

    out += b'--' + boundary.encode('ascii') + b'--\r\n'

    return out, content_type


def multipart_encode(data, boundary=None):
    '''
    Encode a dict to RFC 7578-compliant form-data

    data:
        A dict where keys and values can be either Unicode or bytes-like
        objects.
    boundary:
        If specified a Unicode object, it's used as the boundary. Otherwise
        a random boundary is generated.

    Reference: https://tools.ietf.org/html/rfc7578
    '''
    has_specified_boundary = boundary is not None

    while True:
        if boundary is None:
            boundary = '---------------' + str(random.randrange(0x0fffffff, 0xffffffff))

        try:
            out, content_type = _multipart_encode_impl(data, boundary)
            break
        except ValueError:
            if has_specified_boundary:
                raise
            boundary = None

    return out, content_type


def dict_get(d, key_or_keys, default=None, skip_false_values=True):
    if isinstance(key_or_keys, (list, tuple)):
        for key in key_or_keys:
            if key not in d or d[key] is None or skip_false_values and not d[key]:
                continue
            return d[key]
        return default
    return d.get(key_or_keys, default)


def try_get(src, getter, expected_type=None):
    for get in variadic(getter):
        try:
            v = get(src)
        except (AttributeError, KeyError, TypeError, IndexError):
            pass
        else:
            if expected_type is None or isinstance(v, expected_type):
                return v


def merge_dicts(*dicts):
    merged = {}
    for a_dict in dicts:
        for k, v in a_dict.items():
            if v is None:
                continue
            if (k not in merged
                    or (isinstance(v, compat_str) and v
                        and isinstance(merged[k], compat_str)
                        and not merged[k])):
                merged[k] = v
    return merged


def encode_compat_str(string, encoding=preferredencoding(), errors='strict'):
    return string if isinstance(string, compat_str) else compat_str(string, encoding, errors)


US_RATINGS = {
    'G': 0,
    'PG': 10,
    'PG-13': 13,
    'R': 16,
    'NC': 18,
}


TV_PARENTAL_GUIDELINES = {
    'TV-Y': 0,
    'TV-Y7': 7,
    'TV-G': 0,
    'TV-PG': 0,
    'TV-14': 14,
    'TV-MA': 17,
}


def parse_age_limit(s):
    if type(s) == int:
        return s if 0 <= s <= 21 else None
    if not isinstance(s, str):
        return None
    m = re.match(r'^(?P<age>\d{1,2})\+?$', s)
    if m:
        return int(m.group('age'))
    s = s.upper()
    if s in US_RATINGS:
        return US_RATINGS[s]
    m = re.match(r'^TV[_-]?(%s)$' % '|'.join(k[3:] for k in TV_PARENTAL_GUIDELINES), s)
    if m:
        return TV_PARENTAL_GUIDELINES['TV-' + m.group(1)]
    return None


def strip_jsonp(code):
    return re.sub(
        r'''(?sx)^
            (?:window\.)?(?P<func_name>[a-zA-Z0-9_.$]*)
            (?:\s*&&\s*(?P=func_name))?
            \s*\(\s*(?P<callback_data>.*)\);?
            \s*?(?://[^\n]*)*$''',
        r'\g<callback_data>', code)


def js_to_json(code, vars={}):
    # vars is a dict of var, val pairs to substitute
    COMMENT_RE = r'/\*(?:(?!\*/).)*?\*/|//[^\n]*\n'
    SKIP_RE = r'\s*(?:{comment})?\s*'.format(comment=COMMENT_RE)
    INTEGER_TABLE = (
        (r'(?s)^(0[xX][0-9a-fA-F]+){skip}:?$'.format(skip=SKIP_RE), 16),
        (r'(?s)^(0+[0-7]+){skip}:?$'.format(skip=SKIP_RE), 8),
    )

    def fix_kv(m):
        v = m.group(0)
        if v in ('true', 'false', 'null'):
            return v
        elif v in ('undefined', 'void 0'):
            return 'null'
        elif v.startswith('/*') or v.startswith('//') or v.startswith('!') or v == ',':
            return ""

        if v[0] in ("'", '"'):
            v = re.sub(r'(?s)\\.|"', lambda m: {
                '"': '\\"',
                "\\'": "'",
                '\\\n': '',
                '\\x': '\\u00',
            }.get(m.group(0), m.group(0)), v[1:-1])
        else:
            for regex, base in INTEGER_TABLE:
                im = re.match(regex, v)
                if im:
                    i = int(im.group(1), base)
                    return '"%d":' % i if v.endswith(':') else '%d' % i

            if v in vars:
                return vars[v]

        return '"%s"' % v

    return re.sub(r'''(?sx)
        "(?:[^"\\]*(?:\\\\|\\['"nurtbfx/\n]))*[^"\\]*"|
        '(?:[^'\\]*(?:\\\\|\\['"nurtbfx/\n]))*[^'\\]*'|
        {comment}|,(?={skip}[\]}}])|
        void\s0|(?:(?<![0-9])[eE]|[a-df-zA-DF-Z_$])[.a-zA-Z_$0-9]*|
        \b(?:0[xX][0-9a-fA-F]+|0+[0-7]+)(?:{skip}:)?|
        [0-9]+(?={skip}:)|
        !+
        '''.format(comment=COMMENT_RE, skip=SKIP_RE), fix_kv, code)


def qualities(quality_ids):
    """ Get a numeric quality value out of a list of possible values """
    def q(qid):
        try:
            return quality_ids.index(qid)
        except ValueError:
            return -1
    return q


POSTPROCESS_WHEN = {'pre_process', 'before_dl', 'after_move', 'post_process', 'after_video', 'playlist'}


DEFAULT_OUTTMPL = {
    'default': '%(title)s [%(id)s].%(ext)s',
    'chapter': '%(title)s - %(section_number)03d %(section_title)s [%(id)s].%(ext)s',
}
OUTTMPL_TYPES = {
    'chapter': None,
    'subtitle': None,
    'thumbnail': None,
    'description': 'description',
    'annotation': 'annotations.xml',
    'infojson': 'info.json',
    'link': None,
    'pl_video': None,
    'pl_thumbnail': None,
    'pl_description': 'description',
    'pl_infojson': 'info.json',
}

# As of [1] format syntax is:
#  %[mapping_key][conversion_flags][minimum_width][.precision][length_modifier]type
# 1. https://docs.python.org/2/library/stdtypes.html#string-formatting
STR_FORMAT_RE_TMPL = r'''(?x)
    (?<!%)(?P<prefix>(?:%%)*)
    %
    (?P<has_key>\((?P<key>{0})\))?
    (?P<format>
        (?P<conversion>[#0\-+ ]+)?
        (?P<min_width>\d+)?
        (?P<precision>\.\d+)?
        (?P<len_mod>[hlL])?  # unused in python
        {1}  # conversion type
    )
'''


STR_FORMAT_TYPES = 'diouxXeEfFgGcrs'


def limit_length(s, length):
    """ Add ellipses to overly long strings """
    if s is None:
        return None
    ELLIPSES = '...'
    if len(s) > length:
        return s[:length - len(ELLIPSES)] + ELLIPSES
    return s


def version_tuple(v):
    return tuple(int(e) for e in re.split(r'[-.]', v))


def is_outdated_version(version, limit, assume_new=True):
    if not version:
        return not assume_new
    try:
        return version_tuple(version) < version_tuple(limit)
    except ValueError:
        return not assume_new


def ytdl_is_updateable():
    """ Returns if yt-dlp can be updated with -U """

    from .update import is_non_updateable

    return not is_non_updateable()


def args_to_str(args):
    # Get a short string representation for a subprocess command
    return ' '.join(compat_shlex_quote(a) for a in args)


def error_to_compat_str(err):
    return str(err)


def mimetype2ext(mt):
    if mt is None:
        return None

    mt, _, params = mt.partition(';')
    mt = mt.strip()

    FULL_MAP = {
        'audio/mp4': 'm4a',
        # Per RFC 3003, audio/mpeg can be .mp1, .mp2 or .mp3. Here use .mp3 as
        # it's the most popular one
        'audio/mpeg': 'mp3',
        'audio/x-wav': 'wav',
        'audio/wav': 'wav',
        'audio/wave': 'wav',
    }

    ext = FULL_MAP.get(mt)
    if ext is not None:
        return ext

    SUBTYPE_MAP = {
        '3gpp': '3gp',
        'smptett+xml': 'tt',
        'ttaf+xml': 'dfxp',
        'ttml+xml': 'ttml',
        'x-flv': 'flv',
        'x-mp4-fragmented': 'mp4',
        'x-ms-sami': 'sami',
        'x-ms-wmv': 'wmv',
        'mpegurl': 'm3u8',
        'x-mpegurl': 'm3u8',
        'vnd.apple.mpegurl': 'm3u8',
        'dash+xml': 'mpd',
        'f4m+xml': 'f4m',
        'hds+xml': 'f4m',
        'vnd.ms-sstr+xml': 'ism',
        'quicktime': 'mov',
        'mp2t': 'ts',
        'x-wav': 'wav',
        'filmstrip+json': 'fs',
        'svg+xml': 'svg',
    }

    _, _, subtype = mt.rpartition('/')
    ext = SUBTYPE_MAP.get(subtype.lower())
    if ext is not None:
        return ext

    SUFFIX_MAP = {
        'json': 'json',
        'xml': 'xml',
        'zip': 'zip',
        'gzip': 'gz',
    }

    _, _, suffix = subtype.partition('+')
    ext = SUFFIX_MAP.get(suffix)
    if ext is not None:
        return ext

    return subtype.replace('+', '.')


def ext2mimetype(ext_or_url):
    if not ext_or_url:
        return None
    if '.' not in ext_or_url:
        ext_or_url = f'file.{ext_or_url}'
    return mimetypes.guess_type(ext_or_url)[0]


def parse_codecs(codecs_str):
    # http://tools.ietf.org/html/rfc6381
    if not codecs_str:
        return {}
    split_codecs = list(filter(None, map(
        str.strip, codecs_str.strip().strip(',').split(','))))
    vcodec, acodec, tcodec, hdr = None, None, None, None
    for full_codec in split_codecs:
        parts = full_codec.split('.')
        codec = parts[0].replace('0', '')
        if codec in ('avc1', 'avc2', 'avc3', 'avc4', 'vp9', 'vp8', 'hev1', 'hev2',
                     'h263', 'h264', 'mp4v', 'hvc1', 'av1', 'theora', 'dvh1', 'dvhe'):
            if not vcodec:
                vcodec = '.'.join(parts[:4]) if codec in ('vp9', 'av1', 'hvc1') else full_codec
                if codec in ('dvh1', 'dvhe'):
                    hdr = 'DV'
                elif codec == 'av1' and len(parts) > 3 and parts[3] == '10':
                    hdr = 'HDR10'
                elif full_codec.replace('0', '').startswith('vp9.2'):
                    hdr = 'HDR10'
        elif codec in ('flac', 'mp4a', 'opus', 'vorbis', 'mp3', 'aac', 'ac-3', 'ec-3', 'eac3', 'dtsc', 'dtse', 'dtsh', 'dtsl'):
            if not acodec:
                acodec = full_codec
        elif codec in ('stpp', 'wvtt',):
            if not tcodec:
                tcodec = full_codec
        else:
            write_string('WARNING: Unknown codec %s\n' % full_codec, sys.stderr)
    if vcodec or acodec or tcodec:
        return {
            'vcodec': vcodec or 'none',
            'acodec': acodec or 'none',
            'dynamic_range': hdr,
            **({'tcodec': tcodec} if tcodec is not None else {}),
        }
    elif len(split_codecs) == 2:
        return {
            'vcodec': split_codecs[0],
            'acodec': split_codecs[1],
        }
    return {}


def urlhandle_detect_ext(url_handle):
    getheader = url_handle.headers.get

    cd = getheader('Content-Disposition')
    if cd:
        m = re.match(r'attachment;\s*filename="(?P<filename>[^"]+)"', cd)
        if m:
            e = determine_ext(m.group('filename'), default_ext=None)
            if e:
                return e

    return mimetype2ext(getheader('Content-Type'))


def encode_data_uri(data, mime_type):
    return 'data:%s;base64,%s' % (mime_type, base64.b64encode(data).decode('ascii'))


def age_restricted(content_limit, age_limit):
    """ Returns True iff the content should be blocked """

    if age_limit is None:  # No limit set
        return False
    if content_limit is None:
        return False  # Content available for everyone
    return age_limit < content_limit


def is_html(first_bytes):
    """ Detect whether a file contains HTML by examining its first bytes. """

    BOMS = [
        (b'\xef\xbb\xbf', 'utf-8'),
        (b'\x00\x00\xfe\xff', 'utf-32-be'),
        (b'\xff\xfe\x00\x00', 'utf-32-le'),
        (b'\xff\xfe', 'utf-16-le'),
        (b'\xfe\xff', 'utf-16-be'),
    ]
    for bom, enc in BOMS:
        if first_bytes.startswith(bom):
            s = first_bytes[len(bom):].decode(enc, 'replace')
            break
    else:
        s = first_bytes.decode('utf-8', 'replace')

    return re.match(r'^\s*<', s)


def determine_protocol(info_dict):
    protocol = info_dict.get('protocol')
    if protocol is not None:
        return protocol

    url = sanitize_url(info_dict['url'])
    if url.startswith('rtmp'):
        return 'rtmp'
    elif url.startswith('mms'):
        return 'mms'
    elif url.startswith('rtsp'):
        return 'rtsp'

    ext = determine_ext(url)
    if ext == 'm3u8':
        return 'm3u8'
    elif ext == 'f4m':
        return 'f4m'

    return compat_urllib_parse_urlparse(url).scheme


def render_table(header_row, data, delim=False, extra_gap=0, hide_empty=False):
    """ Render a list of rows, each as a list of values.
    Text after a \t will be right aligned """
    def width(string):
        return len(remove_terminal_sequences(string).replace('\t', ''))

    def get_max_lens(table):
        return [max(width(str(v)) for v in col) for col in zip(*table)]

    def filter_using_list(row, filterArray):
        return [col for take, col in itertools.zip_longest(filterArray, row, fillvalue=True) if take]

    max_lens = get_max_lens(data) if hide_empty else []
    header_row = filter_using_list(header_row, max_lens)
    data = [filter_using_list(row, max_lens) for row in data]

    table = [header_row] + data
    max_lens = get_max_lens(table)
    extra_gap += 1
    if delim:
        table = [header_row, [delim * (ml + extra_gap) for ml in max_lens]] + data
        table[1][-1] = table[1][-1][:-extra_gap]  # Remove extra_gap from end of delimiter
    for row in table:
        for pos, text in enumerate(map(str, row)):
            if '\t' in text:
                row[pos] = text.replace('\t', ' ' * (max_lens[pos] - width(text))) + ' ' * extra_gap
            else:
                row[pos] = text + ' ' * (max_lens[pos] - width(text) + extra_gap)
    ret = '\n'.join(''.join(row).rstrip() for row in table)
    return ret


def _match_one(filter_part, dct, incomplete):
    # TODO: Generalize code with YoutubeDL._build_format_filter
    STRING_OPERATORS = {
        '*=': operator.contains,
        '^=': lambda attr, value: attr.startswith(value),
        '$=': lambda attr, value: attr.endswith(value),
        '~=': lambda attr, value: re.search(value, attr),
    }
    COMPARISON_OPERATORS = {
        **STRING_OPERATORS,
        '<=': operator.le,  # "<=" must be defined above "<"
        '<': operator.lt,
        '>=': operator.ge,
        '>': operator.gt,
        '=': operator.eq,
    }

    operator_rex = re.compile(r'''(?x)\s*
        (?P<key>[a-z_]+)
        \s*(?P<negation>!\s*)?(?P<op>%s)(?P<none_inclusive>\s*\?)?\s*
        (?:
            (?P<quote>["\'])(?P<quotedstrval>.+?)(?P=quote)|
            (?P<strval>.+?)
        )
        \s*$
        ''' % '|'.join(map(re.escape, COMPARISON_OPERATORS.keys())))
    m = operator_rex.search(filter_part)
    if m:
        m = m.groupdict()
        unnegated_op = COMPARISON_OPERATORS[m['op']]
        if m['negation']:
            op = lambda attr, value: not unnegated_op(attr, value)
        else:
            op = unnegated_op
        comparison_value = m['quotedstrval'] or m['strval'] or m['intval']
        if m['quote']:
            comparison_value = comparison_value.replace(r'\%s' % m['quote'], m['quote'])
        actual_value = dct.get(m['key'])
        numeric_comparison = None
        if isinstance(actual_value, (int, float)):
            # If the original field is a string and matching comparisonvalue is
            # a number we should respect the origin of the original field
            # and process comparison value as a string (see
            # https://github.com/ytdl-org/youtube-dl/issues/11082)
            try:
                numeric_comparison = int(comparison_value)
            except ValueError:
                numeric_comparison = parse_filesize(comparison_value)
                if numeric_comparison is None:
                    numeric_comparison = parse_filesize(f'{comparison_value}B')
                if numeric_comparison is None:
                    numeric_comparison = parse_duration(comparison_value)
        if numeric_comparison is not None and m['op'] in STRING_OPERATORS:
            raise ValueError('Operator %s only supports string values!' % m['op'])
        if actual_value is None:
            return incomplete or m['none_inclusive']
        return op(actual_value, comparison_value if numeric_comparison is None else numeric_comparison)

    UNARY_OPERATORS = {
        '': lambda v: (v is True) if isinstance(v, bool) else (v is not None),
        '!': lambda v: (v is False) if isinstance(v, bool) else (v is None),
    }
    operator_rex = re.compile(r'''(?x)\s*
        (?P<op>%s)\s*(?P<key>[a-z_]+)
        \s*$
        ''' % '|'.join(map(re.escape, UNARY_OPERATORS.keys())))
    m = operator_rex.search(filter_part)
    if m:
        op = UNARY_OPERATORS[m.group('op')]
        actual_value = dct.get(m.group('key'))
        if incomplete and actual_value is None:
            return True
        return op(actual_value)

    raise ValueError('Invalid filter part %r' % filter_part)


def match_str(filter_str, dct, incomplete=False):
    """ Filter a dictionary with a simple string syntax. Returns True (=passes filter) or false
        When incomplete, all conditions passes on missing fields
    """
    return all(
        _match_one(filter_part.replace(r'\&', '&'), dct, incomplete)
        for filter_part in re.split(r'(?<!\\)&', filter_str))


def match_filter_func(filter_str):
    def _match_func(info_dict, *args, **kwargs):
        if match_str(filter_str, info_dict, *args, **kwargs):
            return None
        else:
            video_title = info_dict.get('title', info_dict.get('id', 'video'))
            return '%s does not pass filter %s, skipping ..' % (video_title, filter_str)
    return _match_func


def parse_dfxp_time_expr(time_expr):
    if not time_expr:
        return

    mobj = re.match(r'^(?P<time_offset>\d+(?:\.\d+)?)s?$', time_expr)
    if mobj:
        return float(mobj.group('time_offset'))

    mobj = re.match(r'^(\d+):(\d\d):(\d\d(?:(?:\.|:)\d+)?)$', time_expr)
    if mobj:
        return 3600 * int(mobj.group(1)) + 60 * int(mobj.group(2)) + float(mobj.group(3).replace(':', '.'))


def srt_subtitles_timecode(seconds):
    return '%02d:%02d:%02d,%03d' % timetuple_from_msec(seconds * 1000)


def ass_subtitles_timecode(seconds):
    time = timetuple_from_msec(seconds * 1000)
    return '%01d:%02d:%02d.%02d' % (*time[:-1], time.milliseconds / 10)


def dfxp2srt(dfxp_data):
    '''
    @param dfxp_data A bytes-like object containing DFXP data
    @returns A unicode object containing converted SRT data
    '''
    LEGACY_NAMESPACES = (
        (b'http://www.w3.org/ns/ttml', [
            b'http://www.w3.org/2004/11/ttaf1',
            b'http://www.w3.org/2006/04/ttaf1',
            b'http://www.w3.org/2006/10/ttaf1',
        ]),
        (b'http://www.w3.org/ns/ttml#styling', [
            b'http://www.w3.org/ns/ttml#style',
        ]),
    )

    SUPPORTED_STYLING = [
        'color',
        'fontFamily',
        'fontSize',
        'fontStyle',
        'fontWeight',
        'textDecoration'
    ]

    _x = functools.partial(xpath_with_ns, ns_map={
        'xml': 'http://www.w3.org/XML/1998/namespace',
        'ttml': 'http://www.w3.org/ns/ttml',
        'tts': 'http://www.w3.org/ns/ttml#styling',
    })

    styles = {}
    default_style = {}

    class TTMLPElementParser(object):
        _out = ''
        _unclosed_elements = []
        _applied_styles = []

        def start(self, tag, attrib):
            if tag in (_x('ttml:br'), 'br'):
                self._out += '\n'
            else:
                unclosed_elements = []
                style = {}
                element_style_id = attrib.get('style')
                if default_style:
                    style.update(default_style)
                if element_style_id:
                    style.update(styles.get(element_style_id, {}))
                for prop in SUPPORTED_STYLING:
                    prop_val = attrib.get(_x('tts:' + prop))
                    if prop_val:
                        style[prop] = prop_val
                if style:
                    font = ''
                    for k, v in sorted(style.items()):
                        if self._applied_styles and self._applied_styles[-1].get(k) == v:
                            continue
                        if k == 'color':
                            font += ' color="%s"' % v
                        elif k == 'fontSize':
                            font += ' size="%s"' % v
                        elif k == 'fontFamily':
                            font += ' face="%s"' % v
                        elif k == 'fontWeight' and v == 'bold':
                            self._out += '<b>'
                            unclosed_elements.append('b')
                        elif k == 'fontStyle' and v == 'italic':
                            self._out += '<i>'
                            unclosed_elements.append('i')
                        elif k == 'textDecoration' and v == 'underline':
                            self._out += '<u>'
                            unclosed_elements.append('u')
                    if font:
                        self._out += '<font' + font + '>'
                        unclosed_elements.append('font')
                    applied_style = {}
                    if self._applied_styles:
                        applied_style.update(self._applied_styles[-1])
                    applied_style.update(style)
                    self._applied_styles.append(applied_style)
                self._unclosed_elements.append(unclosed_elements)

        def end(self, tag):
            if tag not in (_x('ttml:br'), 'br'):
                unclosed_elements = self._unclosed_elements.pop()
                for element in reversed(unclosed_elements):
                    self._out += '</%s>' % element
                if unclosed_elements and self._applied_styles:
                    self._applied_styles.pop()

        def data(self, data):
            self._out += data

        def close(self):
            return self._out.strip()

    def parse_node(node):
        target = TTMLPElementParser()
        parser = xml.etree.ElementTree.XMLParser(target=target)
        parser.feed(xml.etree.ElementTree.tostring(node))
        return parser.close()

    for k, v in LEGACY_NAMESPACES:
        for ns in v:
            dfxp_data = dfxp_data.replace(ns, k)

    dfxp = compat_etree_fromstring(dfxp_data)
    out = []
    paras = dfxp.findall(_x('.//ttml:p')) or dfxp.findall('.//p')

    if not paras:
        raise ValueError('Invalid dfxp/TTML subtitle')

    repeat = False
    while True:
        for style in dfxp.findall(_x('.//ttml:style')):
            style_id = style.get('id') or style.get(_x('xml:id'))
            if not style_id:
                continue
            parent_style_id = style.get('style')
            if parent_style_id:
                if parent_style_id not in styles:
                    repeat = True
                    continue
                styles[style_id] = styles[parent_style_id].copy()
            for prop in SUPPORTED_STYLING:
                prop_val = style.get(_x('tts:' + prop))
                if prop_val:
                    styles.setdefault(style_id, {})[prop] = prop_val
        if repeat:
            repeat = False
        else:
            break

    for p in ('body', 'div'):
        ele = xpath_element(dfxp, [_x('.//ttml:' + p), './/' + p])
        if ele is None:
            continue
        style = styles.get(ele.get('style'))
        if not style:
            continue
        default_style.update(style)

    for para, index in zip(paras, itertools.count(1)):
        begin_time = parse_dfxp_time_expr(para.attrib.get('begin'))
        end_time = parse_dfxp_time_expr(para.attrib.get('end'))
        dur = parse_dfxp_time_expr(para.attrib.get('dur'))
        if begin_time is None:
            continue
        if not end_time:
            if not dur:
                continue
            end_time = begin_time + dur
        out.append('%d\n%s --> %s\n%s\n\n' % (
            index,
            srt_subtitles_timecode(begin_time),
            srt_subtitles_timecode(end_time),
            parse_node(para)))

    return ''.join(out)


def cli_option(params, command_option, param):
    param = params.get(param)
    if param:
        param = compat_str(param)
    return [command_option, param] if param is not None else []


def cli_bool_option(params, command_option, param, true_value='true', false_value='false', separator=None):
    param = params.get(param)
    if param is None:
        return []
    assert isinstance(param, bool)
    if separator:
        return [command_option + separator + (true_value if param else false_value)]
    return [command_option, true_value if param else false_value]


def cli_valueless_option(params, command_option, param, expected_value=True):
    param = params.get(param)
    return [command_option] if param == expected_value else []


def cli_configuration_args(argdict, keys, default=[], use_compat=True):
    if isinstance(argdict, (list, tuple)):  # for backward compatibility
        if use_compat:
            return argdict
        else:
            argdict = None
    if argdict is None:
        return default
    assert isinstance(argdict, dict)

    assert isinstance(keys, (list, tuple))
    for key_list in keys:
        arg_list = list(filter(
            lambda x: x is not None,
            [argdict.get(key.lower()) for key in variadic(key_list)]))
        if arg_list:
            return [arg for args in arg_list for arg in args]
    return default


def _configuration_args(main_key, argdict, exe, keys=None, default=[], use_compat=True):
    main_key, exe = main_key.lower(), exe.lower()
    root_key = exe if main_key == exe else f'{main_key}+{exe}'
    keys = [f'{root_key}{k}' for k in (keys or [''])]
    if root_key in keys:
        if main_key != exe:
            keys.append((main_key, exe))
        keys.append('default')
    else:
        use_compat = False
    return cli_configuration_args(argdict, keys, default, use_compat)


class ISO639Utils(object):
    # See http://www.loc.gov/standards/iso639-2/ISO-639-2_utf-8.txt
    _lang_map = {
        'aa': 'aar',
        'ab': 'abk',
        'ae': 'ave',
        'af': 'afr',
        'ak': 'aka',
        'am': 'amh',
        'an': 'arg',
        'ar': 'ara',
        'as': 'asm',
        'av': 'ava',
        'ay': 'aym',
        'az': 'aze',
        'ba': 'bak',
        'be': 'bel',
        'bg': 'bul',
        'bh': 'bih',
        'bi': 'bis',
        'bm': 'bam',
        'bn': 'ben',
        'bo': 'bod',
        'br': 'bre',
        'bs': 'bos',
        'ca': 'cat',
        'ce': 'che',
        'ch': 'cha',
        'co': 'cos',
        'cr': 'cre',
        'cs': 'ces',
        'cu': 'chu',
        'cv': 'chv',
        'cy': 'cym',
        'da': 'dan',
        'de': 'deu',
        'dv': 'div',
        'dz': 'dzo',
        'ee': 'ewe',
        'el': 'ell',
        'en': 'eng',
        'eo': 'epo',
        'es': 'spa',
        'et': 'est',
        'eu': 'eus',
        'fa': 'fas',
        'ff': 'ful',
        'fi': 'fin',
        'fj': 'fij',
        'fo': 'fao',
        'fr': 'fra',
        'fy': 'fry',
        'ga': 'gle',
        'gd': 'gla',
        'gl': 'glg',
        'gn': 'grn',
        'gu': 'guj',
        'gv': 'glv',
        'ha': 'hau',
        'he': 'heb',
        'iw': 'heb',  # Replaced by he in 1989 revision
        'hi': 'hin',
        'ho': 'hmo',
        'hr': 'hrv',
        'ht': 'hat',
        'hu': 'hun',
        'hy': 'hye',
        'hz': 'her',
        'ia': 'ina',
        'id': 'ind',
        'in': 'ind',  # Replaced by id in 1989 revision
        'ie': 'ile',
        'ig': 'ibo',
        'ii': 'iii',
        'ik': 'ipk',
        'io': 'ido',
        'is': 'isl',
        'it': 'ita',
        'iu': 'iku',
        'ja': 'jpn',
        'jv': 'jav',
        'ka': 'kat',
        'kg': 'kon',
        'ki': 'kik',
        'kj': 'kua',
        'kk': 'kaz',
        'kl': 'kal',
        'km': 'khm',
        'kn': 'kan',
        'ko': 'kor',
        'kr': 'kau',
        'ks': 'kas',
        'ku': 'kur',
        'kv': 'kom',
        'kw': 'cor',
        'ky': 'kir',
        'la': 'lat',
        'lb': 'ltz',
        'lg': 'lug',
        'li': 'lim',
        'ln': 'lin',
        'lo': 'lao',
        'lt': 'lit',
        'lu': 'lub',
        'lv': 'lav',
        'mg': 'mlg',
        'mh': 'mah',
        'mi': 'mri',
        'mk': 'mkd',
        'ml': 'mal',
        'mn': 'mon',
        'mr': 'mar',
        'ms': 'msa',
        'mt': 'mlt',
        'my': 'mya',
        'na': 'nau',
        'nb': 'nob',
        'nd': 'nde',
        'ne': 'nep',
        'ng': 'ndo',
        'nl': 'nld',
        'nn': 'nno',
        'no': 'nor',
        'nr': 'nbl',
        'nv': 'nav',
        'ny': 'nya',
        'oc': 'oci',
        'oj': 'oji',
        'om': 'orm',
        'or': 'ori',
        'os': 'oss',
        'pa': 'pan',
        'pi': 'pli',
        'pl': 'pol',
        'ps': 'pus',
        'pt': 'por',
        'qu': 'que',
        'rm': 'roh',
        'rn': 'run',
        'ro': 'ron',
        'ru': 'rus',
        'rw': 'kin',
        'sa': 'san',
        'sc': 'srd',
        'sd': 'snd',
        'se': 'sme',
        'sg': 'sag',
        'si': 'sin',
        'sk': 'slk',
        'sl': 'slv',
        'sm': 'smo',
        'sn': 'sna',
        'so': 'som',
        'sq': 'sqi',
        'sr': 'srp',
        'ss': 'ssw',
        'st': 'sot',
        'su': 'sun',
        'sv': 'swe',
        'sw': 'swa',
        'ta': 'tam',
        'te': 'tel',
        'tg': 'tgk',
        'th': 'tha',
        'ti': 'tir',
        'tk': 'tuk',
        'tl': 'tgl',
        'tn': 'tsn',
        'to': 'ton',
        'tr': 'tur',
        'ts': 'tso',
        'tt': 'tat',
        'tw': 'twi',
        'ty': 'tah',
        'ug': 'uig',
        'uk': 'ukr',
        'ur': 'urd',
        'uz': 'uzb',
        've': 'ven',
        'vi': 'vie',
        'vo': 'vol',
        'wa': 'wln',
        'wo': 'wol',
        'xh': 'xho',
        'yi': 'yid',
        'ji': 'yid',  # Replaced by yi in 1989 revision
        'yo': 'yor',
        'za': 'zha',
        'zh': 'zho',
        'zu': 'zul',
    }

    @classmethod
    def short2long(cls, code):
        """Convert language code from ISO 639-1 to ISO 639-2/T"""
        return cls._lang_map.get(code[:2])

    @classmethod
    def long2short(cls, code):
        """Convert language code from ISO 639-2/T to ISO 639-1"""
        for short_name, long_name in cls._lang_map.items():
            if long_name == code:
                return short_name


class ISO3166Utils(object):
    # From http://data.okfn.org/data/core/country-list
    _country_map = {
        'AF': 'Afghanistan',
        'AX': 'Åland Islands',
        'AL': 'Albania',
        'DZ': 'Algeria',
        'AS': 'American Samoa',
        'AD': 'Andorra',
        'AO': 'Angola',
        'AI': 'Anguilla',
        'AQ': 'Antarctica',
        'AG': 'Antigua and Barbuda',
        'AR': 'Argentina',
        'AM': 'Armenia',
        'AW': 'Aruba',
        'AU': 'Australia',
        'AT': 'Austria',
        'AZ': 'Azerbaijan',
        'BS': 'Bahamas',
        'BH': 'Bahrain',
        'BD': 'Bangladesh',
        'BB': 'Barbados',
        'BY': 'Belarus',
        'BE': 'Belgium',
        'BZ': 'Belize',
        'BJ': 'Benin',
        'BM': 'Bermuda',
        'BT': 'Bhutan',
        'BO': 'Bolivia, Plurinational State of',
        'BQ': 'Bonaire, Sint Eustatius and Saba',
        'BA': 'Bosnia and Herzegovina',
        'BW': 'Botswana',
        'BV': 'Bouvet Island',
        'BR': 'Brazil',
        'IO': 'British Indian Ocean Territory',
        'BN': 'Brunei Darussalam',
        'BG': 'Bulgaria',
        'BF': 'Burkina Faso',
        'BI': 'Burundi',
        'KH': 'Cambodia',
        'CM': 'Cameroon',
        'CA': 'Canada',
        'CV': 'Cape Verde',
        'KY': 'Cayman Islands',
        'CF': 'Central African Republic',
        'TD': 'Chad',
        'CL': 'Chile',
        'CN': 'China',
        'CX': 'Christmas Island',
        'CC': 'Cocos (Keeling) Islands',
        'CO': 'Colombia',
        'KM': 'Comoros',
        'CG': 'Congo',
        'CD': 'Congo, the Democratic Republic of the',
        'CK': 'Cook Islands',
        'CR': 'Costa Rica',
        'CI': 'Côte d\'Ivoire',
        'HR': 'Croatia',
        'CU': 'Cuba',
        'CW': 'Curaçao',
        'CY': 'Cyprus',
        'CZ': 'Czech Republic',
        'DK': 'Denmark',
        'DJ': 'Djibouti',
        'DM': 'Dominica',
        'DO': 'Dominican Republic',
        'EC': 'Ecuador',
        'EG': 'Egypt',
        'SV': 'El Salvador',
        'GQ': 'Equatorial Guinea',
        'ER': 'Eritrea',
        'EE': 'Estonia',
        'ET': 'Ethiopia',
        'FK': 'Falkland Islands (Malvinas)',
        'FO': 'Faroe Islands',
        'FJ': 'Fiji',
        'FI': 'Finland',
        'FR': 'France',
        'GF': 'French Guiana',
        'PF': 'French Polynesia',
        'TF': 'French Southern Territories',
        'GA': 'Gabon',
        'GM': 'Gambia',
        'GE': 'Georgia',
        'DE': 'Germany',
        'GH': 'Ghana',
        'GI': 'Gibraltar',
        'GR': 'Greece',
        'GL': 'Greenland',
        'GD': 'Grenada',
        'GP': 'Guadeloupe',
        'GU': 'Guam',
        'GT': 'Guatemala',
        'GG': 'Guernsey',
        'GN': 'Guinea',
        'GW': 'Guinea-Bissau',
        'GY': 'Guyana',
        'HT': 'Haiti',
        'HM': 'Heard Island and McDonald Islands',
        'VA': 'Holy See (Vatican City State)',
        'HN': 'Honduras',
        'HK': 'Hong Kong',
        'HU': 'Hungary',
        'IS': 'Iceland',
        'IN': 'India',
        'ID': 'Indonesia',
        'IR': 'Iran, Islamic Republic of',
        'IQ': 'Iraq',
        'IE': 'Ireland',
        'IM': 'Isle of Man',
        'IL': 'Israel',
        'IT': 'Italy',
        'JM': 'Jamaica',
        'JP': 'Japan',
        'JE': 'Jersey',
        'JO': 'Jordan',
        'KZ': 'Kazakhstan',
        'KE': 'Kenya',
        'KI': 'Kiribati',
        'KP': 'Korea, Democratic People\'s Republic of',
        'KR': 'Korea, Republic of',
        'KW': 'Kuwait',
        'KG': 'Kyrgyzstan',
        'LA': 'Lao People\'s Democratic Republic',
        'LV': 'Latvia',
        'LB': 'Lebanon',
        'LS': 'Lesotho',
        'LR': 'Liberia',
        'LY': 'Libya',
        'LI': 'Liechtenstein',
        'LT': 'Lithuania',
        'LU': 'Luxembourg',
        'MO': 'Macao',
        'MK': 'Macedonia, the Former Yugoslav Republic of',
        'MG': 'Madagascar',
        'MW': 'Malawi',
        'MY': 'Malaysia',
        'MV': 'Maldives',
        'ML': 'Mali',
        'MT': 'Malta',
        'MH': 'Marshall Islands',
        'MQ': 'Martinique',
        'MR': 'Mauritania',
        'MU': 'Mauritius',
        'YT': 'Mayotte',
        'MX': 'Mexico',
        'FM': 'Micronesia, Federated States of',
        'MD': 'Moldova, Republic of',
        'MC': 'Monaco',
        'MN': 'Mongolia',
        'ME': 'Montenegro',
        'MS': 'Montserrat',
        'MA': 'Morocco',
        'MZ': 'Mozambique',
        'MM': 'Myanmar',
        'NA': 'Namibia',
        'NR': 'Nauru',
        'NP': 'Nepal',
        'NL': 'Netherlands',
        'NC': 'New Caledonia',
        'NZ': 'New Zealand',
        'NI': 'Nicaragua',
        'NE': 'Niger',
        'NG': 'Nigeria',
        'NU': 'Niue',
        'NF': 'Norfolk Island',
        'MP': 'Northern Mariana Islands',
        'NO': 'Norway',
        'OM': 'Oman',
        'PK': 'Pakistan',
        'PW': 'Palau',
        'PS': 'Palestine, State of',
        'PA': 'Panama',
        'PG': 'Papua New Guinea',
        'PY': 'Paraguay',
        'PE': 'Peru',
        'PH': 'Philippines',
        'PN': 'Pitcairn',
        'PL': 'Poland',
        'PT': 'Portugal',
        'PR': 'Puerto Rico',
        'QA': 'Qatar',
        'RE': 'Réunion',
        'RO': 'Romania',
        'RU': 'Russian Federation',
        'RW': 'Rwanda',
        'BL': 'Saint Barthélemy',
        'SH': 'Saint Helena, Ascension and Tristan da Cunha',
        'KN': 'Saint Kitts and Nevis',
        'LC': 'Saint Lucia',
        'MF': 'Saint Martin (French part)',
        'PM': 'Saint Pierre and Miquelon',
        'VC': 'Saint Vincent and the Grenadines',
        'WS': 'Samoa',
        'SM': 'San Marino',
        'ST': 'Sao Tome and Principe',
        'SA': 'Saudi Arabia',
        'SN': 'Senegal',
        'RS': 'Serbia',
        'SC': 'Seychelles',
        'SL': 'Sierra Leone',
        'SG': 'Singapore',
        'SX': 'Sint Maarten (Dutch part)',
        'SK': 'Slovakia',
        'SI': 'Slovenia',
        'SB': 'Solomon Islands',
        'SO': 'Somalia',
        'ZA': 'South Africa',
        'GS': 'South Georgia and the South Sandwich Islands',
        'SS': 'South Sudan',
        'ES': 'Spain',
        'LK': 'Sri Lanka',
        'SD': 'Sudan',
        'SR': 'Suriname',
        'SJ': 'Svalbard and Jan Mayen',
        'SZ': 'Swaziland',
        'SE': 'Sweden',
        'CH': 'Switzerland',
        'SY': 'Syrian Arab Republic',
        'TW': 'Taiwan, Province of China',
        'TJ': 'Tajikistan',
        'TZ': 'Tanzania, United Republic of',
        'TH': 'Thailand',
        'TL': 'Timor-Leste',
        'TG': 'Togo',
        'TK': 'Tokelau',
        'TO': 'Tonga',
        'TT': 'Trinidad and Tobago',
        'TN': 'Tunisia',
        'TR': 'Turkey',
        'TM': 'Turkmenistan',
        'TC': 'Turks and Caicos Islands',
        'TV': 'Tuvalu',
        'UG': 'Uganda',
        'UA': 'Ukraine',
        'AE': 'United Arab Emirates',
        'GB': 'United Kingdom',
        'US': 'United States',
        'UM': 'United States Minor Outlying Islands',
        'UY': 'Uruguay',
        'UZ': 'Uzbekistan',
        'VU': 'Vanuatu',
        'VE': 'Venezuela, Bolivarian Republic of',
        'VN': 'Viet Nam',
        'VG': 'Virgin Islands, British',
        'VI': 'Virgin Islands, U.S.',
        'WF': 'Wallis and Futuna',
        'EH': 'Western Sahara',
        'YE': 'Yemen',
        'ZM': 'Zambia',
        'ZW': 'Zimbabwe',
    }

    @classmethod
    def short2full(cls, code):
        """Convert an ISO 3166-2 country code to the corresponding full name"""
        return cls._country_map.get(code.upper())


class GeoUtils(object):
    # Major IPv4 address blocks per country
    _country_ip_map = {
        'AD': '46.172.224.0/19',
        'AE': '94.200.0.0/13',
        'AF': '149.54.0.0/17',
        'AG': '209.59.64.0/18',
        'AI': '204.14.248.0/21',
        'AL': '46.99.0.0/16',
        'AM': '46.70.0.0/15',
        'AO': '105.168.0.0/13',
        'AP': '182.50.184.0/21',
        'AQ': '23.154.160.0/24',
        'AR': '181.0.0.0/12',
        'AS': '202.70.112.0/20',
        'AT': '77.116.0.0/14',
        'AU': '1.128.0.0/11',
        'AW': '181.41.0.0/18',
        'AX': '185.217.4.0/22',
        'AZ': '5.197.0.0/16',
        'BA': '31.176.128.0/17',
        'BB': '65.48.128.0/17',
        'BD': '114.130.0.0/16',
        'BE': '57.0.0.0/8',
        'BF': '102.178.0.0/15',
        'BG': '95.42.0.0/15',
        'BH': '37.131.0.0/17',
        'BI': '154.117.192.0/18',
        'BJ': '137.255.0.0/16',
        'BL': '185.212.72.0/23',
        'BM': '196.12.64.0/18',
        'BN': '156.31.0.0/16',
        'BO': '161.56.0.0/16',
        'BQ': '161.0.80.0/20',
        'BR': '191.128.0.0/12',
        'BS': '24.51.64.0/18',
        'BT': '119.2.96.0/19',
        'BW': '168.167.0.0/16',
        'BY': '178.120.0.0/13',
        'BZ': '179.42.192.0/18',
        'CA': '99.224.0.0/11',
        'CD': '41.243.0.0/16',
        'CF': '197.242.176.0/21',
        'CG': '160.113.0.0/16',
        'CH': '85.0.0.0/13',
        'CI': '102.136.0.0/14',
        'CK': '202.65.32.0/19',
        'CL': '152.172.0.0/14',
        'CM': '102.244.0.0/14',
        'CN': '36.128.0.0/10',
        'CO': '181.240.0.0/12',
        'CR': '201.192.0.0/12',
        'CU': '152.206.0.0/15',
        'CV': '165.90.96.0/19',
        'CW': '190.88.128.0/17',
        'CY': '31.153.0.0/16',
        'CZ': '88.100.0.0/14',
        'DE': '53.0.0.0/8',
        'DJ': '197.241.0.0/17',
        'DK': '87.48.0.0/12',
        'DM': '192.243.48.0/20',
        'DO': '152.166.0.0/15',
        'DZ': '41.96.0.0/12',
        'EC': '186.68.0.0/15',
        'EE': '90.190.0.0/15',
        'EG': '156.160.0.0/11',
        'ER': '196.200.96.0/20',
        'ES': '88.0.0.0/11',
        'ET': '196.188.0.0/14',
        'EU': '2.16.0.0/13',
        'FI': '91.152.0.0/13',
        'FJ': '144.120.0.0/16',
        'FK': '80.73.208.0/21',
        'FM': '119.252.112.0/20',
        'FO': '88.85.32.0/19',
        'FR': '90.0.0.0/9',
        'GA': '41.158.0.0/15',
        'GB': '25.0.0.0/8',
        'GD': '74.122.88.0/21',
        'GE': '31.146.0.0/16',
        'GF': '161.22.64.0/18',
        'GG': '62.68.160.0/19',
        'GH': '154.160.0.0/12',
        'GI': '95.164.0.0/16',
        'GL': '88.83.0.0/19',
        'GM': '160.182.0.0/15',
        'GN': '197.149.192.0/18',
        'GP': '104.250.0.0/19',
        'GQ': '105.235.224.0/20',
        'GR': '94.64.0.0/13',
        'GT': '168.234.0.0/16',
        'GU': '168.123.0.0/16',
        'GW': '197.214.80.0/20',
        'GY': '181.41.64.0/18',
        'HK': '113.252.0.0/14',
        'HN': '181.210.0.0/16',
        'HR': '93.136.0.0/13',
        'HT': '148.102.128.0/17',
        'HU': '84.0.0.0/14',
        'ID': '39.192.0.0/10',
        'IE': '87.32.0.0/12',
        'IL': '79.176.0.0/13',
        'IM': '5.62.80.0/20',
        'IN': '117.192.0.0/10',
        'IO': '203.83.48.0/21',
        'IQ': '37.236.0.0/14',
        'IR': '2.176.0.0/12',
        'IS': '82.221.0.0/16',
        'IT': '79.0.0.0/10',
        'JE': '87.244.64.0/18',
        'JM': '72.27.0.0/17',
        'JO': '176.29.0.0/16',
        'JP': '133.0.0.0/8',
        'KE': '105.48.0.0/12',
        'KG': '158.181.128.0/17',
        'KH': '36.37.128.0/17',
        'KI': '103.25.140.0/22',
        'KM': '197.255.224.0/20',
        'KN': '198.167.192.0/19',
        'KP': '175.45.176.0/22',
        'KR': '175.192.0.0/10',
        'KW': '37.36.0.0/14',
        'KY': '64.96.0.0/15',
        'KZ': '2.72.0.0/13',
        'LA': '115.84.64.0/18',
        'LB': '178.135.0.0/16',
        'LC': '24.92.144.0/20',
        'LI': '82.117.0.0/19',
        'LK': '112.134.0.0/15',
        'LR': '102.183.0.0/16',
        'LS': '129.232.0.0/17',
        'LT': '78.56.0.0/13',
        'LU': '188.42.0.0/16',
        'LV': '46.109.0.0/16',
        'LY': '41.252.0.0/14',
        'MA': '105.128.0.0/11',
        'MC': '88.209.64.0/18',
        'MD': '37.246.0.0/16',
        'ME': '178.175.0.0/17',
        'MF': '74.112.232.0/21',
        'MG': '154.126.0.0/17',
        'MH': '117.103.88.0/21',
        'MK': '77.28.0.0/15',
        'ML': '154.118.128.0/18',
        'MM': '37.111.0.0/17',
        'MN': '49.0.128.0/17',
        'MO': '60.246.0.0/16',
        'MP': '202.88.64.0/20',
        'MQ': '109.203.224.0/19',
        'MR': '41.188.64.0/18',
        'MS': '208.90.112.0/22',
        'MT': '46.11.0.0/16',
        'MU': '105.16.0.0/12',
        'MV': '27.114.128.0/18',
        'MW': '102.70.0.0/15',
        'MX': '187.192.0.0/11',
        'MY': '175.136.0.0/13',
        'MZ': '197.218.0.0/15',
        'NA': '41.182.0.0/16',
        'NC': '101.101.0.0/18',
        'NE': '197.214.0.0/18',
        'NF': '203.17.240.0/22',
        'NG': '105.112.0.0/12',
        'NI': '186.76.0.0/15',
        'NL': '145.96.0.0/11',
        'NO': '84.208.0.0/13',
        'NP': '36.252.0.0/15',
        'NR': '203.98.224.0/19',
        'NU': '49.156.48.0/22',
        'NZ': '49.224.0.0/14',
        'OM': '5.36.0.0/15',
        'PA': '186.72.0.0/15',
        'PE': '186.160.0.0/14',
        'PF': '123.50.64.0/18',
        'PG': '124.240.192.0/19',
        'PH': '49.144.0.0/13',
        'PK': '39.32.0.0/11',
        'PL': '83.0.0.0/11',
        'PM': '70.36.0.0/20',
        'PR': '66.50.0.0/16',
        'PS': '188.161.0.0/16',
        'PT': '85.240.0.0/13',
        'PW': '202.124.224.0/20',
        'PY': '181.120.0.0/14',
        'QA': '37.210.0.0/15',
        'RE': '102.35.0.0/16',
        'RO': '79.112.0.0/13',
        'RS': '93.86.0.0/15',
        'RU': '5.136.0.0/13',
        'RW': '41.186.0.0/16',
        'SA': '188.48.0.0/13',
        'SB': '202.1.160.0/19',
        'SC': '154.192.0.0/11',
        'SD': '102.120.0.0/13',
        'SE': '78.64.0.0/12',
        'SG': '8.128.0.0/10',
        'SI': '188.196.0.0/14',
        'SK': '78.98.0.0/15',
        'SL': '102.143.0.0/17',
        'SM': '89.186.32.0/19',
        'SN': '41.82.0.0/15',
        'SO': '154.115.192.0/18',
        'SR': '186.179.128.0/17',
        'SS': '105.235.208.0/21',
        'ST': '197.159.160.0/19',
        'SV': '168.243.0.0/16',
        'SX': '190.102.0.0/20',
        'SY': '5.0.0.0/16',
        'SZ': '41.84.224.0/19',
        'TC': '65.255.48.0/20',
        'TD': '154.68.128.0/19',
        'TG': '196.168.0.0/14',
        'TH': '171.96.0.0/13',
        'TJ': '85.9.128.0/18',
        'TK': '27.96.24.0/21',
        'TL': '180.189.160.0/20',
        'TM': '95.85.96.0/19',
        'TN': '197.0.0.0/11',
        'TO': '175.176.144.0/21',
        'TR': '78.160.0.0/11',
        'TT': '186.44.0.0/15',
        'TV': '202.2.96.0/19',
        'TW': '120.96.0.0/11',
        'TZ': '156.156.0.0/14',
        'UA': '37.52.0.0/14',
        'UG': '102.80.0.0/13',
        'US': '6.0.0.0/8',
        'UY': '167.56.0.0/13',
        'UZ': '84.54.64.0/18',
        'VA': '212.77.0.0/19',
        'VC': '207.191.240.0/21',
        'VE': '186.88.0.0/13',
        'VG': '66.81.192.0/20',
        'VI': '146.226.0.0/16',
        'VN': '14.160.0.0/11',
        'VU': '202.80.32.0/20',
        'WF': '117.20.32.0/21',
        'WS': '202.4.32.0/19',
        'YE': '134.35.0.0/16',
        'YT': '41.242.116.0/22',
        'ZA': '41.0.0.0/11',
        'ZM': '102.144.0.0/13',
        'ZW': '102.177.192.0/18',
    }

    @classmethod
    def random_ipv4(cls, code_or_block):
        if len(code_or_block) == 2:
            block = cls._country_ip_map.get(code_or_block.upper())
            if not block:
                return None
        else:
            block = code_or_block
        addr, preflen = block.split('/')
        addr_min = compat_struct_unpack('!L', socket.inet_aton(addr))[0]
        addr_max = addr_min | (0xffffffff >> int(preflen))
        return compat_str(socket.inet_ntoa(
            compat_struct_pack('!L', random.randint(addr_min, addr_max))))


# Both long_to_bytes and bytes_to_long are adapted from PyCrypto, which is
# released into Public Domain
# https://github.com/dlitz/pycrypto/blob/master/lib/Crypto/Util/number.py#L387

def long_to_bytes(n, blocksize=0):
    """long_to_bytes(n:long, blocksize:int) : string
    Convert a long integer to a byte string.

    If optional blocksize is given and greater than zero, pad the front of the
    byte string with binary zeros so that the length is a multiple of
    blocksize.
    """
    # after much testing, this algorithm was deemed to be the fastest
    s = b''
    n = int(n)
    while n > 0:
        s = compat_struct_pack('>I', n & 0xffffffff) + s
        n = n >> 32
    # strip off leading zeros
    for i in range(len(s)):
        if s[i] != b'\000'[0]:
            break
    else:
        # only happens when n == 0
        s = b'\000'
        i = 0
    s = s[i:]
    # add back some pad bytes.  this could be done more efficiently w.r.t. the
    # de-padding being done above, but sigh...
    if blocksize > 0 and len(s) % blocksize:
        s = (blocksize - len(s) % blocksize) * b'\000' + s
    return s


def bytes_to_long(s):
    """bytes_to_long(string) : long
    Convert a byte string to a long integer.

    This is (essentially) the inverse of long_to_bytes().
    """
    acc = 0
    length = len(s)
    if length % 4:
        extra = (4 - length % 4)
        s = b'\000' * extra + s
        length = length + extra
    for i in range(0, length, 4):
        acc = (acc << 32) + compat_struct_unpack('>I', s[i:i + 4])[0]
    return acc


def ohdave_rsa_encrypt(data, exponent, modulus):
    '''
    Implement OHDave's RSA algorithm. See http://www.ohdave.com/rsa/

    Input:
        data: data to encrypt, bytes-like object
        exponent, modulus: parameter e and N of RSA algorithm, both integer
    Output: hex string of encrypted data

    Limitation: supports one block encryption only
    '''

    payload = int(binascii.hexlify(data[::-1]), 16)
    encrypted = pow(payload, exponent, modulus)
    return '%x' % encrypted


def pkcs1pad(data, length):
    """
    Padding input data with PKCS#1 scheme

    @param {int[]} data        input data
    @param {int}   length      target length
    @returns {int[]}           padded data
    """
    if len(data) > length - 11:
        raise ValueError('Input data too long for PKCS#1 padding')

    pseudo_random = [random.randint(0, 254) for _ in range(length - len(data) - 3)]
    return [0, 2] + pseudo_random + [0] + data


def encode_base_n(num, n, table=None):
    FULL_TABLE = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    if not table:
        table = FULL_TABLE[:n]

    if n > len(table):
        raise ValueError('base %d exceeds table length %d' % (n, len(table)))

    if num == 0:
        return table[0]

    ret = ''
    while num:
        ret = table[num % n] + ret
        num = num // n
    return ret


def decode_packed_codes(code):
    mobj = re.search(PACKED_CODES_RE, code)
    obfuscated_code, base, count, symbols = mobj.groups()
    base = int(base)
    count = int(count)
    symbols = symbols.split('|')
    symbol_table = {}

    while count:
        count -= 1
        base_n_count = encode_base_n(count, base)
        symbol_table[base_n_count] = symbols[count] or base_n_count

    return re.sub(
        r'\b(\w+)\b', lambda mobj: symbol_table[mobj.group(0)],
        obfuscated_code)


def caesar(s, alphabet, shift):
    if shift == 0:
        return s
    l = len(alphabet)
    return ''.join(
        alphabet[(alphabet.index(c) + shift) % l] if c in alphabet else c
        for c in s)


def rot47(s):
    return caesar(s, r'''!"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}~''', 47)


def parse_m3u8_attributes(attrib):
    info = {}
    for (key, val) in re.findall(r'(?P<key>[A-Z0-9-]+)=(?P<val>"[^"]+"|[^",]+)(?:,|$)', attrib):
        if val.startswith('"'):
            val = val[1:-1]
        info[key] = val
    return info


def urshift(val, n):
    return val >> n if val >= 0 else (val + 0x100000000) >> n


def write_xattr(path, key, value):
    from .exceptions import XAttrMetadataError, XAttrUnavailableError
    # This mess below finds the best xattr tool for the job
    try:
        # try the pyxattr module...
        import xattr

        if hasattr(xattr, 'set'):  # pyxattr
            # Unicode arguments are not supported in python-pyxattr until
            # version 0.5.0
            # See https://github.com/ytdl-org/youtube-dl/issues/5498
            pyxattr_required_version = '0.5.0'
            if version_tuple(xattr.__version__) < version_tuple(pyxattr_required_version):
                # TODO: fallback to CLI tools
                raise XAttrUnavailableError(
                    'python-pyxattr is detected but is too old. '
                    'yt-dlp requires %s or above while your version is %s. '
                    'Falling back to other xattr implementations' % (
                        pyxattr_required_version, xattr.__version__))

            setxattr = xattr.set
        else:  # xattr
            setxattr = xattr.setxattr

        try:
            setxattr(path, key, value)
        except EnvironmentError as e:
            raise XAttrMetadataError(e.errno, e.strerror)

    except ImportError:
        if compat_os_name == 'nt':
            # Write xattrs to NTFS Alternate Data Streams:
            # http://en.wikipedia.org/wiki/NTFS#Alternate_data_streams_.28ADS.29
            assert ':' not in key
            assert os.path.exists(path)

            ads_fn = path + ':' + key
            try:
                with open(ads_fn, 'wb') as f:
                    f.write(value)
            except EnvironmentError as e:
                raise XAttrMetadataError(e.errno, e.strerror)
        else:
            user_has_setfattr = check_executable('setfattr', ['--version'])
            user_has_xattr = check_executable('xattr', ['-h'])

            if user_has_setfattr or user_has_xattr:

                value = value.decode('utf-8')
                if user_has_setfattr:
                    executable = 'setfattr'
                    opts = ['-n', key, '-v', value]
                elif user_has_xattr:
                    executable = 'xattr'
                    opts = ['-w', key, value]

                cmd = ([encodeFilename(executable, True)]
                       + [encodeArgument(o) for o in opts]
                       + [encodeFilename(path, True)])

                try:
                    p = Popen(
                        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
                except EnvironmentError as e:
                    raise XAttrMetadataError(e.errno, e.strerror)
                stdout, stderr = p.communicate_or_kill()
                stderr = stderr.decode('utf-8', 'replace')
                if p.returncode != 0:
                    raise XAttrMetadataError(p.returncode, stderr)

            else:
                # On Unix, and can't find pyxattr, setfattr, or xattr.
                if sys.platform.startswith('linux'):
                    raise XAttrUnavailableError(
                        "Couldn't find a tool to set the xattrs. "
                        "Install either the python 'pyxattr' or 'xattr' "
                        "modules, or the GNU 'attr' package "
                        "(which contains the 'setfattr' tool).")
                else:
                    raise XAttrUnavailableError(
                        "Couldn't find a tool to set the xattrs. "
                        "Install either the python 'xattr' module, "
                        "or the 'xattr' binary.")


def random_birthday(year_field, month_field, day_field):
    start_date = datetime.date(1950, 1, 1)
    end_date = datetime.date(1995, 12, 31)
    offset = random.randint(0, (end_date - start_date).days)
    random_date = start_date + datetime.timedelta(offset)
    return {
        year_field: str(random_date.year),
        month_field: str(random_date.month),
        day_field: str(random_date.day),
    }


# Templates for internet shortcut files, which are plain text files.
DOT_URL_LINK_TEMPLATE = '''
[InternetShortcut]
URL=%(url)s
'''.lstrip()

DOT_WEBLOC_LINK_TEMPLATE = '''
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
\t<key>URL</key>
\t<string>%(url)s</string>
</dict>
</plist>
'''.lstrip()

DOT_DESKTOP_LINK_TEMPLATE = '''
[Desktop Entry]
Encoding=UTF-8
Name=%(filename)s
Type=Link
URL=%(url)s
Icon=text-html
'''.lstrip()

LINK_TEMPLATES = {
    'url': DOT_URL_LINK_TEMPLATE,
    'desktop': DOT_DESKTOP_LINK_TEMPLATE,
    'webloc': DOT_WEBLOC_LINK_TEMPLATE,
}


def iri_to_uri(iri):
    """
    Converts an IRI (Internationalized Resource Identifier, allowing Unicode characters) to a URI (Uniform Resource Identifier, ASCII-only).

    The function doesn't add an additional layer of escaping; e.g., it doesn't escape `%3C` as `%253C`. Instead, it percent-escapes characters with an underlying UTF-8 encoding *besides* those already escaped, leaving the URI intact.
    """

    iri_parts = compat_urllib_parse_urlparse(iri)

    if '[' in iri_parts.netloc:
        raise ValueError('IPv6 URIs are not, yet, supported.')
        # Querying `.netloc`, when there's only one bracket, also raises a ValueError.

    # The `safe` argument values, that the following code uses, contain the characters that should not be percent-encoded. Everything else but letters, digits and '_.-' will be percent-encoded with an underlying UTF-8 encoding. Everything already percent-encoded will be left as is.

    net_location = ''
    if iri_parts.username:
        net_location += compat_urllib_parse_quote(iri_parts.username, safe=r"!$%&'()*+,~")
        if iri_parts.password is not None:
            net_location += ':' + compat_urllib_parse_quote(iri_parts.password, safe=r"!$%&'()*+,~")
        net_location += '@'

    net_location += iri_parts.hostname.encode('idna').decode('utf-8')  # Punycode for Unicode hostnames.
    # The 'idna' encoding produces ASCII text.
    if iri_parts.port is not None and iri_parts.port != 80:
        net_location += ':' + str(iri_parts.port)

    return compat_urllib_parse_urlunparse(
        (iri_parts.scheme,
            net_location,

            compat_urllib_parse_quote_plus(iri_parts.path, safe=r"!$%&'()*+,/:;=@|~"),

            # Unsure about the `safe` argument, since this is a legacy way of handling parameters.
            compat_urllib_parse_quote_plus(iri_parts.params, safe=r"!$%&'()*+,/:;=@|~"),

            # Not totally sure about the `safe` argument, since the source does not explicitly mention the query URI component.
            compat_urllib_parse_quote_plus(iri_parts.query, safe=r"!$%&'()*+,/:;=?@{|}~"),

            compat_urllib_parse_quote_plus(iri_parts.fragment, safe=r"!#$%&'()*+,/:;=?@{|}~")))

    # Source for `safe` arguments: https://url.spec.whatwg.org/#percent-encoded-bytes.


def to_high_limit_path(path):
    if sys.platform in ['win32', 'cygwin']:
        # Work around MAX_PATH limitation on Windows. The maximum allowed length for the individual path segments may still be quite limited.
        return '\\\\?\\' + os.path.abspath(path)

    return path


def format_field(obj, field=None, template='%s', ignore=(None, ''), default='', func=None):
    val = traverse_obj(obj, *variadic(field))
    if val in ignore:
        return default
    return template % (func(val) if func else val)


def clean_podcast_url(url):
    return re.sub(r'''(?x)
        (?:
            (?:
                chtbl\.com/track|
                media\.blubrry\.com| # https://create.blubrry.com/resources/podcast-media-download-statistics/getting-started/
                play\.podtrac\.com
            )/[^/]+|
            (?:dts|www)\.podtrac\.com/(?:pts/)?redirect\.[0-9a-z]{3,4}| # http://analytics.podtrac.com/how-to-measure
            flex\.acast\.com|
            pd(?:
                cn\.co| # https://podcorn.com/analytics-prefix/
                st\.fm # https://podsights.com/docs/
            )/e
        )/''', '', url)


def make_dir(path, to_screen=None):
    try:
        dn = os.path.dirname(path)
        if dn and not os.path.exists(dn):
            os.makedirs(dn)
        return True
    except (OSError, IOError) as err:
        if callable(to_screen) is not None:
            to_screen('unable to create directory ' + error_to_compat_str(err))
        return False


def get_executable_path():
    from zipimport import zipimporter
    if hasattr(sys, 'frozen'):  # Running from PyInstaller
        path = os.path.dirname(sys.executable)
    elif isinstance(globals().get('__loader__'), zipimporter):  # Running from ZIP
        path = os.path.join(os.path.dirname(__file__), '../..')
    else:
        path = os.path.join(os.path.dirname(__file__), '..')
    return os.path.abspath(path)


def load_plugins(name, suffix, namespace):
    classes = {}
    try:
        plugins_spec = importlib.util.spec_from_file_location(
            name, os.path.join(get_executable_path(), 'ytdlp_plugins', name, '__init__.py'))
        plugins = importlib.util.module_from_spec(plugins_spec)
        sys.modules[plugins_spec.name] = plugins
        plugins_spec.loader.exec_module(plugins)
        for name in dir(plugins):
            if name in namespace:
                continue
            if not name.endswith(suffix):
                continue
            klass = getattr(plugins, name)
            classes[name] = namespace[name] = klass
    except FileNotFoundError:
        pass
    return classes


def traverse_obj(
        obj, *path_list, default=None, expected_type=None, get_all=True,
        casesense=True, is_user_input=False, traverse_string=False):
    ''' Traverse nested list/dict/tuple
    @param path_list        A list of paths which are checked one by one.
                            Each path is a list of keys where each key is a string,
                            a function, a tuple of strings/None or "...".
                            When a fuction is given, it takes the key as argument and
                            returns whether the key matches or not. When a tuple is given,
                            all the keys given in the tuple are traversed, and
                            "..." traverses all the keys in the object
                            "None" returns the object without traversal
    @param default          Default value to return
    @param expected_type    Only accept final value of this type (Can also be any callable)
    @param get_all          Return all the values obtained from a path or only the first one
    @param casesense        Whether to consider dictionary keys as case sensitive
    @param is_user_input    Whether the keys are generated from user input. If True,
                            strings are converted to int/slice if necessary
    @param traverse_string  Whether to traverse inside strings. If True, any
                            non-compatible object will also be converted into a string
    # TODO: Write tests
    '''
    if not casesense:
        _lower = lambda k: (k.lower() if isinstance(k, str) else k)
        path_list = (map(_lower, variadic(path)) for path in path_list)

    def _traverse_obj(obj, path, _current_depth=0):
        nonlocal depth
        path = tuple(variadic(path))
        for i, key in enumerate(path):
            if None in (key, obj):
                return obj
            if isinstance(key, (list, tuple)):
                obj = [_traverse_obj(obj, sub_key, _current_depth) for sub_key in key]
                key = ...
            if key is ...:
                obj = (obj.values() if isinstance(obj, dict)
                       else obj if isinstance(obj, (list, tuple, LazyList))
                       else str(obj) if traverse_string else [])
                _current_depth += 1
                depth = max(depth, _current_depth)
                return [_traverse_obj(inner_obj, path[i + 1:], _current_depth) for inner_obj in obj]
            elif callable(key):
                if isinstance(obj, (list, tuple, LazyList)):
                    obj = enumerate(obj)
                elif isinstance(obj, dict):
                    obj = obj.items()
                else:
                    if not traverse_string:
                        return None
                    obj = str(obj)
                _current_depth += 1
                depth = max(depth, _current_depth)
                return [_traverse_obj(v, path[i + 1:], _current_depth) for k, v in obj if key(k)]
            elif isinstance(obj, dict) and not (is_user_input and key == ':'):
                obj = (obj.get(key) if casesense or (key in obj)
                       else next((v for k, v in obj.items() if _lower(k) == key), None))
            else:
                if is_user_input:
                    key = (int_or_none(key) if ':' not in key
                           else slice(*map(int_or_none, key.split(':'))))
                    if key == slice(None):
                        return _traverse_obj(obj, (..., *path[i + 1:]), _current_depth)
                if not isinstance(key, (int, slice)):
                    return None
                if not isinstance(obj, (list, tuple, LazyList)):
                    if not traverse_string:
                        return None
                    obj = str(obj)
                try:
                    obj = obj[key]
                except IndexError:
                    return None
        return obj

    if isinstance(expected_type, type):
        type_test = lambda val: val if isinstance(val, expected_type) else None
    elif expected_type is not None:
        type_test = expected_type
    else:
        type_test = lambda val: val

    for path in path_list:
        depth = 0
        val = _traverse_obj(obj, path)
        if val is not None:
            if depth:
                for _ in range(depth - 1):
                    val = itertools.chain.from_iterable(v for v in val if v is not None)
                val = [v for v in map(type_test, val) if v is not None]
                if val:
                    return val if get_all else val[0]
            else:
                val = type_test(val)
                if val is not None:
                    return val
    return default


def traverse_dict(dictn, keys, casesense=True):
    import warnings
    warnings.warn(DeprecationWarning(
        'yt_dlp.utils.traverse_dict is deprecated '
        'and may be removed in a future version. Use yt_dlp.utils.traverse_obj instead'
    ))
    return traverse_obj(dictn, keys, casesense=casesense, is_user_input=True, traverse_string=True)


def variadic(x, allowed_types=(str, bytes, dict)):
    return x if isinstance(x, collections.abc.Iterable) and not isinstance(x, allowed_types) else (x,)


# create a JSON Web Signature (jws) with HS256 algorithm
# the resulting format is in JWS Compact Serialization
# implemented following JWT https://www.rfc-editor.org/rfc/rfc7519.html
# implemented following JWS https://www.rfc-editor.org/rfc/rfc7515.html
def jwt_encode_hs256(payload_data, key, headers={}):
    header_data = {
        'alg': 'HS256',
        'typ': 'JWT',
    }
    if headers:
        header_data.update(headers)
    header_b64 = base64.b64encode(json.dumps(header_data).encode('utf-8'))
    payload_b64 = base64.b64encode(json.dumps(payload_data).encode('utf-8'))
    h = hmac.new(key.encode('utf-8'), header_b64 + b'.' + payload_b64, hashlib.sha256)
    signature_b64 = base64.b64encode(h.digest())
    token = header_b64 + b'.' + payload_b64 + b'.' + signature_b64
    return token


# can be extended in future to verify the signature and parse header and return the algorithm used if it's not HS256
def jwt_decode_hs256(jwt):
    header_b64, payload_b64, signature_b64 = jwt.split('.')
    payload_data = json.loads(base64.urlsafe_b64decode(payload_b64))
    return payload_data


def supports_terminal_sequences(stream):
    if compat_os_name == 'nt':
        from .compat import WINDOWS_VT_MODE  # Must be imported locally
        if not WINDOWS_VT_MODE or get_windows_version() < (10, 0, 10586):
            return False
    elif not os.getenv('TERM'):
        return False
    try:
        return stream.isatty()
    except BaseException:
        return False


_terminal_sequences_re = re.compile('\033\\[[^m]+m')


def remove_terminal_sequences(string):
    return _terminal_sequences_re.sub('', string)


def number_of_digits(number):
    return len('%d' % number)


def join_nonempty(*values, delim='-', from_dict=None):
    if from_dict is not None:
        values = map(from_dict.get, values)
    return delim.join(map(str, filter(None, values)))


class Config:
    own_args = None
    filename = None
    __initialized = False

    def __init__(self, parser, label=None):
        self._parser, self.label = parser, label
        self._loaded_paths, self.configs = set(), []

    def init(self, args=None, filename=None):
        assert not self.__initialized
        directory = ''
        if filename:
            location = os.path.realpath(filename)
            directory = os.path.dirname(location)
            if location in self._loaded_paths:
                return False
            self._loaded_paths.add(location)

        self.__initialized = True
        self.own_args, self.filename = args, filename
        for location in self._parser.parse_args(args)[0].config_locations or []:
            location = os.path.join(directory, expand_path(location))
            if os.path.isdir(location):
                location = os.path.join(location, 'yt-dlp.conf')
            if not os.path.exists(location):
                self._parser.error(f'config location {location} does not exist')
            self.append_config(self.read_file(location), location)
        return True

    def __str__(self):
        label = join_nonempty(
            self.label, 'config', f'"{self.filename}"' if self.filename else '',
            delim=' ')
        return join_nonempty(
            self.own_args is not None and f'{label[0].upper()}{label[1:]}: {self.hide_login_info(self.own_args)}',
            *(f'\n{c}'.replace('\n', '\n| ')[1:] for c in self.configs),
            delim='\n')

    @staticmethod
    def read_file(filename, default=[]):
        try:
            optionf = open(filename)
        except IOError:
            return default  # silently skip if file is not present
        try:
            # FIXME: https://github.com/ytdl-org/youtube-dl/commit/dfe5fa49aed02cf36ba9f743b11b0903554b5e56
            contents = optionf.read()
            res = compat_shlex_split(contents, comments=True)
        finally:
            optionf.close()
        return res

    @staticmethod
    def hide_login_info(opts):
        PRIVATE_OPTS = set(['-p', '--password', '-u', '--username', '--video-password', '--ap-password', '--ap-username'])
        eqre = re.compile('^(?P<key>' + ('|'.join(re.escape(po) for po in PRIVATE_OPTS)) + ')=.+$')

        def _scrub_eq(o):
            m = eqre.match(o)
            if m:
                return m.group('key') + '=PRIVATE'
            else:
                return o

        opts = list(map(_scrub_eq, opts))
        for idx, opt in enumerate(opts):
            if opt in PRIVATE_OPTS and idx + 1 < len(opts):
                opts[idx + 1] = 'PRIVATE'
        return opts

    def append_config(self, *args, label=None):
        config = type(self)(self._parser, label)
        config._loaded_paths = self._loaded_paths
        if config.init(*args):
            self.configs.append(config)

    @property
    def all_args(self):
        for config in reversed(self.configs):
            yield from config.all_args
        yield from self.own_args or []

    def parse_args(self):
        return self._parser.parse_args(list(self.all_args))


class YDLLogger:
    def __init__(self, ydl=None):
        self._ydl = ydl

    def debug(self, message):
        if self._ydl:
            self._ydl.write_debug(message)

    def info(self, message):
        if self._ydl:
            self._ydl.to_screen(f'[Cookies] {message}')

    def warning(self, message, only_once=False):
        if self._ydl:
            self._ydl.report_warning(message, only_once)

    def error(self, message):
        if self._ydl:
            self._ydl.report_error(message)