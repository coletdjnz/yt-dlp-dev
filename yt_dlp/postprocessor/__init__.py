# flake8: noqa: F401

from .common import PostProcessor
from .embedthumbnail import EmbedThumbnailPP
from .exec import ExecAfterDownloadPP, ExecPP
from .ffmpeg import (
    FFmpegConcatPP,
    FFmpegCopyStreamPP,
    FFmpegEmbedSubtitlePP,
    FFmpegExtractAudioPP,
    FFmpegFixupDuplicateMoovPP,
    FFmpegFixupDurationPP,
    FFmpegFixupM3u8PP,
    FFmpegFixupM4aPP,
    FFmpegFixupStretchedPP,
    FFmpegFixupTimestampPP,
    FFmpegMergerPP,
    FFmpegMetadataPP,
    FFmpegPostProcessor,
    FFmpegSplitChaptersPP,
    FFmpegSubtitlesConvertorPP,
    FFmpegThumbnailsConvertorPP,
    FFmpegVideoConvertorPP,
    FFmpegVideoRemuxerPP,
)
from .metadataparser import (
    MetadataFromFieldPP,
    MetadataFromTitlePP,
    MetadataParserPP,
)
from .modify_chapters import ModifyChaptersPP
from .movefilesafterdownload import MoveFilesAfterDownloadPP
from .sponskrub import SponSkrubPP
from .sponsorblock import SponsorBlockPP
from .xattrpp import XAttrMetadataPP
from ..globals import plugin_pps, postprocessors
from ..plugins import PACKAGE_NAME
from ..utils import deprecation_warning


def __getattr__(name):
    lookup = plugin_pps.get()
    if name in lookup:
        deprecation_warning(
            f'Importing a plugin Post-Processor from {__name__} is deprecated. '
            f'Please import {PACKAGE_NAME}.postprocessor.{name} instead.')
        return lookup[name]

    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def get_postprocessor(key):
    return postprocessors.get()[key + 'PP']


_default_pps = {
    name: value
    for name, value in globals().items()
    if name.endswith('PP') or name in ('PostProcessor', 'FFmpegPostProcessor')
}
postprocessors.set(_default_pps)

__all__ = list(_default_pps.values())
