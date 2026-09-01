"""
Local-testing-only shim: Python 3.10 lacks typing.NotRequired (added in
3.11, PEP 655), which litellm imports directly. This is not a bug in this
repo -- the app's actual required Python version is 3.11+ (see README) and
this shim exists purely so the test suite can also run on this dev
machine's 3.10 interpreter. Harmless if already 3.11+ (NotRequired already
exists there, so this is a no-op).
"""
import sys

if sys.version_info < (3, 11):
    import typing
    import typing_extensions
    if not hasattr(typing, "NotRequired"):
        typing.NotRequired = typing_extensions.NotRequired
    if not hasattr(typing, "Required"):
        typing.Required = typing_extensions.Required
