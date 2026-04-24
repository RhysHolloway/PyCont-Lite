import sys, threading
from collections.abc import Callable
from enum import IntEnum

from typing import TextIO, Union

class Verbosity(IntEnum):
    OFF = 0
    INFO = 1
    VERBOSE = 2

Message = Union[str, Callable[[], str]]

class Logger:
    """Barebones, always-prints-to-screen (or given stream). Thread-safe."""
    def __init__(self, verbosity: Verbosity = Verbosity.INFO, stream: TextIO = sys.stdout):
        self._verbosity = verbosity
        self._stream = stream
        self._lock = threading.Lock()

    # config
    def set(self, verbosity: Verbosity) -> None:
        self._verbosity = verbosity

    def set_stream(self, stream: TextIO) -> None:
        self._stream = stream

    # queries
    def enabled(self, level: Verbosity) -> bool:
        return level <= self._verbosity

    # emit
    def log(self, msg: Message, level: Verbosity = Verbosity.INFO) -> None:
        if not self.enabled(level):
            return

        text = msg() if callable(msg) else msg
        rendered = str(text)
        with self._lock:
            self._stream.write(rendered)
            if not rendered.endswith("\n"):
                self._stream.write("\n")
            self._stream.flush()

    # convenience
    def info(self, msg: Message) -> None:
        self.log(msg, Verbosity.INFO)

    def verbose(self, msg: Message) -> None:
        self.log(msg, Verbosity.VERBOSE)

def coerce_verbosity(v) -> Verbosity:
    if isinstance(v, Verbosity):
        return v
    if isinstance(v, str):
        try:
            return Verbosity[v.upper()]  # "off"|"info"|"verbose"
        except KeyError:
            raise ValueError(f"Unknown verbosity '{v}'")
    try:
        return Verbosity(int(v))
    except Exception as e:
        raise ValueError(f"Bad verbosity {v!r}") from e
    
LOG = Logger()
def configureLOG(*, verbosity: Union[Verbosity, str, int] = Verbosity.INFO, stream: TextIO | None = None) -> None:
    LOG.set(coerce_verbosity(verbosity))
    if stream is not None:
        LOG.set_stream(stream)
