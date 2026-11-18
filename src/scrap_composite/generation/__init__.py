"""Produce and verify composites from pinned scene and asset inputs."""
import os

from scrap_composite.runtime import enable_legacy_tools

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
enable_legacy_tools()

VERSION = "display-batch-v2"
