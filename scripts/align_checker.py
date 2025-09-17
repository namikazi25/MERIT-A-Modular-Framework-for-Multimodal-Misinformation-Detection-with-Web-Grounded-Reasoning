from __future__ import annotations

"""
Deprecated: use scripts.relevancy_checker instead.

This shim re-exports assess_image_headline_alignment from the relevancy checker
for backward compatibility.
"""

from scripts.relevancy_checker import (
    assess_image_headline_relevancy as assess_image_headline_alignment,
)

__all__ = ["assess_image_headline_alignment"]
