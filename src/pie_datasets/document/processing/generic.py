from __future__ import annotations

import logging
from typing import Callable, Generic, TypeVar

from pytorch_ie.core import Document
from pytorch_ie.utils.hydra import resolve_target

logger = logging.getLogger(__name__)


D = TypeVar("D", bound=Document)


class Pipeline:
    def __init__(
        self,
        *processors_list: Callable[[Document], Document],
        **processors_dict: Callable[[Document], Document],
    ):
        if processors_list and processors_dict:
            raise ValueError("You cannot use both positional and keyword arguments.")
        if processors_list:
            self.processors = {
                f"processor_{i}": processor for i, processor in enumerate(processors_list)
            }
        else:
            self.processors = processors_dict

    def __call__(self, doc: Document) -> Document:
        for _, processor in self.processors.items():
            doc = processor(doc)
        return doc


class Caster(Generic[D]):
    def __init__(self, document_type: str | type[D], **kwargs):
        dt = resolve_target(document_type, full_key="")
        if not (isinstance(dt, type) and issubclass(dt, Document)):
            raise TypeError(
                f"(resolved) document_type must be a subclass of Document, but it is: {dt}"
            )
        self.document_type = dt
        self.kwargs = kwargs

    def __call__(self, doc: Document) -> D:
        return doc.as_type(new_type=self.document_type, **self.kwargs)


class Converter(Generic[D]):
    """Generic converter class that takes a function that converts a document to another document
    and passes all other arguments to the function.

    If the function argument is not a callable, it is assumed that it is a string and the function
    is resolved via hydra.
    """

    def __init__(self, function: str | Callable[[Document], D], **kwargs):
        if callable(function):
            self.convert = function
        else:
            self.convert = resolve_target(function, full_key="")
        self.kwargs = kwargs

    def __call__(self, doc: Document) -> D:
        return self.convert(doc, **self.kwargs)
