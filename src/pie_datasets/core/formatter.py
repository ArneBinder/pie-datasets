from typing import Any, Dict, List

import pyarrow as pa
from datasets.formatting.formatting import Formatter
from pie_core import Document, TaskEncoding


class DocumentFormatter(Formatter[Document, list, List[Document]]):
    def __init__(self, document_type, features=None, **kwargs):
        super().__init__(features=None)
        self.document_type = document_type

    def format_row(self, pa_table: pa.Table) -> Document:
        row = self.python_arrow_extractor().extract_row(pa_table)
        return self.document_type.fromdict(row)

    def format_column(self, pa_table: pa.Table) -> list:
        return []

    def format_batch(self, pa_table: pa.Table) -> List[Document]:
        batch = self.simple_arrow_extractor().extract_batch(pa_table).to_pylist()
        return [self.document_type.fromdict(b) for b in batch]


class TaskEncodingFormatter(Formatter[TaskEncoding, List, List[TaskEncoding]]):
    def __init__(self, features=None, **kwargs):
        super().__init__(features=None)

    def construct_task_encoding(self, row: Dict[str, Any]) -> TaskEncoding:
        return TaskEncoding(
            document=Document(), inputs=row["inputs"], targets=row.get("targets", None)
        )

    def format_row(self, pa_table: pa.Table) -> TaskEncoding:
        row = self.python_arrow_extractor().extract_row(pa_table)
        return self.construct_task_encoding(row)

    def format_column(self, pa_table: pa.Table) -> list:
        return []

    def format_batch(self, pa_table: pa.Table) -> List[TaskEncoding]:
        batch = self.simple_arrow_extractor().extract_batch(pa_table).to_pylist()
        return [self.construct_task_encoding(b) for b in batch]
