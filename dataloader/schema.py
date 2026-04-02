"""
schema.py — Shared dataclass for all datasets.
"""
from dataclasses import dataclass, field


@dataclass
class EvalSample:
    id: str
    query: str
    # All documents in this sample's retrieval pool (each has its own _id)
    documents: list            # list of {"id": str, "text": str}
    relevant_doc_ids: list     # list of str — ground-truth relevant doc ids
    reference_answer: str      # reference text answer (empty string if N/A)
    answer_spans: list = field(default_factory=list)  # answer texts for answer-in-context
    choices: list = field(default_factory=list)   # for QuALITY (MCQ)
    correct_choice: int = -1                       # for QuALITY
