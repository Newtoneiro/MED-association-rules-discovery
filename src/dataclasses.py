
from dataclasses import dataclass
from functools import total_ordering


@total_ordering
@dataclass
class Item:
    item: frozenset
    support: float

    def __str__(self):
        return f"Item: {str(self.item):<40} | Support: {self.support:.3f}"

    def __eq__(self, other: "Item") -> bool:
        return self.support == other.support

    def __lt__(self, other: "Item") -> bool:
        return self.support < other.support


@total_ordering
@dataclass
class Rule:
    pre: frozenset
    post: frozenset
    confidence: float

    def __str__(self):
        rule = f"{str(self.pre)} ==> {str(self.post)}"
        return f"{rule:<40} | Confidence: {self.confidence:.3f}"

    def __eq__(self, other: "Rule") -> bool:
        return self.confidence == other.confidence

    def __lt__(self, other: "Rule") -> bool:
        return self.confidence < other.confidence
