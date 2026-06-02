from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Dict, List, Optional


@dataclass
class ComplianceRuleSet:
    source_path: Optional[str]
    raw_text: str
    forbidden_patterns: List[str] = field(default_factory=list)
    required_phrases: List[str] = field(default_factory=list)
    forbidden_by_scope: Dict[str, List[str]] = field(default_factory=dict)
    required_by_scope: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def has_rules(self) -> bool:
        return bool(self.forbidden_patterns or self.required_phrases or self.raw_text.strip())

    @property
    def rule_count(self) -> int:
        scoped_forbidden = sum(len(v) for v in self.forbidden_by_scope.values())
        scoped_required = sum(len(v) for v in self.required_by_scope.values())
        return int(len(self.forbidden_patterns) + len(self.required_phrases) + scoped_forbidden + scoped_required)


def _normalize_scope(raw_scope: str) -> str:
    s = str(raw_scope or "").strip().lower()
    if not s:
        return "common"
    if s in {"공통", "전체", "all", "common"}:
        return "common"
    if s in {"예금", "적금", "수신", "예금성", "deposit"}:
        return "deposit"
    if s in {"펀드", "투자성", "fund"}:
        return "fund"
    return "common"


def _normalize_line(line: str) -> str:
    return str(line or "").strip()


def _parse_rule_lines(raw_text: str) -> ComplianceRuleSet:
    forbidden: List[str] = []
    required: List[str] = []
    forbidden_by_scope: Dict[str, List[str]] = {"common": [], "deposit": [], "fund": []}
    required_by_scope: Dict[str, List[str]] = {"common": [], "deposit": [], "fund": []}

    for raw_line in raw_text.splitlines():
        line = _normalize_line(raw_line)
        if not line:
            continue
        if line.startswith("#"):
            continue

        m = re.match(
            r"^(금지|금지표현|forbid|필수|required)(?:\(([^)]+)\))?\s*:\s*(.+)$",
            line,
            flags=re.IGNORECASE,
        )
        if not m:
            continue
        head = str(m.group(1)).strip().lower()
        scope = _normalize_scope(str(m.group(2) or "common"))
        phrase = _normalize_line(str(m.group(3)))
        if not phrase:
            continue

        if head in {"금지", "금지표현", "forbid"}:
            forbidden.append(phrase)
            forbidden_by_scope.setdefault(scope, []).append(phrase)
            continue
        if head in {"필수", "required"}:
            required.append(phrase)
            required_by_scope.setdefault(scope, []).append(phrase)
            continue

    return ComplianceRuleSet(
        source_path=None,
        raw_text=raw_text,
        forbidden_patterns=forbidden,
        required_phrases=required,
        forbidden_by_scope=forbidden_by_scope,
        required_by_scope=required_by_scope,
    )


def load_compliance_rules(path: Optional[Path]) -> ComplianceRuleSet:
    if path is None:
        return ComplianceRuleSet(source_path=None, raw_text="")
    p = Path(path)
    if not p.exists():
        return ComplianceRuleSet(source_path=str(p), raw_text="")
    raw = p.read_text(encoding="utf-8").strip()
    rs = _parse_rule_lines(raw)
    rs.source_path = str(p)
    return rs


def _pick_scoped(rules_map: Dict[str, List[str]], family: Optional[str]) -> List[str]:
    fam = str(family or "").strip().lower()
    picked: List[str] = []
    picked.extend(rules_map.get("common", []))
    if fam in {"deposit", "fund"}:
        picked.extend(rules_map.get(fam, []))
    return picked


def scoped_forbidden_patterns(rules: ComplianceRuleSet, family: Optional[str]) -> List[str]:
    return _pick_scoped(rules.forbidden_by_scope, family)


def scoped_required_phrases(rules: ComplianceRuleSet, family: Optional[str]) -> List[str]:
    return _pick_scoped(rules.required_by_scope, family)


def evaluate_compliance_rules(
    text: str,
    rules: ComplianceRuleSet,
    family: Optional[str] = None,
) -> Dict[str, List[str]]:
    body = str(text or "")
    lower_body = body.lower()
    forbidden_pool = scoped_forbidden_patterns(rules, family)
    required_pool = scoped_required_phrases(rules, family)

    forbidden_hits: List[str] = []
    for pat in forbidden_pool:
        if pat and pat.lower() in lower_body:
            forbidden_hits.append(pat)

    missing_required: List[str] = []
    for phrase in required_pool:
        if phrase and phrase.lower() not in lower_body:
            missing_required.append(phrase)

    return {
        "forbidden_hits": forbidden_hits,
        "missing_required": missing_required,
    }
