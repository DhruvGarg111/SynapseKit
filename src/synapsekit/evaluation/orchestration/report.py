"""Orchestration evaluation report."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .detectors import DetectorFinding


@dataclass
class OrchestrationEvalReport:
    """Report summarizing findings from orchestration evaluation."""

    run_ids: list[str] = field(default_factory=list)
    findings: list[DetectorFinding] = field(default_factory=list)
    thresholds: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, int] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """True if there are no warning or critical findings."""
        return not any(f.severity in ("warning", "critical") for f in self.findings)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "run_ids": self.run_ids,
            "passed": self.passed,
            "summary": self.summary,
            "thresholds": self.thresholds,
            "findings": [asdict(f) for f in self.findings],
        }

    def to_markdown(self) -> str:
        """Render human-readable markdown report."""
        lines: list[str] = []
        status_symbol = "PASSED" if self.passed else "FAILED"
        lines.append(f"# Orchestration Evaluation Report - [{status_symbol}]")
        lines.append("")
        lines.append(f"**Runs Evaluated:** {len(self.run_ids)} ({', '.join(self.run_ids)})")
        lines.append("")
        lines.append("## Summary")
        lines.append("")
        lines.append("| Failure Mode | Findings Count |")
        lines.append("|---|---|")
        lines.append(f"| Infinite Loops | {self.summary.get('loop', 0)} |")
        lines.append(f"| Context Loss | {self.summary.get('context_loss', 0)} |")
        lines.append(f"| Mis-routing | {self.summary.get('misrouting', 0)} |")
        lines.append("")

        if not self.findings:
            lines.append("No orchestration failure modes detected.")
        else:
            lines.append("## Detailed Findings")
            lines.append("")
            for f in self.findings:
                sev_label = f"[{f.severity.upper()}]"
                lines.append(f"- {sev_label} **[{f.mode.upper()}]**: {f.message}")
                if f.evidence:
                    for k, v in f.evidence.items():
                        lines.append(f"  - `{k}`: {v}")

        return "\n".join(lines)
