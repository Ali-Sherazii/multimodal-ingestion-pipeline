"""Report (FR-14): a summary of each run, printed and saved as JSON."""
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from ingest.models import FileResult


@dataclass
class RunReport:
    started_at: str
    finished_at: str | None = None
    files: list[FileResult] = field(default_factory=list)

    @classmethod
    def start(cls) -> "RunReport":
        return cls(started_at=datetime.now(timezone.utc).isoformat())

    def add(self, result: FileResult) -> None:
        self.files.append(result)

    def finish(self) -> None:
        self.finished_at = datetime.now(timezone.utc).isoformat()

    def _count(self, status: str) -> int:
        return sum(1 for f in self.files if f.status == status)

    @property
    def totals(self) -> dict:
        return {
            "files": len(self.files),
            "processed": self._count("processed"),
            "skipped": self._count("skipped"),
            "failed": self._count("failed"),
            "removed": self._count("removed"),  # deleted from a synced folder
            "with_warnings": sum(1 for f in self.files if f.warning),
            "chunks_created": sum(f.chunks for f in self.files),
        }

    def save(self, reports_dir: Path) -> Path:
        reports_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.fromisoformat(self.started_at).strftime("%Y%m%d-%H%M%S")
        path = reports_dir / f"run-{stamp}.json"
        data = {"started_at": self.started_at, "finished_at": self.finished_at,
                "totals": self.totals, "files": [asdict(f) for f in self.files]}
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def summary(self) -> str:
        t = self.totals
        lines = ["", "Run summary",
                 f"  files: {t['files']}   processed: {t['processed']}   skipped: {t['skipped']}"
                 f"   failed: {t['failed']}" + (f"   removed: {t['removed']}" if t["removed"] else "")
                 + f"   with warnings: {t['with_warnings']}"
                 f"   chunks created: {t['chunks_created']}", ""]
        width = max((len(Path(f.path).name) for f in self.files), default=4)
        for f in self.files:
            detail = (f"{f.chunks} chunk{'s' * (f.chunks != 1)}" if f.status == "processed"
                      else (f.reason or ""))
            lines.append(f"  {f.status:<9} {Path(f.path).name:<{width}}  {f.file_type or '-':<5}"
                         f"  {f.seconds:>6.1f}s  {detail}")
            if f.warning:
                lines.append(f"  {'':<9} WARNING: {f.warning}")
        return "\n".join(lines)
