"""
Step 2 test: proves the DB layer works before anything else depends on it.

Run with:  python scripts/test_db_roundtrip.py

Expected: prints the inserted paper's id and title, with no errors.
Run it TWICE - the second run should print the SAME id (proves get_or_create
is idempotent and doesn't create duplicate rows).
"""

from core.db.session import get_session
from core.db.repositories.paper_repository import PaperRepository
from core.db.models import PaperSource

with get_session() as session:
    repo = PaperRepository(session)
    paper = repo.get_or_create(
        source=PaperSource.peerread,
        external_id="test-roundtrip-001",
        title="A Test Paper For DB Round-Trip Verification",
        abstract="This is not a real paper - just proving insert/read works.",
        year=2026,
    )
    print(f"Inserted/found paper -> id={paper.id}, title={paper.title!r}")
