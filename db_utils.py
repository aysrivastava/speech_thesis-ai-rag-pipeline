"""
db_utils.py
-------------
The one place that knows the shape of the `sign_embeddings` table, used by
04 (bulk store), retrieval_utils (read), and enrollment_utils (add a new
sign later). Deliberately imports only duckdb + config - no mediapipe - so
04 stays light.

Why the schema changed from the original 04:
  1. `label` was a PRIMARY KEY, which meant exactly one clip per sign. The
     moment you enroll a second example of a sign (or re-record one), that
     insert would fail. Removing the key allows several examples per label;
     retrieval collapses them to the best-scoring example per label.
  2. New `provenance` column: 'curated' (came from data_raw_videos via
     steps 01-04) vs 'manual' (you enrolled it with 07) vs 'vlm_confirmed'
     (a VLM proposed the label and you confirmed it). Keeping this lets you
     (a) report results with/without auto-enrolled signs in a paper and
     (b) re-run 04 without wiping enrolled signs.

ensure_schema() also MIGRATES an existing database in place (the one you
already built with the old 04), preserving every row.
"""

from datetime import datetime

import duckdb

import config

CREATE_SQL = """
CREATE TABLE IF NOT EXISTS sign_embeddings (
    label         VARCHAR,
    source_video  VARCHAR,
    embedding     DOUBLE[],
    num_frames    INTEGER,
    duration_sec  DOUBLE,
    stored_at     TIMESTAMP,
    provenance    VARCHAR
)
"""

PROVENANCE_CURATED = "curated"
PROVENANCE_MANUAL = "manual"
PROVENANCE_VLM_CONFIRMED = "vlm_confirmed"


def connect(path=None):
    return duckdb.connect(path or config.PATHS["duckdb_file"])


def ensure_schema(con, verbose=False):
    """Creates the table if missing, or migrates an old-schema table
    (PRIMARY KEY on label, no provenance column) to the current one,
    keeping all existing rows (marked 'curated')."""
    con.execute(CREATE_SQL)

    cols = {r[0] for r in con.execute("DESCRIBE sign_embeddings").fetchall()}
    has_pk = con.execute(
        "SELECT COUNT(*) FROM duckdb_constraints() "
        "WHERE table_name = 'sign_embeddings' AND constraint_type = 'PRIMARY KEY'"
    ).fetchone()[0] > 0

    if has_pk or "provenance" not in cols:
        if verbose:
            print("[db] Migrating sign_embeddings to the multi-example schema "
                  "(dropping PRIMARY KEY on label, adding 'provenance')...")
        con.execute(
            """
            CREATE TABLE sign_embeddings_new (
                label VARCHAR, source_video VARCHAR, embedding DOUBLE[],
                num_frames INTEGER, duration_sec DOUBLE, stored_at TIMESTAMP,
                provenance VARCHAR
            )
            """
        )
        prov_expr = "provenance" if "provenance" in cols else f"'{PROVENANCE_CURATED}'"
        con.execute(
            f"INSERT INTO sign_embeddings_new "
            f"SELECT label, source_video, embedding, num_frames, duration_sec, stored_at, {prov_expr} "
            f"FROM sign_embeddings"
        )
        con.execute("DROP TABLE sign_embeddings")
        con.execute("ALTER TABLE sign_embeddings_new RENAME TO sign_embeddings")


def insert_row(con, label, source_video, embedding, num_frames, duration_sec,
               provenance=PROVENANCE_CURATED, stored_at=None):
    con.execute(
        "INSERT INTO sign_embeddings "
        "(label, source_video, embedding, num_frames, duration_sec, stored_at, provenance) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [label, source_video, [float(v) for v in embedding], int(num_frames),
         float(duration_sec), stored_at or datetime.now(), provenance],
    )


def delete_curated_rows(con):
    """What 04 uses on re-run: replace only the rows that came from
    data_raw_videos, leaving enrolled signs alone."""
    con.execute("DELETE FROM sign_embeddings WHERE provenance = ?", [PROVENANCE_CURATED])
