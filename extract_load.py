"""
Extract & Load (sekwencyjny wg relacji PK -> FK): Neon (PostgreSQL) -> BigQuery

Skrypt automatycznie odczytuje relacje kluczy obcych z bazy Neon,
wyznacza kolejność ładowania (sortowanie topologiczne) i ładuje tabele
sekwencyjnie: od tabel nadrzędnych (PK) do podrzędnych (FK).
"""

import heapq
import os
from collections import defaultdict

import pandas as pd
import psycopg2
from psycopg2 import sql
from google.cloud import bigquery

# ============================================================
# KONFIGURACJA
# ============================================================

NEON_CONNECTION_STRING = os.environ["NEON_CONNECTION_STRING"]
BQ_PROJECT_ID = os.environ["BQ_PROJECT_ID"]
BQ_DATASET = os.environ["BQ_DATASET"]
BQ_TABLE_PREFIX = os.getenv("BQ_TABLE_PREFIX", "")

DB_SCHEMA = os.getenv("DB_SCHEMA", "public")
DEFAULT_LOAD_MODE = os.getenv("DEFAULT_LOAD_MODE", "incremental")
DEFAULT_INCREMENTAL_COLUMN = os.getenv("DEFAULT_INCREMENTAL_COLUMN", "updated_at")

EXCLUDE_TABLES = {
    t.strip() for t in os.getenv("EXCLUDE_TABLES", "").split(",") if t.strip()
}

# Opcjonalne nadpisania dla konkretnych tabel (klucz = nazwa tabeli w Neon).
# Jeśli tabela nie jest tu wpisana, użyje ustawień domyślnych powyżej.
TABLE_CONFIG = {
    # Przykład dla tabeli słownikowej bez kolumny updated_at:
    # "specializations": {"load_mode": "full_refresh"},
    # Przykład z inną kolumną przyrostową lub docelową nazwą w BigQuery:
    # "appointments": {
    #     "load_mode": "incremental",
    #     "incremental_column": "updated_at",
    #     "bq_table": "appointments",
    # },
}

# ============================================================
# ANALIZA RELACJI PK-FK I SORTOWANIE TOPOLOGICZNE
# ============================================================


def fetch_schema_metadata(conn, schema: str):
    """Pobiera listę tabel, ich kolumny oraz krawędzie zależności (parent -> child)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT tablename
            FROM pg_tables
            WHERE schemaname = %s
            ORDER BY tablename;
            """,
            (schema,),
        )
        tables = [row[0] for row in cur.fetchall() if row[0] not in EXCLUDE_TABLES]

        cur.execute(
            """
            SELECT table_name, column_name
            FROM information_schema.columns
            WHERE table_schema = %s;
            """,
            (schema,),
        )
        columns_by_table = defaultdict(set)
        for table_name, col_name in cur.fetchall():
            columns_by_table[table_name].add(col_name)

        # Relacje FK (pomijamy self-referencing FK, żeby nie blokować sortowania)
        cur.execute(
            """
            SELECT DISTINCT
                parent_tbl.relname AS parent_table,
                child_tbl.relname AS child_table
            FROM pg_constraint con
            JOIN pg_class child_tbl ON child_tbl.oid = con.conrelid
            JOIN pg_namespace child_ns ON child_ns.oid = child_tbl.relnamespace
            JOIN pg_class parent_tbl ON parent_tbl.oid = con.confrelid
            JOIN pg_namespace parent_ns ON parent_ns.oid = parent_tbl.relnamespace
            WHERE con.contype = 'f'
              AND child_ns.nspname = %s
              AND parent_ns.nspname = %s
              AND child_tbl.relname <> parent_tbl.relname;
            """,
            (schema, schema),
        )
        fk_edges = cur.fetchall()

    return tables, columns_by_table, fk_edges


def topological_sort_tables(tables: list[str], fk_edges: list[tuple[str, str]]) -> list[str]:
    """Wyznacza kolejność ładowania tabel od rodziców (PK) do dzieci (FK) algorytmem Kahna."""
    table_set = set(tables)
    in_degree = {t: 0 for t in tables}
    children = {t: set() for t in tables}

    for parent, child in fk_edges:
        if parent in table_set and child in table_set:
            if child not in children[parent]:
                children[parent].add(child)
                in_degree[child] += 1

    # Min-heap zapewnia deterministyczną kolejność (alfabetyczną w ramach tego samego poziomu)
    ready = [t for t in tables if in_degree[t] == 0]
    heapq.heapify(ready)

    ordered = []
    while ready:
        current = heapq.heappop(ready)
        ordered.append(current)
        for child in children[current]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                heapq.heappush(ready, child)

    if len(ordered) != len(tables):
        cyclic = sorted(table_set - set(ordered))
        raise RuntimeError(
            f"Wykryto cykliczną zależność FK między tabelami: {', '.join(cyclic)}. "
            "Rozwiąż cykl lub wyklucz wybrane tabele w EXCLUDE_TABLES."
        )

    return ordered


# ============================================================
# LOGIKA EXTRACT & LOAD
# ============================================================


def get_last_loaded_timestamp(
    bq_client: bigquery.Client, table_ref: str, incremental_col: str
):
    """Odczytuje maksymalny timestamp już obecny w tabeli docelowej BigQuery."""
    query = f"SELECT MAX(`{incremental_col}`) AS max_ts FROM `{table_ref}`"
    try:
        result = bq_client.query(query).result()
        row = next(iter(result))
        return row.max_ts
    except Exception:
        return None


def extract_from_neon(
    conn,
    schema: str,
    table_name: str,
    mode: str,
    incremental_col: str | None = None,
    last_ts=None,
) -> pd.DataFrame:
    """Wyciąga dane z pojedynczej tabeli w Neon."""
    if mode == "incremental" and incremental_col and last_ts is not None:
        query = sql.SQL("SELECT * FROM {}.{} WHERE {} > %s").format(
            sql.Identifier(schema),
            sql.Identifier(table_name),
            sql.Identifier(incremental_col),
        )
        return pd.read_sql(query.as_string(conn), conn, params=(last_ts,))

    query = sql.SQL("SELECT * FROM {}.{}").format(
        sql.Identifier(schema),
        sql.Identifier(table_name),
    )
    return pd.read_sql(query.as_string(conn), conn)


def load_to_bigquery(
    bq_client: bigquery.Client, df: pd.DataFrame, table_ref: str, mode: str
):
    """Ładuje DataFrame do BigQuery (WRITE_TRUNCATE lub WRITE_APPEND)."""
    write_disposition = (
        bigquery.WriteDisposition.WRITE_TRUNCATE
        if mode == "full_refresh"
        else bigquery.WriteDisposition.WRITE_APPEND
    )
    job_config = bigquery.LoadJobConfig(write_disposition=write_disposition)
    job = bq_client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()
    return job


def main():
    bq_client = bigquery.Client(project=BQ_PROJECT_ID)
    conn = psycopg2.connect(NEON_CONNECTION_STRING)

    try:
        tables, columns_by_table, fk_edges = fetch_schema_metadata(conn, DB_SCHEMA)
        if not tables:
            print(f"Brak tabel do załadowania w schemacie '{DB_SCHEMA}'.")
            return

        load_order = topological_sort_tables(tables, fk_edges)
        print(f"Kolejność ładowania (PK -> FK): {' -> '.join(load_order)}")

        for index, source_table in enumerate(load_order, start=1):
            cfg = TABLE_CONFIG.get(source_table, {})
            mode = cfg.get("load_mode", DEFAULT_LOAD_MODE)
            incremental_col = cfg.get("incremental_column", DEFAULT_INCREMENTAL_COLUMN)
            bq_table_name = cfg.get("bq_table", f"{BQ_TABLE_PREFIX}{source_table}")
            table_ref = f"{BQ_PROJECT_ID}.{BQ_DATASET}.{bq_table_name}"

            print(f"\n[{index}/{len(load_order)}] Tabela: {source_table} -> {table_ref}")

            # Fallback: jeśli tabela nie posiada kolumny przyrostowej, przełącz na full_refresh
            if mode == "incremental" and incremental_col not in columns_by_table[source_table]:
                print(
                    f"  Uwaga: brak kolumny '{incremental_col}' w '{source_table}' "
                    "-- przełączam na full_refresh."
                )
                mode = "full_refresh"

            last_ts = None
            if mode == "incremental":
                last_ts = get_last_loaded_timestamp(bq_client, table_ref, incremental_col)
                print(f"  Ostatni załadowany timestamp ({incremental_col}): {last_ts}")

            df = extract_from_neon(
                conn, DB_SCHEMA, source_table, mode, incremental_col, last_ts
            )
            print(f"  Wyciągnięto {len(df)} wierszy z Neon (tryb: {mode})")

            if df.empty:
                print("  Brak nowych danych -- pomijam ładowanie.")
                continue

            job = load_to_bigquery(bq_client, df, table_ref, mode)
            print(f"  Załadowano {job.output_rows} wierszy do {table_ref}")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
