#!/usr/bin/env python3
"""Build a product-level feature database for recall prediction."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_initialize.build_match_database import (  # noqa: E402
    apply_sqlite_ram_settings,
    build_event_rows,
    parse_yyyymmdd,
)

# change data directory and default DB paths as needed
DATA_DIR = Path("/Volumes/雷电硬盘盒/data") 

DEFAULT_FAERS_DB_PATH = DATA_DIR / "faers_2004_2008.db"  # the path of original data
DEFAULT_MATCH_DB_PATH = DATA_DIR / "match_through_2008_04.db"    # the path of matched data
DEFAULT_OUTPUT_DB_PATH = DATA_DIR / "product_recall_features_2004_2008.db"  # output path for the product feature database


DEFAULT_FAERS_START_DATE = "20040102"
DEFAULT_FAERS_END_DATE = "20080430"

HIGH_RISK_SCORE = 3.5

ROUTE_RULES = [
    (4.0, ("intrathecal", "intraventricular", "intravitreal", "intraocular", "epidural")),
    (3.5, ("intravenous", "infusion", "injectable", "injection")),
    (3.0, ("intramuscular", "subcutaneous", "implant", "inhalation", "nebul")),
    (2.5, ("ophthalmic", "otic", "nasal", "rectal", "vaginal", "transdermal")),
    (1.5, ("topical", "dermal", "cutaneous")),
    (1.0, ("oral", "enteral", "sublingual", "buccal")),
]

INDICATION_RULES = [
    (4.0, ("cancer", "carcinoma", "tumor", "tumour", "leukemia", "lymphoma", "myeloma", "neoplasm", "metastatic")),
    (4.0, ("transplant", "graft", "rejection", "gvhd", "immunosuppression")),
    (3.5, ("sepsis", "septic", "meningitis", "hiv", "aids", "tuberculosis", "hepatitis", "pneumonia")),
    (3.5, ("heart failure", "myocardial infarction", "arrhythmia", "thrombosis", "embolism", "stroke")),
    (3.0, ("renal failure", "kidney failure", "dialysis", "epilepsy", "seizure", "schizophrenia", "bipolar")),
    (2.5, ("diabetes", "hypertension", "asthma", "copd", "depression")),
    (1.5, ("pain", "nausea", "vomiting", "rash", "dermatitis", "allergy", "acne", "cold")),
]


@dataclass
class Stats:
    faers_report_count_total: int = 0
    unique_safetyreport_count: int = 0
    faers_report_count_last_365d: int = 0
    faers_report_count_prev_365d: int = 0
    serious_count: int = 0
    death_count: int = 0
    hospitalization_count: int = 0
    lifethreatening_count: int = 0
    disabling_count: int = 0
    other_serious_count: int = 0
    suspect_drug_count: int = 0
    reaction_count: int = 0
    indication_score_sum: float = 0.0
    indication_score_n: int = 0
    indication_score_max: float = 0.0
    high_risk_indication_count: int = 0
    route_score_sum: float = 0.0
    route_score_n: int = 0
    route_score_max: float = 0.0
    high_risk_route_count: int = 0
    patient_age_sum: float = 0.0
    patient_age_count: int = 0
    patient_age_missing_count: int = 0
    female_count: int = 0
    male_count: int = 0
    unknown_sex_count: int = 0
    first_faers_date: date | None = None
    last_faers_date: date | None = None
    has_application_number: int = 0
    has_brand_name: int = 0
    has_generic_name: int = 0
    has_substance_name: int = 0
    has_manufacturer_name: int = 0
    has_drugindication: int = 0
    has_administrationroute: int = 0
    reactions: set[str] = field(default_factory=set)
    indications: set[str] = field(default_factory=set)
    routes: set[str] = field(default_factory=set)


# Read command-line options.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build product-level FAERS feature DB with recall labels."
    )
    parser.add_argument(
        "--faers-db",
        default=str(DEFAULT_FAERS_DB_PATH),
        help="Path to filtered FAERS SQLite DB.",
    )
    parser.add_argument(
        "--match-db",
        default=str(DEFAULT_MATCH_DB_PATH),
        help="Path to matched SQLite DB used for recall labels.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=str(DEFAULT_OUTPUT_DB_PATH),
        help="Output SQLite DB path.",
    )
    parser.add_argument(
        "--faers-table",
        default="openfda_drug_event_raw",
        help="FAERS source table name.",
    )
    parser.add_argument(
        "--match-table",
        default="match_index",
        help="Match table name.",
    )
    parser.add_argument(
        "--entity-level",
        choices=["product", "application", "ingredient"],
        default="product",
        help="Entity level. Default: product.",
    )
    parser.add_argument(
        "--faers-start-date",
        default=DEFAULT_FAERS_START_DATE,
        help="Inclusive FAERS feature start date, YYYYMMDD.",
    )
    parser.add_argument(
        "--faers-end-date",
        default=DEFAULT_FAERS_END_DATE,
        help="Inclusive FAERS feature end/cutoff date, YYYYMMDD.",
    )
    parser.add_argument(
        "--max-faers-rows",
        type=int,
        default=None,
        help="Debug cap on scanned FAERS rows.",
    )
    parser.add_argument(
        "--max-match-rows",
        type=int,
        default=None,
        help="Debug cap on scanned match rows.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50_000,
        help="Print progress every N FAERS rows.",
    )
    parser.add_argument(
        "--sqlite-cache-mib",
        type=int,
        default=512,
        help="SQLite page cache size in MiB.",
    )
    parser.add_argument(
        "--sqlite-mmap-mib",
        type=int,
        default=2048,
        help="SQLite mmap cap in MiB.",
    )
    return parser.parse_args()


# Open a SQLite database without writes.
def connect_read_only(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


# Quote table names for SQL.
def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


# Parse FAERS payload JSON.
def parse_json_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    parsed = json.loads(value)
    return parsed if isinstance(parsed, dict) else {}


# Convert YYYYMMDD values to date.
def parse_date_yyyymmdd(value: Any) -> date | None:
    parsed = parse_yyyymmdd(value)
    if parsed is None:
        return None
    if isinstance(parsed, datetime):
        return parsed.date()
    return parsed


# Convert common FAERS flags to booleans.
def truthy_flag(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def clean_lower(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


# Score route and indication text with simple keyword rules.
def text_score(text: str, rules: list[tuple[float, tuple[str, ...]]]) -> float:
    score = 0.0
    for weight, keys in rules:
        if any(key in text for key in keys):
            score = max(score, weight)
    return score


def split_joined_values(value: Any) -> Iterable[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in str(value).split("|") if item.strip()]


# Load product keys that appear in the match database.
def load_recalled_keys(
    conn: sqlite3.Connection,
    *,
    table: str,
    entity_level: str,
    max_match_rows: int | None,
    progress_every: int,
) -> set[str]:
    recalled: set[str] = set()
    n = 0
    cur = conn.execute(
        f"""
        SELECT entity_key
        FROM {quote_identifier(table)}
        WHERE entity_level = ?
        """,
        (entity_level,),
    )
    for row in cur:
        if max_match_rows is not None and n >= max_match_rows:
            break
        n += 1
        key = str(row["entity_key"] or "").strip()
        if key:
            recalled.add(key)
        if progress_every > 0 and n % progress_every == 0:
            print(
                f"Scanned {n:,} match rows | recalled_keys={len(recalled):,}",
                flush=True,
            )
    print(
        f"Loaded recalled keys: {len(recalled):,} from {n:,} match rows",
        flush=True,
    )
    return recalled


# Collect reaction terms for one FAERS report.
def reaction_terms(payload: dict[str, Any]) -> list[str]:
    patient = payload.get("patient")
    if not isinstance(patient, dict):
        return []
    reactions = patient.get("reaction")
    if not isinstance(reactions, list):
        return []
    terms: list[str] = []
    for reaction in reactions:
        if not isinstance(reaction, dict):
            continue
        term = clean_lower(reaction.get("reactionmeddrapt"))
        if term:
            terms.append(term)
    return terms


# Convert FAERS patient age to years.
def patient_age_years(payload: dict[str, Any]) -> float | None:
    patient = payload.get("patient")
    if not isinstance(patient, dict):
        return None
    raw_age = patient.get("patientonsetage")
    if raw_age in (None, ""):
        return None
    age = float(raw_age)
    unit = str(patient.get("patientonsetageunit") or "").strip()
    if unit in {"801", "year", "years", "yr", "yrs", ""}:
        return age
    if unit in {"800", "decade", "decades"}:
        return age * 10.0
    if unit in {"802", "month", "months", "mo"}:
        return age / 12.0
    if unit in {"803", "week", "weeks", "wk"}:
        return age / 52.0
    if unit in {"804", "day", "days"}:
        return age / 365.25
    if unit in {"805", "hour", "hours"}:
        return age / (365.25 * 24.0)
    return None


# Convert patient sex code to a short category.
def patient_sex(payload: dict[str, Any]) -> str:
    patient = payload.get("patient")
    if not isinstance(patient, dict):
        return "unknown"
    sex = str(patient.get("patientsex") or "").strip().lower()
    if sex in {"1", "m", "male"}:
        return "male"
    if sex in {"2", "f", "female"}:
        return "female"
    return "unknown"


# Track first and last FAERS dates for one product.
def update_dates(stats: Stats, received: date) -> None:
    if stats.first_faers_date is None or received < stats.first_faers_date:
        stats.first_faers_date = received
    if stats.last_faers_date is None or received > stats.last_faers_date:
        stats.last_faers_date = received


# Add one report/product occurrence into the aggregate stats.
def add_report(
    stats: Stats,
    *,
    payload: dict[str, Any],
    entity_row: dict[str, Any],
    received: date,
    reaction_list: list[str],
    age_years: float | None,
    sex: str,
    cutoff_date: date,
    last_365_start: date,
    prev_365_start: date,
    prev_365_end: date,
) -> None:
    stats.faers_report_count_total += 1
    stats.unique_safetyreport_count += 1
    if last_365_start <= received <= cutoff_date:
        stats.faers_report_count_last_365d += 1
    if prev_365_start <= received <= prev_365_end:
        stats.faers_report_count_prev_365d += 1

    if truthy_flag(payload.get("serious") or entity_row.get("serious")):
        stats.serious_count += 1
    if truthy_flag(payload.get("seriousnessdeath") or entity_row.get("seriousnessdeath")):
        stats.death_count += 1
    if truthy_flag(payload.get("seriousnesshospitalization")):
        stats.hospitalization_count += 1
    if truthy_flag(payload.get("seriousnesslifethreatening")):
        stats.lifethreatening_count += 1
    if truthy_flag(payload.get("seriousnessdisabling")):
        stats.disabling_count += 1
    if truthy_flag(payload.get("seriousnessother")):
        stats.other_serious_count += 1

    if str(entity_row.get("drugcharacterization") or "").strip() == "1":
        stats.suspect_drug_count += 1

    stats.reaction_count += len(reaction_list)
    stats.reactions.update(reaction_list)

    indication = clean_lower(entity_row.get("drugindication"))
    if indication:
        stats.indications.add(indication)
        stats.has_drugindication = 1
        score = text_score(indication, INDICATION_RULES)
        stats.indication_score_sum += score
        stats.indication_score_n += 1
        stats.indication_score_max = max(stats.indication_score_max, score)
        if score >= HIGH_RISK_SCORE:
            stats.high_risk_indication_count += 1
    route = clean_lower(entity_row.get("drugadministrationroute"))
    if route:
        stats.routes.add(route)
        stats.has_administrationroute = 1
        score = text_score(route, ROUTE_RULES)
        stats.route_score_sum += score
        stats.route_score_n += 1
        stats.route_score_max = max(stats.route_score_max, score)
        if score >= HIGH_RISK_SCORE:
            stats.high_risk_route_count += 1

    if list(split_joined_values(entity_row.get("openfda_application_number"))):
        stats.has_application_number = 1
    if list(split_joined_values(entity_row.get("openfda_brand_name"))):
        stats.has_brand_name = 1
    if list(split_joined_values(entity_row.get("openfda_generic_name"))):
        stats.has_generic_name = 1
    if list(split_joined_values(entity_row.get("openfda_substance_name"))):
        stats.has_substance_name = 1
    if list(split_joined_values(entity_row.get("openfda_manufacturer_name"))):
        stats.has_manufacturer_name = 1

    if age_years is None or not math.isfinite(age_years) or age_years < 0:
        stats.patient_age_missing_count += 1
    else:
        stats.patient_age_sum += age_years
        stats.patient_age_count += 1

    if sex == "female":
        stats.female_count += 1
    elif sex == "male":
        stats.male_count += 1
    else:
        stats.unknown_sex_count += 1

    update_dates(stats, received)


# Scan FAERS rows and aggregate features by product key.
def aggregate_faers(
    conn: sqlite3.Connection,
    *,
    table: str,
    entity_level: str,
    faers_start_date: str,
    faers_end_date: str,
    cutoff_date: date,
    max_faers_rows: int | None,
    progress_every: int,
) -> tuple[dict[str, Stats], int, int]:
    by_key: dict[str, Stats] = {}
    n = 0
    updates = 0
    last_365_start = cutoff_date - timedelta(days=364)
    prev_365_end = last_365_start - timedelta(days=1)
    prev_365_start = prev_365_end - timedelta(days=364)
    cur = conn.execute(
        f"""
        SELECT id, safetyreportid, receivedate, serious, payload_json
        FROM {quote_identifier(table)}
        WHERE receivedate BETWEEN ? AND ?
        ORDER BY receivedate, id
        """,
        (faers_start_date, faers_end_date),
    )
    for row in cur:
        if max_faers_rows is not None and n >= max_faers_rows:
            break
        n += 1
        received = parse_date_yyyymmdd(row["receivedate"])
        payload = parse_json_obj(row["payload_json"])
        if received is None or not payload:
            continue

        reactions = reaction_terms(payload)
        age = patient_age_years(payload)
        sex = patient_sex(payload)
        seen: set[str] = set()
        for entity_row in build_event_rows(
            payload,
            entity_level=entity_level,
            allow_text_product_fallback=False,
            suspect_only=False,
        ):
            key = str(entity_row.get("entity_key") or "").strip()
            if not key or key in seen:
                continue
            seen.add(key)
            stats = by_key.setdefault(key, Stats())
            add_report(
                stats,
                payload=payload,
                entity_row=entity_row,
                received=received,
                reaction_list=reactions,
                age_years=age,
                sex=sex,
                cutoff_date=cutoff_date,
                last_365_start=last_365_start,
                prev_365_start=prev_365_start,
                prev_365_end=prev_365_end,
            )
            updates += 1

        if progress_every > 0 and n % progress_every == 0:
            print(
                f"Scanned {n:,} FAERS rows | "
                f"products={len(by_key):,} | "
                f"updates={updates:,}",
                flush=True,
            )
    return by_key, n, updates


def safe_rate(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def growth_ratio(recent: int, previous: int) -> float:
    return (recent - previous) / max(previous, 1)


def active_months(first: date | None, last: date | None) -> int:
    if first is None or last is None:
        return 0
    return (last.year - first.year) * 12 + (last.month - first.month) + 1


# Turn one product's aggregate stats into the output row.
def feature_tuple(key: str, stats: Stats, label: int) -> tuple[Any, ...]:
    n = stats.faers_report_count_total
    months = active_months(stats.first_faers_date, stats.last_faers_date)
    active_days = (
        (stats.last_faers_date - stats.first_faers_date).days + 1
        if stats.first_faers_date is not None and stats.last_faers_date is not None
        else 0
    )
    patient_age_total = stats.patient_age_count + stats.patient_age_missing_count
    return (
        key,
        label,
        n,
        stats.unique_safetyreport_count,
        stats.faers_report_count_last_365d,
        stats.faers_report_count_prev_365d,
        growth_ratio(stats.faers_report_count_last_365d, stats.faers_report_count_prev_365d),
        stats.serious_count,
        safe_rate(stats.serious_count, n),
        stats.death_count,
        safe_rate(stats.death_count, n),
        stats.hospitalization_count,
        safe_rate(stats.hospitalization_count, n),
        stats.lifethreatening_count,
        safe_rate(stats.lifethreatening_count, n),
        stats.disabling_count,
        safe_rate(stats.disabling_count, n),
        stats.other_serious_count,
        safe_rate(stats.other_serious_count, n),
        stats.suspect_drug_count,
        safe_rate(stats.suspect_drug_count, n),
        stats.reaction_count,
        len(stats.reactions),
        safe_rate(stats.reaction_count, n),
        len(stats.indications),
        len(stats.routes),
        stats.indication_score_sum,
        safe_rate(stats.indication_score_sum, stats.indication_score_n),
        stats.indication_score_max,
        stats.high_risk_indication_count,
        safe_rate(stats.high_risk_indication_count, stats.indication_score_n),
        stats.route_score_sum,
        safe_rate(stats.route_score_sum, stats.route_score_n),
        stats.route_score_max,
        stats.high_risk_route_count,
        safe_rate(stats.high_risk_route_count, stats.route_score_n),
        stats.has_application_number,
        stats.has_brand_name,
        stats.has_generic_name,
        stats.has_substance_name,
        stats.has_manufacturer_name,
        stats.has_drugindication,
        stats.has_administrationroute,
        stats.first_faers_date.strftime("%Y%m%d") if stats.first_faers_date else None,
        stats.last_faers_date.strftime("%Y%m%d") if stats.last_faers_date else None,
        active_days,
        months,
        safe_rate(n, months),
        safe_rate(stats.patient_age_sum, stats.patient_age_count),
        safe_rate(stats.patient_age_missing_count, patient_age_total),
        safe_rate(stats.female_count, n),
        safe_rate(stats.male_count, n),
    )


# Create a fresh output schema.
def ensure_output_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS dataset_metadata;
        DROP TABLE IF EXISTS product_features;

        CREATE TABLE product_features (
            entity_key TEXT PRIMARY KEY,
            label_recalled INTEGER NOT NULL,
            faers_report_count_total INTEGER NOT NULL,
            unique_safetyreport_count INTEGER NOT NULL,
            faers_report_count_last_365d INTEGER NOT NULL,
            faers_report_count_prev_365d INTEGER NOT NULL,
            faers_report_growth_365d REAL NOT NULL,
            serious_count INTEGER NOT NULL,
            serious_rate REAL NOT NULL,
            death_count INTEGER NOT NULL,
            death_rate REAL NOT NULL,
            hospitalization_count INTEGER NOT NULL,
            hospitalization_rate REAL NOT NULL,
            lifethreatening_count INTEGER NOT NULL,
            lifethreatening_rate REAL NOT NULL,
            disabling_count INTEGER NOT NULL,
            disabling_rate REAL NOT NULL,
            other_serious_count INTEGER NOT NULL,
            other_serious_rate REAL NOT NULL,
            suspect_drug_count INTEGER NOT NULL,
            suspect_drug_rate REAL NOT NULL,
            reaction_count INTEGER NOT NULL,
            unique_reaction_count INTEGER NOT NULL,
            reaction_per_report_mean REAL NOT NULL,
            unique_indication_count INTEGER NOT NULL,
            unique_route_count INTEGER NOT NULL,
            indication_score_sum REAL NOT NULL,
            indication_score_mean REAL NOT NULL,
            indication_score_max REAL NOT NULL,
            high_risk_indication_count INTEGER NOT NULL,
            high_risk_indication_rate REAL NOT NULL,
            route_score_sum REAL NOT NULL,
            route_score_mean REAL NOT NULL,
            route_score_max REAL NOT NULL,
            high_risk_route_count INTEGER NOT NULL,
            high_risk_route_rate REAL NOT NULL,
            has_application_number INTEGER NOT NULL,
            has_brand_name INTEGER NOT NULL,
            has_generic_name INTEGER NOT NULL,
            has_substance_name INTEGER NOT NULL,
            has_manufacturer_name INTEGER NOT NULL,
            has_drugindication INTEGER NOT NULL,
            has_administrationroute INTEGER NOT NULL,
            first_faers_date TEXT,
            last_faers_date TEXT,
            active_days INTEGER NOT NULL,
            active_months INTEGER NOT NULL,
            report_count_per_active_month REAL NOT NULL,
            patient_age_mean REAL NOT NULL,
            patient_age_missing_rate REAL NOT NULL,
            female_rate REAL NOT NULL,
            male_rate REAL NOT NULL
        );

        CREATE TABLE dataset_metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        );

        CREATE INDEX idx_product_features_label
            ON product_features(label_recalled);
        CREATE INDEX idx_product_features_report_count
            ON product_features(faers_report_count_total);
        """
    )
    conn.commit()


# Write build settings for later inspection.
def write_metadata(conn: sqlite3.Connection, values: dict[str, Any]) -> None:
    conn.executemany(
        "INSERT INTO dataset_metadata (key, value) VALUES (?, ?)",
        [(str(key), str(value)) for key, value in values.items()],
    )
    conn.commit()


# Insert product feature rows in batches.
def write_product_features(
    conn: sqlite3.Connection,
    by_key: dict[str, Stats],
    recalled_keys: set[str],
    batch_rows: int = 5000,
) -> tuple[int, int]:
    sql = """
        INSERT INTO product_features (
            entity_key, label_recalled,
            faers_report_count_total, unique_safetyreport_count,
            faers_report_count_last_365d, faers_report_count_prev_365d,
            faers_report_growth_365d,
            serious_count, serious_rate,
            death_count, death_rate,
            hospitalization_count, hospitalization_rate,
            lifethreatening_count, lifethreatening_rate,
            disabling_count, disabling_rate,
            other_serious_count, other_serious_rate,
            suspect_drug_count, suspect_drug_rate,
            reaction_count, unique_reaction_count, reaction_per_report_mean,
            unique_indication_count, unique_route_count,
            indication_score_sum, indication_score_mean, indication_score_max,
            high_risk_indication_count, high_risk_indication_rate,
            route_score_sum, route_score_mean, route_score_max,
            high_risk_route_count, high_risk_route_rate,
            has_application_number, has_brand_name, has_generic_name,
            has_substance_name, has_manufacturer_name,
            has_drugindication, has_administrationroute,
            first_faers_date, last_faers_date,
            active_days, active_months, report_count_per_active_month,
            patient_age_mean, patient_age_missing_rate,
            female_rate, male_rate
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    batch: list[tuple[Any, ...]] = []
    written = 0
    positive = 0
    for key in sorted(by_key):
        label = 1 if key in recalled_keys else 0
        positive += label
        batch.append(feature_tuple(key, by_key[key], label))
        if len(batch) >= batch_rows:
            conn.executemany(sql, batch)
            written += len(batch)
            batch.clear()
            conn.commit()
    if batch:
        conn.executemany(sql, batch)
        written += len(batch)
        batch.clear()
        conn.commit()
    return written, positive


def main() -> int:
    args = parse_args()
    start = parse_date_yyyymmdd(args.faers_start_date)
    cutoff = parse_date_yyyymmdd(args.faers_end_date)
    if start is None or cutoff is None or start > cutoff:
        print("Error: use valid YYYYMMDD dates with start <= end", file=sys.stderr)
        return 2

    faers_path = Path(args.faers_db).expanduser().resolve()
    match_path = Path(args.match_db).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not faers_path.is_file():
        print(f"Error: FAERS DB not found: {faers_path}", file=sys.stderr)
        return 1
    if not match_path.is_file():
        print(f"Error: match DB not found: {match_path}", file=sys.stderr)
        return 1

    print(f"FAERS DB: {faers_path}")
    print(f"Match DB: {match_path}")
    print(f"Output DB: {output_path}")
    print(f"FAERS window: {args.faers_start_date} to {args.faers_end_date}")
    print(f"Entity level: {args.entity_level}")

    mc = connect_read_only(match_path)
    apply_sqlite_ram_settings(mc, cache_mib=args.sqlite_cache_mib, mmap_mib=args.sqlite_mmap_mib)
    recalled = load_recalled_keys(
        mc,
        table=args.match_table,
        entity_level=args.entity_level,
        max_match_rows=args.max_match_rows,
        progress_every=max(args.progress_every, 1_000_000),
    )
    mc.close()

    fc = connect_read_only(faers_path)
    apply_sqlite_ram_settings(fc, cache_mib=args.sqlite_cache_mib, mmap_mib=args.sqlite_mmap_mib)
    by_key, n_rows, updates = aggregate_faers(
        fc,
        table=args.faers_table,
        entity_level=args.entity_level,
        faers_start_date=args.faers_start_date,
        faers_end_date=args.faers_end_date,
        cutoff_date=cutoff,
        max_faers_rows=args.max_faers_rows,
        progress_every=args.progress_every,
    )
    fc.close()

    out = sqlite3.connect(str(output_path))
    apply_sqlite_ram_settings(out, cache_mib=args.sqlite_cache_mib, mmap_mib=args.sqlite_mmap_mib)
    ensure_output_schema(out)
    write_metadata(
        out,
        {
            "faers_db": faers_path,
            "match_db": match_path,
            "faers_table": args.faers_table,
            "match_table": args.match_table,
            "entity_level": args.entity_level,
            "faers_start_date": args.faers_start_date,
            "faers_end_date": args.faers_end_date,
            "cutoff_date": cutoff.strftime("%Y%m%d"),
            "label_definition": "label_recalled=1 if entity_key appears in match_index",
            "risk_score_rules": "indication and route use keyword-based weights",
            "scanned_faers_rows": n_rows,
            "product_updates": updates,
            "recalled_keys_loaded": len(recalled),
            "max_faers_rows": args.max_faers_rows,
            "max_match_rows": args.max_match_rows,
        },
    )
    written, positive = write_product_features(out, by_key, recalled)
    out.close()

    print(f"Scanned FAERS rows: {n_rows:,}")
    print(f"Product update events: {updates:,}")
    print(f"Product rows written: {written:,}")
    print(f"Positive products: {positive:,}")
    print(f"Negative products: {written - positive:,}")
    print("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
