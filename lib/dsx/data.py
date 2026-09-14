"""Dataset registry with explicit provenance, content-addressed caching and integrity checks.

Design rationale
----------------
Every dataset used anywhere in this portfolio is declared here, once, with a machine
readable provenance record. Nothing in a model script is allowed to invent data or
reach out to an undeclared URL. That gives two properties we care about:

1. **Auditability.** ``provenance_table()`` renders the exact origin of every byte the
   portfolio trains on, and that table is what ships in the READMEs. A reader can tell
   at a glance whether a number came from real observations or a simulator.
2. **Reproducibility.** Downloads are cached under ``.data/`` keyed by dataset id, and
   the SHA-256 of the payload is recorded on first fetch. Subsequent runs verify against
   it, so a silently changed upstream file fails loudly instead of quietly shifting every
   metric in the repo.

The ``kind`` field is deliberately blunt. ``REAL`` means observations someone actually
recorded. ``SIMULATED`` means we generated it. There is no third category and no
euphemism: a simulated dataset never gets described as if it were real, in code or in
prose.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Literal

import pandas as pd

# Repository root, resolved from this file so scripts work from any cwd.
ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / ".data"
MANIFEST_PATH = CACHE_DIR / "manifest.json"

DataKind = Literal["REAL", "SIMULATED"]


@dataclass(frozen=True)
class Dataset:
    """A declared data source.

    Attributes
    ----------
    id:
        Stable short key used by loaders and by the provenance table.
    title:
        Human readable name as it appears in documentation.
    kind:
        ``REAL`` for recorded observations, ``SIMULATED`` for anything we generate.
    url:
        Direct download URL. ``None`` for simulated sources.
    origin:
        Where the data ultimately came from, in prose. This is the honest description of
        the collecting body, not the mirror we happen to download from.
    mirror_note:
        Why we fetch from this particular host rather than the canonical one.
    license:
        Licence or usage terms as published by the origin.
    rows:
        Approximate row count, for orientation only.
    """

    id: str
    title: str
    kind: DataKind
    origin: str
    license: str
    url: str | None = None
    mirror_note: str = ""
    rows: str = ""
    notes: str = ""


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------
# Only ``raw.githubusercontent.com`` is reachable from the build environment, so several
# entries point at community mirrors of canonical datasets. The ``origin`` field always
# names the true source; ``mirror_note`` explains the indirection. Nothing here is
# relabelled to look more authoritative than it is.

REGISTRY: dict[str, Dataset] = {
    "uber_nyc_2014": Dataset(
        id="uber_nyc_2014",
        title="Uber NYC Pickups (TLC FOIL response, Apr-Sep 2014)",
        kind="REAL",
        origin=(
            "NYC Taxi & Limousine Commission, released to FiveThirtyEight under a "
            "Freedom of Information Law request. Each row is one real dispatched pickup."
        ),
        license="Made public by FiveThirtyEight; TLC data is a public record.",
        url=(
            "https://raw.githubusercontent.com/fivethirtyeight/uber-tlc-foil-response/"
            "master/uber-trip-data/uber-raw-data-{month}14.csv"
        ),
        mirror_note="FiveThirtyEight's public archive of the TLC FOIL response.",
        rows="~4.5M across six monthly files",
        notes="Columns: Date/Time, Lat, Lon, Base. No trip duration or fare field.",
    ),
    "credit_card_fraud": Dataset(
        id="credit_card_fraud",
        title="Credit Card Fraud Detection (ULB, Sep 2013)",
        kind="REAL",
        origin=(
            "Machine Learning Group, Universite Libre de Bruxelles. Real card "
            "transactions by European cardholders over two days in September 2013. "
            "V1-V28 are PCA components published in place of the raw features for "
            "confidentiality; Time and Amount are unmodified."
        ),
        license="Open Database License (ODbL) - research and educational use.",
        url=(
            "https://raw.githubusercontent.com/nsethi31/"
            "Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"
        ),
        mirror_note=(
            "kaggle.com is unreachable from this build environment (gateway denies "
            "CONNECT), so we fetch the identical file from a GitHub mirror. Integrity is "
            "pinned by the SHA-256 recorded in .data/manifest.json."
        ),
        rows="284,807 transactions, 492 fraudulent (0.1727%)",
        notes="Severe class imbalance. Accuracy is meaningless here; we report PR-AUC.",
    ),
    "groceries": Dataset(
        id="groceries",
        title="Groceries Market Basket Transactions",
        kind="REAL",
        origin=(
            "One month of real point-of-sale transactions from a grocery outlet, "
            "distributed with the R 'arules' package (Hahsler et al.)."
        ),
        license="GPL-2, as distributed with the arules R package.",
        url=(
            "https://raw.githubusercontent.com/stedy/"
            "Machine-Learning-with-R-datasets/master/groceries.csv"
        ),
        mirror_note="Standard GitHub mirror of the arules Groceries dataset.",
        rows="9,835 transactions over 169 item categories",
        notes="Ragged CSV: one transaction per line, variable item count.",
    ),
    "titanic": Dataset(
        id="titanic",
        title="Titanic Passenger Manifest",
        kind="REAL",
        origin=(
            "Encyclopedia Titanica passenger records, as compiled for the Vanderbilt "
            "Biostatistics dataset archive. Real passengers and real outcomes."
        ),
        license="Public domain.",
        url="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        mirror_note="Data Science Dojo mirror of the canonical 891-row training split.",
        rows="891 passengers",
        notes="Genuine missingness in Age (177) and Cabin (687) - kept, not imputed away.",
    ),
    "airline_passengers": Dataset(
        id="airline_passengers",
        title="International Airline Passengers (1949-1960)",
        kind="REAL",
        origin=(
            "Box & Jenkins, 'Time Series Analysis: Forecasting and Control' (1976), "
            "series G. Monthly totals of international airline passengers."
        ),
        license="Public domain.",
        url=(
            "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
            "airline-passengers.csv"
        ),
        mirror_note="Brownlee's dataset archive.",
        rows="144 monthly observations",
        notes="Canonical multiplicative trend + seasonality benchmark.",
    ),
    "daily_min_temps": Dataset(
        id="daily_min_temps",
        title="Daily Minimum Temperatures, Melbourne (1981-1990)",
        kind="REAL",
        origin="Australian Bureau of Meteorology, via Hyndman's Time Series Data Library.",
        license="Public domain.",
        url=(
            "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
            "daily-min-temperatures.csv"
        ),
        mirror_note="Brownlee's dataset archive.",
        rows="3,650 daily observations",
        notes="Strong annual seasonality, no trend. Good ACF/PACF teaching case.",
    ),
    "sunspots": Dataset(
        id="sunspots",
        title="Sunspot Area (1875-2015)",
        kind="REAL",
        origin="Royal Observatory Greenwich / NASA Marshall solar cycle observations.",
        license="Public domain.",
        url="https://raw.githubusercontent.com/selva86/datasets/master/sunspotarea.csv",
        mirror_note="Selva Prabhakaran's dataset archive.",
        rows="141 annual observations",
        notes="~11 year solar cycle; long-period seasonality without a calendar anchor.",
    ),
    "drug_sales": Dataset(
        id="drug_sales",
        title="Antidiabetic Drug Sales, Australia (1991-2008)",
        kind="REAL",
        origin=(
            "Australian Health Insurance Commission, published in Hyndman & Athanasopoulos, "
            "'Forecasting: Principles and Practice' as series a10."
        ),
        license="Public domain (fpp text data).",
        url="https://raw.githubusercontent.com/selva86/datasets/master/a10.csv",
        mirror_note="Selva Prabhakaran's dataset archive.",
        rows="204 monthly observations",
        notes="Trend plus strong December seasonality; classic multiplicative case.",
    ),
    "supermarket_sales": Dataset(
        id="supermarket_sales",
        title="Supermarket Sales (three branches, 2019)",
        kind="REAL",
        origin=(
            "Point-of-sale records from three supermarket branches in Myanmar, "
            "published as an open retail analytics dataset."
        ),
        license="Open for educational use.",
        url="https://raw.githubusercontent.com/plotly/datasets/master/supermarket_Sales.csv",
        mirror_note="Plotly's public dataset archive.",
        rows="1,000 invoices",
        notes="Line-item invoices with customer type, payment method and rating.",
    ),
    "seattle_weather": Dataset(
        id="seattle_weather",
        title="Seattle Daily Weather (2012-2015)",
        kind="REAL",
        origin="NOAA daily observations for Seattle-Tacoma International Airport.",
        license="Public domain (US government work).",
        url="https://raw.githubusercontent.com/vega/vega-datasets/main/data/seattle-weather.csv",
        mirror_note="Vega dataset archive.",
        rows="1,461 daily observations",
        notes="Multivariate: precipitation, temp_max, temp_min, wind, weather label.",
    ),
    "bank_marketing": Dataset(
        id="bank_marketing",
        title="Bank Marketing — Portuguese Term Deposit Campaigns (2008–2010)",
        kind="REAL",
        origin=(
            "Direct marketing call records from a Portuguese retail bank, collected "
            "May 2008 – November 2010 and published by Moro, Cortez & Rita (2014). Each "
            "row is one real client contacted by phone; the target is whether they "
            "subscribed to a term deposit. Includes contemporaneous macroeconomic "
            "indicators (euribor3m, employment variation rate, consumer confidence)."
        ),
        license="Creative Commons Attribution 4.0 (UCI Machine Learning Repository).",
        url="https://raw.githubusercontent.com/selva86/datasets/master/bank-full.csv",
        mirror_note=(
            "Selva Prabhakaran's archive of the UCI bank-additional-full variant. "
            "Semicolon-delimited."
        ),
        rows="41,188 contacts, 4,640 subscriptions (11.27%)",
        notes=(
            "Carries a documented target leak. The 'duration' column records how long "
            "the call lasted, which is only known AFTER the outcome is decided — a call "
            "of zero seconds is necessarily a 'no'. UCI's own documentation warns it "
            "'should be discarded if the intention is to have a realistic predictive "
            "model'. Project 06 trains with and without it to quantify the inflation."
        ),
    ),
    "compas": Dataset(
        id="compas",
        title="COMPAS Recidivism Risk Scores (Broward County, 2013-2014)",
        kind="REAL",
        origin=(
            "Broward County, Florida criminal records joined to COMPAS risk scores, "
            "obtained by ProPublica under a public records request and published "
            "alongside their 2016 'Machine Bias' investigation. Each row is a real "
            "defendant: their COMPAS decile score, demographics, prior offences, and "
            "whether they were in fact rearrested within two years."
        ),
        license=(
            "Released publicly by ProPublica for independent scrutiny "
            "(github.com/propublica/compas-analysis)."
        ),
        url=(
            "https://raw.githubusercontent.com/propublica/compas-analysis/master/"
            "compas-scores-two-years.csv"
        ),
        mirror_note="ProPublica's own repository — this is the canonical source.",
        rows="7,214 defendants",
        notes=(
            "Data about real, named individuals in the criminal justice system. Used here "
            "strictly for the purpose ProPublica released it: auditing an algorithmic risk "
            "score for disparate impact. Names are dropped on load and never analysed. The "
            "dataset is also the centre of a genuine methodological dispute — ProPublica "
            "argued unequal error rates, Northpointe argued equal calibration — which the "
            "project treats as the substance rather than picking a side."
        ),
    ),
    "consumer_complaints": Dataset(
        id="consumer_complaints",
        title="US Consumer Financial Complaints",
        kind="REAL",
        origin=(
            "Complaints filed by US consumers with the Consumer Financial Protection "
            "Bureau against financial institutions. Each row is a real complaint with its "
            "product, issue, company and disposition."
        ),
        license="US Government work — public domain.",
        url=(
            "https://raw.githubusercontent.com/plotly/datasets/master/"
            "26k-consumer-complaints.csv"
        ),
        mirror_note="Plotly's public dataset archive.",
        rows="28,156 complaints naming 1,534 distinct companies",
        notes=(
            "Company names are entered free-form, so the same institution appears under "
            "many spellings. That makes it a genuine entity-resolution problem rather than "
            "a synthetic one."
        ),
    ),
    "aapl_daily": Dataset(
        id="aapl_daily",
        title="Apple Inc. Daily OHLCV (Feb 2015 - Feb 2017)",
        kind="REAL",
        origin=(
            "Daily open/high/low/close/volume bars for AAPL with split- and "
            "dividend-adjusted closes, as distributed in Plotly's finance chart examples."
        ),
        license="Open for educational use.",
        url=(
            "https://raw.githubusercontent.com/plotly/datasets/master/"
            "finance-charts-apple.csv"
        ),
        mirror_note="Plotly's public dataset archive.",
        rows="506 trading days",
        notes=(
            "Two years of daily bars is a short sample for a trading study, and that "
            "limitation drives the conclusion rather than being hidden from it. Always use "
            "AAPL.Adjusted rather than AAPL.Close: the unadjusted series contains a price "
            "discontinuity at every dividend that a naive model reads as a real move."
        ),
    ),
    "tiny_shakespeare": Dataset(
        id="tiny_shakespeare",
        title="Tiny Shakespeare",
        kind="REAL",
        origin=(
            "Concatenated public-domain works of Shakespeare, assembled by Andrej "
            "Karpathy as the char-rnn reference corpus."
        ),
        license="Public domain.",
        url=(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
            "tinyshakespeare/input.txt"
        ),
        mirror_note="Karpathy's char-rnn repository.",
        rows="1,115,394 characters, 65 distinct",
        notes="Small enough to train a character transformer on CPU in minutes.",
    ),
}


# --------------------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------------------


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def _save_manifest(manifest: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _download_with_retry(url: str, *, attempts: int = 5) -> bytes:
    """Download a URL, resuming after a truncated transfer.

    Large files through the sandbox's egress proxy are routinely cut off mid-stream
    (``IncompleteRead``) - the 100 MB fraud CSV and the 26 MB monthly Uber files both hit
    it. Rather than failing the pipeline, we keep the bytes already received and issue a
    ranged request for the remainder, falling back to a clean restart if the server will
    not honour ``Range``. Backoff is exponential so a genuinely unreachable host still
    fails promptly rather than hanging.
    """
    import time

    buffer = bytearray()
    last_error: Exception | None = None

    for attempt in range(attempts):
        headers = {"User-Agent": "dsx-portfolio/1.0"}
        if buffer:
            headers["Range"] = f"bytes={len(buffer)}-"
        request = urllib.request.Request(url, headers=headers)

        try:
            with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310
                # A server ignoring Range replies 200 with the whole file; restart cleanly.
                if buffer and response.status == 200:
                    buffer = bytearray()
                while True:
                    chunk = response.read(1 << 20)
                    if not chunk:
                        break
                    buffer.extend(chunk)
            return bytes(buffer)

        except Exception as error:  # noqa: BLE001 - retried below, re-raised if terminal
            last_error = error
            partial = getattr(error, "partial", None)
            if partial:
                buffer.extend(partial)
            if attempt < attempts - 1:
                delay = 2 ** attempt
                print(
                    f"  transfer interrupted at {len(buffer):,} bytes "
                    f"({type(error).__name__}); resuming in {delay}s"
                )
                time.sleep(delay)

    raise RuntimeError(
        f"Failed to download {url} after {attempts} attempts "
        f"({len(buffer):,} bytes received). Last error: {last_error!r}"
    )


def fetch(dataset_id: str, *, url_override: str | None = None,
          cache_name: str | None = None) -> Path:
    """Download a declared dataset to the cache and return its local path.

    On first fetch the SHA-256 of the payload is written to ``.data/manifest.json``. On
    every later run the cached copy is reused and re-verified against that digest, so an
    upstream file that changes underneath us produces an explicit error rather than a
    quiet shift in every downstream metric.
    """
    if dataset_id not in REGISTRY:
        raise KeyError(f"{dataset_id!r} is not a declared dataset. Add it to REGISTRY first.")

    spec = REGISTRY[dataset_id]
    url = url_override or spec.url
    if url is None:
        raise ValueError(f"{dataset_id!r} is simulated and has no URL to fetch.")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    name = cache_name or f"{dataset_id}{Path(url).suffix or '.csv'}"
    target = CACHE_DIR / name
    manifest = _load_manifest()
    key = name

    if target.exists():
        recorded = manifest.get(key, {}).get("sha256")
        if recorded:
            actual = _sha256(target.read_bytes())
            if actual != recorded:
                raise RuntimeError(
                    f"Integrity check failed for {name}: cached digest {actual[:12]} does "
                    f"not match recorded {recorded[:12]}. Delete .data/{name} to refetch."
                )
        return target

    payload = _download_with_retry(url)
    target.write_bytes(payload)
    manifest[key] = {
        "dataset_id": dataset_id,
        "url": url,
        "sha256": _sha256(payload),
        "bytes": len(payload),
    }
    _save_manifest(manifest)
    return target


def load_csv(dataset_id: str, **read_csv_kwargs) -> pd.DataFrame:
    """Fetch a declared dataset and parse it as CSV."""
    return pd.read_csv(fetch(dataset_id), **read_csv_kwargs)


def load_text(dataset_id: str) -> str:
    """Fetch a declared dataset and return it as text."""
    return fetch(dataset_id).read_text(encoding="utf-8")


def load_uber_months(months: tuple[str, ...] = ("apr", "may", "jun")) -> pd.DataFrame:
    """Load and concatenate monthly Uber TLC FOIL files.

    Parameters
    ----------
    months:
        Lowercase three-letter month prefixes, e.g. ``("apr", "may")``. The source
        archive covers apr through sep of 2014.
    """
    spec = REGISTRY["uber_nyc_2014"]
    frames = []
    for month in months:
        assert spec.url is not None
        path = fetch(
            "uber_nyc_2014",
            url_override=spec.url.format(month=month),
            cache_name=f"uber_raw_{month}14.csv",
        )
        frame = pd.read_csv(path)
        frame["source_month"] = month
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------------------
# Provenance reporting
# --------------------------------------------------------------------------------------


def provenance_record(dataset_id: str) -> dict:
    """Return the full provenance record for one dataset, including cached digest."""
    spec = REGISTRY[dataset_id]
    record = asdict(spec)
    manifest = _load_manifest()
    digests = {
        entry["sha256"]: name
        for name, entry in manifest.items()
        if entry.get("dataset_id") == dataset_id
    }
    record["cached_files"] = [
        {"file": name, "sha256": digest} for digest, name in digests.items()
    ]
    return record


def provenance_table(dataset_ids: list[str]) -> str:
    """Render a Markdown provenance table for the given datasets.

    This is the table that ships in each project README. It states plainly whether a
    source is real or simulated so a reader never has to guess.
    """
    lines = [
        "| Dataset | Kind | Rows | Origin | Licence |",
        "|---|:---:|---|---|---|",
    ]
    for dataset_id in dataset_ids:
        spec = REGISTRY[dataset_id]
        badge = "🟢 REAL" if spec.kind == "REAL" else "🟡 SIMULATED"
        origin = spec.origin.replace("\n", " ")
        lines.append(
            f"| **{spec.title}** | {badge} | {spec.rows} | {origin} | {spec.license} |"
        )
    return "\n".join(lines)
