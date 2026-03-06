"""Equity beta positioning and exposure functions for Prescient equity funds.

This module provides four public equity analysis functions and a suite of
reusable helper functions that encapsulate shared data-retrieval and
computation logic. All functions are public and may be imported and called
independently.

Dependencies:
    pandas, numpy, sqlalchemy
    DB engines: prime_eagle, prime_equities, prime_jse, prime_msci,
                prime_compliance, prime_eav
"""

import json

import numpy as np
import pandas as pd


# =============================================================================
# Helper functions
# =============================================================================


def validate_db_engines(db_engines: dict, required_dbs: list) -> None:
    """Raise RuntimeError if any required DB engine is absent from db_engines.

    Args:
        db_engines: Mapping of engine name to SQLAlchemy engine.
        required_dbs: List of engine names that must be present.

    Raises:
        RuntimeError: One or more required DB connections are missing.

    Example:
        >>> from ppym.data.db import create_engine_multi
        >>> db_engines = create_engine_multi(['prime_eagle'], user)
        >>> validate_db_engines(db_engines, ['prime_eagle', 'prime_equities'])
        RuntimeError: Missing required DB connection(s): prime_equities. ...

    Owner:
        Romelon Chetty
    """
    missing = [db for db in required_dbs if not db_engines.get(db)]
    if missing:
        raise RuntimeError(
            f"Missing required DB connection(s): {', '.join(missing)}. "
            "Please initialise them before executing this operation."
        )


def get_fund_list(db_engines: dict) -> pd.DataFrame:
    """Return equity portfolio strategy metadata from prime_equities.

    Args:
        db_engines: Mapping of engine name to SQLAlchemy engine. Required
            key is 'prime_equities'.

    Returns:
        DataFrame containing all rows from equity_portfolio_strategy with
        columns including portfolio_code, portfolio_name, vehicle, strategy.

    Raises:
        RuntimeError: If 'prime_equities' is missing from db_engines.

    Example:
        >>> from ppym.data.db import create_engine_multi
        >>> db_engines = create_engine_multi(['prime_equities'], user)
        >>> df = get_fund_list(db_engines)
        >>> df[['portfolio_code', 'portfolio_name']].head()
          portfolio_code              portfolio_name
        0            PEQ  Prescient Core Top 40 ...

    Owner:
        Romelon Chetty
    """
    validate_db_engines(db_engines, ["prime_equities"])
    return pd.read_sql(
        "SELECT * FROM prime_equities.equity_portfolio_strategy",
        con=db_engines["prime_equities"],
    )


def get_benchmark_data(db_engines: dict) -> pd.DataFrame:
    """Return all benchmark mappings from prime_eagle.s_portfolio_bench.

    Args:
        db_engines: Mapping of engine name to SQLAlchemy engine. Required
            key is 'prime_eagle'.

    Returns:
        DataFrame with columns including portfolio_code, benchmark_code,
        benchmark_type covering all portfolio-to-benchmark relationships.

    Raises:
        RuntimeError: If 'prime_eagle' is missing from db_engines.

    Example:
        >>> from ppym.data.db import create_engine_multi
        >>> db_engines = create_engine_multi(['prime_eagle'], user)
        >>> df = get_benchmark_data(db_engines)
        >>> df[df['benchmark_type'] == 'monthly performance'].head()
          portfolio_code benchmark_code      benchmark_type
        0            PEQ        TOP40TR  monthly performance

    Owner:
        Romelon Chetty
    """
    validate_db_engines(db_engines, ["prime_eagle"])
    return pd.read_sql(
        "SELECT * FROM prime_eagle.s_portfolio_bench",
        con=db_engines["prime_eagle"],
    )


def get_instrument_metadata(db_engines: dict) -> pd.DataFrame:
    """Return instrument metadata including underlying codes, types and tags.

    Fetches all rows from prime_compliance.tmp_instruments, explodes the JSON
    tags column into flat columns, and returns a subset of the most useful
    fields.

    Args:
        db_engines: Mapping of engine name to SQLAlchemy engine. Required
            key is 'prime_compliance'.

    Returns:
        DataFrame with columns [instrument_code, underlying_instrument_code,
        security_subtype, instrument_type].

    Raises:
        RuntimeError: If 'prime_compliance' is missing from db_engines.

    Example:
        >>> from ppym.data.db import create_engine_multi
        >>> db_engines = create_engine_multi(['prime_compliance'], user)
        >>> df = get_instrument_metadata(db_engines)
        >>> df[df['instrument_type'] == 'IDXFT'].head()
          instrument_code underlying_instrument_code instrument_type
        0          ALSI40                       J200          IDXFT

    Owner:
        Romelon Chetty
    """
    validate_db_engines(db_engines, ["prime_compliance"])
    df = pd.read_sql(
        "SELECT * FROM prime_compliance.tmp_instruments",
        con=db_engines["prime_compliance"],
    )
    tags_exploded = pd.json_normalize(
        df["tags"].apply(lambda x: json.loads(x) if pd.notnull(x) else x).tolist()
    )
    extra_cols = tags_exploded.columns[
        ~tags_exploded.columns.isin(["instrument_code", "instrument_name"])
    ]
    df[extra_cols] = tags_exploded[extra_cols]
    return df[
        [
            "instrument_code",
            "underlying_instrument_code",
            "security_subtype",
            "instrument_type",
        ]
    ]


def get_holdings_with_effective_exposure(
    db_engines: dict,
    portfolio_codes: list,
    val_date: str,
    df_instrument_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Retrieve holdings and compute effective exposure for each instrument type.

    Effective exposure is calculated as follows:
        - Index futures (IDXFT): holding × price × multiplier / sum_market_value
        - TRS: holding × price / sum_market_value
        - Physical equity & ELN (EQ, EQUITY - ELN): allin_market_value / sum_market_value

    Args:
        db_engines: Mapping of engine name to SQLAlchemy engine. Required
            key is 'prime_eagle'.
        portfolio_codes: List of portfolio codes to retrieve holdings for.
        val_date: Valuation date in YYYY-MM-DD format. Required.
        df_instrument_meta: Output of get_instrument_metadata(), used to join
            underlying_instrument_code onto holdings.

    Returns:
        Full holdings DataFrame with additional columns effective_exposure,
        sum_market_value, and multiplier (NaN where not applicable).

    Raises:
        RuntimeError: If 'prime_eagle' is missing from db_engines.

    Example:
        >>> from ppym.data.db import create_engine_multi
        >>> db_engines = create_engine_multi(['prime_eagle', 'prime_compliance'], user)
        >>> df_meta = get_instrument_metadata(db_engines)
        >>> df = get_holdings_with_effective_exposure(
        ...     db_engines, ['PCEQTF'], '2025-01-09', df_meta)
        >>> df[['portfolio_code', 'instrument_code', 'effective_exposure']].head()
          portfolio_code instrument_code  effective_exposure
        0         PCEQTF            J200            0.087...

    Owner:
        Romelon Chetty
    """
    validate_db_engines(db_engines, ["prime_eagle"])

    df_holdings = pd.read_sql(
        """CALL prime_eagle.proc_get_eagle_holdings(
               %(fcode)s, %(item)s, %(sdate)s, %(edate)s)""",
        con=db_engines["prime_eagle"],
        params={
            "fcode": ",".join(portfolio_codes),
            "item": None,
            "sdate": val_date,
            "edate": val_date,
        },
        parse_dates=["datestamp"],
    )

    # Attach underlying instrument codes
    df_holdings = pd.merge(
        left=df_holdings,
        right=df_instrument_meta[["instrument_code", "underlying_instrument_code"]],
        how="left",
        on="instrument_code",
    )

    # Attach futures multipliers
    df_futures_codes = df_holdings.loc[
        df_holdings["instrument_type"] == "IDXFT", "instrument_code"
    ].unique()

    if len(df_futures_codes) > 0:
        df_deriv = pd.read_sql(
            """SELECT DISTINCT `instrument_code`, `multiplier`
               FROM tmp_eagle_derivatives
               WHERE instrument_code IN %(instrument_code)s""",
            con=db_engines["prime_eagle"],
            params={"instrument_code": tuple(df_futures_codes)},
        )
        df_holdings = pd.merge(
            left=df_holdings,
            right=df_deriv,
            on="instrument_code",
            how="left",
        )
    else:
        df_holdings["multiplier"] = np.nan

    # Compute effective exposures
    df_holdings["sum_market_value"] = df_holdings.groupby("portfolio_code")[
        "allin_market_value"
    ].transform("sum")
    df_holdings["effective_exposure"] = 0.0

    idx_ft = df_holdings["instrument_type"] == "IDXFT"
    df_holdings.loc[idx_ft, "effective_exposure"] = (
        df_holdings.loc[idx_ft, "holding"]
        * df_holdings.loc[idx_ft, "price"]
        * df_holdings.loc[idx_ft, "multiplier"]
        / df_holdings.loc[idx_ft, "sum_market_value"]
    )

    idx_trs = df_holdings["instrument_type"] == "TRS"
    df_holdings.loc[idx_trs, "effective_exposure"] = (
        df_holdings.loc[idx_trs, "holding"]
        * df_holdings.loc[idx_trs, "price"]
        / df_holdings.loc[idx_trs, "sum_market_value"]
    )

    idx_eq = df_holdings["instrument_type"].isin(["EQ", "EQUITY - ELN"])
    df_holdings.loc[idx_eq, "effective_exposure"] = (
        df_holdings.loc[idx_eq, "allin_market_value"]
        / df_holdings.loc[idx_eq, "sum_market_value"]
    )

    df_holdings["effective_exposure"] = df_holdings["effective_exposure"].fillna(0)

    return df_holdings


def get_index_constituents(
    db_engines: dict,
    list_of_indices: list,
    val_date: str,
) -> tuple:
    """Fetch JSE and MSCI index constituents and return them as a combined DataFrame.

    MSCI real_time_tickers MXWO and MXEF are resolved to their internal
    msci_index_code before fetching constituents. JSE and MSCI weights are
    normalised to decimals (MSCI weights are divided by 100).

    Args:
        db_engines: Mapping of engine name to SQLAlchemy engine. Required
            keys are 'prime_jse' and 'prime_msci'.
        list_of_indices: Index codes to retrieve. May include JSE index codes
            (e.g. 'J200') and/or MSCI real_time_tickers (e.g. 'MXWO').
        val_date: Valuation date in YYYY-MM-DD format. Required.

    Returns:
        Tuple of (df_indices_combined, df_gics_sector_names) where:
            df_indices_combined has columns [index, equity_alpha_code,
                constituent_name, weight] with weights as decimals.
            df_gics_sector_names has columns [ticker, sector] mapping each
                MSCI bb_ticker to its GICS sector name.

    Raises:
        RuntimeError: If 'prime_jse' or 'prime_msci' is missing from db_engines.

    Example:
        >>> from ppym.data.db import create_engine_multi
        >>> db_engines = create_engine_multi(['prime_jse', 'prime_msci'], user)
        >>> df_combined, df_gics = get_index_constituents(
        ...     db_engines, ['J200', 'MXWO'], '2025-01-09')
        >>> df_combined[df_combined['index'] == 'J200'].head()
          index equity_alpha_code constituent_name  weight
        0  J200               NPN          Naspers   0.212

    Owner:
        Romelon Chetty
    """
    validate_db_engines(db_engines, ["prime_jse", "prime_msci"])

    # JSE index constituents
    df_jse = pd.read_sql(
        """CALL prime_jse.proc_get_jse_index(
               %(p_indices)s, %(p_sdate)s, %(p_edate)s)""",
        con=db_engines["prime_jse"],
        params={
            "p_indices": ",".join(list_of_indices),
            "p_sdate": val_date,
            "p_edate": val_date,
        },
    )
    df_jse = df_jse[["index", "equity_alpha_code", "constituent_name", "weight"]]

    # Resolve MSCI index codes from real_time_tickers
    msci_rt_tickers = np.array(["MXWO", "MXEF"])
    df_map = pd.read_sql(
        """SELECT * FROM prime_msci.tmp_msci_indices
           WHERE real_time_ticker IN %(portfolio)s
             AND calc_date = %(start_date)s""",
        con=db_engines["prime_msci"],
        params={"portfolio": tuple(msci_rt_tickers), "start_date": val_date},
    )
    msci_index_codes = (
        df_map["msci_index_code"].astype(str).drop_duplicates().dropna().tolist()
    )

    # MSCI index constituents
    df_msci = pd.read_sql(
        """CALL prime_msci.proc_get_msci_index_constit_safe(
               %(p_indices)s, %(p_sdate)s, %(p_edate)s)""",
        con=db_engines["prime_msci"],
        params={
            "p_indices": ",".join(msci_index_codes),
            "p_sdate": val_date,
            "p_edate": val_date,
        },
    )

    # Build GICS sector name mapping
    gics_codes = df_msci["GICS_sector"].unique().tolist()
    df_gics = pd.read_sql(
        """SELECT gics_code, gics_name
           FROM s_gics_classification
           WHERE gics_code IN %(assets)s""",
        con=db_engines["prime_msci"],
        params={"assets": tuple(gics_codes)},
    ).rename(columns={"gics_code": "GICS_sector"})
    df_gics = pd.merge(
        df_gics,
        df_msci[["GICS_sector", "bb_ticker"]],
        on="GICS_sector",
        how="left",
    )
    df_gics.rename(columns={"bb_ticker": "ticker", "gics_name": "sector"}, inplace=True)
    df_gics_sector_names = df_gics[["ticker", "sector"]]

    # Prepare and combine JSE + MSCI
    df_msci = df_msci[["real_time_ticker", "bb_ticker", "security_name", "weight"]].copy()
    df_msci["weight"] = df_msci["weight"] / 100
    df_msci.rename(
        columns={
            "real_time_ticker": "index",
            "bb_ticker": "equity_alpha_code",
            "security_name": "constituent_name",
        },
        inplace=True,
    )

    df_indices_combined = pd.concat(
        [df_jse, df_msci], axis=0, ignore_index=True
    ).reset_index(drop=True)

    return df_indices_combined, df_gics_sector_names


def compute_lookthrough_exposures(
    df_holdings: pd.DataFrame,
    df_holdings_eff_exp: pd.DataFrame,
    df_indices_combined: pd.DataFrame,
) -> pd.DataFrame:
    """Expand derivative positions into underlying share-level exposures.

    For each derivative row (futures, TRS, ELN) the fund's effective exposure
    is multiplied by each index constituent's weight to produce a synthetic
    physical position. The PCGEF feeder fund is also handled by scaling
    PGPCGE share-level exposures by the CIS weight held in PCGEF.

    Args:
        df_holdings: Full holdings DataFrame returned by
            get_holdings_with_effective_exposure(). Must include columns
            portfolio_code, instrument_type, allin_market_value,
            sum_market_value.
        df_holdings_eff_exp: Subset of df_holdings filtered to equity
            instrument types (IDXFT, TRS, EQUITY - ELN, EQ) with non-zero
            effective_exposure.
        df_indices_combined: Output of get_index_constituents() with columns
            [index, equity_alpha_code, weight].

    Returns:
        DataFrame with columns [portfolio_code, instrument_code,
        effective_exposure] representing look-through share-level positions.

    Example:
        >>> df_eq = compute_lookthrough_exposures(
        ...     df_holdings, df_holdings_eff_exp, df_indices_combined)
        >>> df_eq.groupby('portfolio_code')['effective_exposure'].sum()
        portfolio_code
        PCEQTF    0.993
        PEQ       0.988

    Owner:
        Romelon Chetty
    """
    df_derivatives = df_holdings_eff_exp[
        df_holdings_eff_exp["instrument_type"] != "EQ"
    ].reset_index(drop=True)

    df_eq_rows = df_holdings_eff_exp[
        df_holdings_eff_exp["instrument_type"] == "EQ"
    ][["portfolio_code", "instrument_code", "effective_exposure"]].copy()

    for i in range(len(df_derivatives)):
        idx_code = df_derivatives.loc[i, "underlying_instrument_code"]
        idx_exp = df_derivatives.loc[i, "effective_exposure"]
        portfolio = df_derivatives.loc[i, "portfolio_code"]

        constituents = df_indices_combined.loc[
            df_indices_combined["index"] == idx_code,
            ["equity_alpha_code", "weight"],
        ].copy()
        constituents["weight"] = constituents["weight"] * idx_exp
        constituents["portfolio_code"] = portfolio
        constituents.rename(
            columns={"equity_alpha_code": "instrument_code", "weight": "effective_exposure"},
            inplace=True,
        )
        df_eq_rows = pd.concat([df_eq_rows, constituents], ignore_index=True)

    # Feeder fund: PCGEF holds PGPCGE as a CIS; scale PGPCGE exposures by
    # PCGEF's CIS weight
    if "PCGEF" in df_holdings["portfolio_code"].values:
        df_feeder = df_holdings.loc[df_holdings["portfolio_code"] == "PCGEF"].copy()
        df_feeder.loc[df_feeder["instrument_type"] == "CIS", "effective_exposure"] = (
            df_feeder["allin_market_value"] / df_feeder["sum_market_value"]
        )
        cis_series = df_feeder.loc[
            df_feeder["instrument_type"] == "CIS", "effective_exposure"
        ].reset_index(drop=True)
        if len(cis_series) > 0:
            ucits_weight = cis_series.iloc[0]
            df_feeder_fund = df_eq_rows.loc[
                df_eq_rows["portfolio_code"] == "PGPCGE"
            ].copy()
            df_feeder_fund["portfolio_code"] = "PCGEF"
            df_feeder_fund["effective_exposure"] = (
                df_feeder_fund["effective_exposure"] * ucits_weight
            )
            df_eq_rows = pd.concat([df_eq_rows, df_feeder_fund], ignore_index=True)

    return df_eq_rows


def compute_share_level_exposures(db_engines: dict, val_date: str) -> tuple:
    """Orchestrate the full share-level exposure computation for all equity funds.

    This is the core helper used by equity_beta_share_level_exposure,
    equity_active_sector_exposures, and equity_top10_exposures. It retrieves
    fund metadata, holdings, index constituents, and benchmark mappings, then
    delegates to get_holdings_with_effective_exposure(),
    get_index_constituents(), and compute_lookthrough_exposures() before
    aggregating to portfolio/share level.

    Args:
        db_engines: Mapping of engine name to SQLAlchemy engine. Required
            keys are 'prime_eagle', 'prime_equities', 'prime_jse',
            'prime_msci', and 'prime_compliance'.
        val_date: Valuation date in YYYY-MM-DD format. Required.

    Returns:
        Tuple of four objects:
            df_share_level (DataFrame): columns [portfolio_code,
                instrument_code, effective_exposure], sorted by portfolio
                ascending and effective_exposure descending.
            df_indices_combined (DataFrame): Combined JSE + MSCI constituent
                data from get_index_constituents().
            df_mdd_bench (DataFrame): Monthly-performance benchmark codes per
                fund, columns [portfolio_code, benchmark_code].
            df_gics_sector_names (DataFrame): GICS sector name per MSCI
                bb_ticker, columns [ticker, sector].

    Raises:
        RuntimeError: If any required DB connection is missing.

    Example:
        >>> from ppym.data.db import create_engine_multi
        >>> db_engines = create_engine_multi(
        ...     ['prime_eagle', 'prime_equities', 'prime_jse',
        ...      'prime_msci', 'prime_compliance'], user)
        >>> df_sl, df_idx, df_bench, df_gics = compute_share_level_exposures(
        ...     db_engines, '2025-01-09')
        >>> df_sl.head()
          portfolio_code instrument_code  effective_exposure
        0       ECICBALE             NPN            0.098707

    Owner:
        Romelon Chetty
    """
    validate_db_engines(
        db_engines,
        ["prime_eagle", "prime_equities", "prime_jse", "prime_msci", "prime_compliance"],
    )

    df_funds = get_fund_list(db_engines)
    df_bench_all = get_benchmark_data(db_engines)
    df_instrument_meta = get_instrument_metadata(db_engines)

    df_mdd_bench = df_bench_all.loc[
        df_bench_all["portfolio_code"].isin(df_funds["portfolio_code"])
        & (df_bench_all["benchmark_type"] == "monthly performance"),
        ["portfolio_code", "benchmark_code"],
    ]

    df_holdings = get_holdings_with_effective_exposure(
        db_engines,
        df_funds["portfolio_code"].tolist(),
        val_date,
        df_instrument_meta,
    )

    df_holdings_eff_exp = df_holdings[
        ["portfolio_code", "instrument_code", "instrument_type",
         "underlying_instrument_code", "effective_exposure"]
    ].loc[
        df_holdings["instrument_type"].isin(["IDXFT", "TRS", "EQUITY - ELN", "EQ"])
        & (df_holdings["effective_exposure"] != 0)
    ]

    list_of_indices = (
        df_holdings_eff_exp["underlying_instrument_code"]
        .dropna()
        .drop_duplicates()
        .tolist()
        + df_mdd_bench["benchmark_code"].str.rstrip("T").tolist()
    )

    df_indices_combined, df_gics_sector_names = get_index_constituents(
        db_engines, list_of_indices, val_date
    )

    df_eq_rows = compute_lookthrough_exposures(
        df_holdings, df_holdings_eff_exp, df_indices_combined
    )

    df_share_level = (
        df_eq_rows.groupby(["portfolio_code", "instrument_code"])["effective_exposure"]
        .sum()
        .reset_index()
        .sort_values(
            by=["portfolio_code", "effective_exposure"], ascending=[True, False]
        )
        .reset_index(drop=True)
    )

    return df_share_level, df_indices_combined, df_mdd_bench, df_gics_sector_names


# =============================================================================
# Public analysis functions
# =============================================================================


def equity_cover_positioning(db_engines: dict, val_date: str) -> dict:
    """Return equity cover positioning data grouped by strategy for the Cover tab.

    Cover ratio is defined as physical equity / total equity — i.e. the
    proportion of each fund's equity exposure that is held physically rather
    than synthetically through derivatives. A ratio of 1.0 means the entire
    equity exposure is in physical shares; a ratio of 0.0 means it is fully
    synthetic.

    Each strategy group (CISCA PA, REG28 PA, Global) gets its own column
    layout because the relevant metrics differ by regulatory context:

    * CISCA PA  — unit trusts.  Shows the full derivative breakdown (futures,
                  TRS, ELN) alongside the cover ratio and un-equitised cash.
    * REG28 PA  — pension funds.  Adds an equity headroom column (75% Reg28
                  limit minus actual total equity) so exposure-to-limit
                  utilisation is immediately visible.
    * Global    — offshore/MSCI-benchmarked funds.  ELN column is omitted
                  (not used in offshore mandates); otherwise mirrors CISCA PA.

    Args:
        db_engines: A dictionary of SQLAlchemy engines keyed by name. Required
            keys are 'prime_eagle' and 'prime_equities'. Use
            ``ppym.data.db.create_engine_multi()`` to create it.
        val_date: Valuation date in YYYY-MM-DD format. Required.

    Returns:
        Dict with two keys:

        * ``'data'`` — dict keyed by strategy name.  Each value is a list of
          row dicts ready for DataTables.  Common fields per row:
          portfolio_code, portfolio_name, vehicle, fund_size, total_equity,
          physical, derivatives, futures, trs, notes, Un-equitised,
          cover_ratio, strategy, datestamp (ISO-8601 string).
          REG28 PA rows also include equity_headroom.

        * ``'columns'`` — dict keyed by strategy name.  Each value is a list
          of DataTables column-definition dicts ``{title, data}`` in column
          order, matching the fields present in the corresponding data rows.

    Raises:
        RuntimeError: If 'prime_eagle' or 'prime_equities' is missing from
            db_engines.

    Example:
        >>> import ppym.data.db as pimdb
        >>> db_engines = pimdb.create_engine_multi(
        ...     ['prime_eagle', 'prime_equities'], user)
        >>> result = equity_cover_positioning(db_engines, '2025-01-09')
        >>> result['data']['CISCA PA'][0]
        {'portfolio_code': 'PCEQTF', 'total_equity': 0.993,
         'physical': 0.157, 'cover_ratio': 0.158, ...}
        >>> [c['title'] for c in result['columns']['REG28 PA']]
        ['Fund Code', 'Fund Name', ..., 'Headroom (75%)', ..., 'Cover']

    Owner:
        Romelon Chetty
    """
    validate_db_engines(db_engines, ["prime_eagle", "prime_equities"])

    df = equity_beta_positioning(db_engines, val_date)

    # Cover ratio: fraction of equity exposure held as physical shares.
    # Clipped to [0, 1] to guard against edge cases (e.g. negative physical).
    df["cover_ratio"] = (
        (df["physical"] / df["total_equity"])
        .fillna(0)
        .clip(0, 1)
        .round(3)
    )

    # REG28 equity limit is 75%; headroom = room remaining before the cap.
    df["equity_headroom"] = (0.75 - df["total_equity"]).round(3)

    # Datestamp → ISO string so JSON serialisation is unambiguous.
    if pd.api.types.is_datetime64_any_dtype(df["datestamp"]):
        df["datestamp"] = df["datestamp"].dt.strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # Column definitions per strategy (DataTables {title, data} format)
    # ------------------------------------------------------------------
    _COMMON_HEAD = [
        {"title": "Fund Code",     "data": "portfolio_code"},
        {"title": "Fund Name",     "data": "portfolio_name"},
        {"title": "Vehicle",       "data": "vehicle"},
        {"title": "Fund Size (Rm)","data": "fund_size"},
        {"title": "Total Equity",  "data": "total_equity"},
        {"title": "Physical",      "data": "physical"},
        {"title": "Derivatives",   "data": "derivatives"},
        {"title": "Futures",       "data": "futures"},
        {"title": "TRS",           "data": "trs"},
    ]
    _COMMON_TAIL = [
        {"title": "Un-Equitised",  "data": "Un-equitised"},
        {"title": "Cover",         "data": "cover_ratio"},
        {"title": "Strategy",      "data": "strategy"},     # hidden
        {"title": "Date",          "data": "datestamp"},    # hidden
    ]

    columns_by_strategy = {
        # CISCA PA: includes ELN / notes column
        "CISCA PA": (
            _COMMON_HEAD
            + [{"title": "ELN / Notes", "data": "notes"}]
            + _COMMON_TAIL
        ),
        # REG28 PA: replaces ELN with equity headroom (75% limit)
        "REG28 PA": (
            _COMMON_HEAD
            + [
                {"title": "ELN / Notes",   "data": "notes"},
                {"title": "Headroom (75%)", "data": "equity_headroom"},
            ]
            + _COMMON_TAIL
        ),
        # Global: offshore mandates do not use ELN; omit notes column
        "Global": _COMMON_HEAD + _COMMON_TAIL,
    }

    # ------------------------------------------------------------------
    # Fields to include in each strategy's row dicts
    # ------------------------------------------------------------------
    _COMMON_FIELDS = [
        "portfolio_code", "portfolio_name", "vehicle", "fund_size",
        "total_equity", "physical", "derivatives", "futures", "trs",
        "Un-equitised", "cover_ratio", "strategy", "datestamp",
    ]

    fields_by_strategy = {
        "CISCA PA": _COMMON_FIELDS[:9] + ["notes"] + _COMMON_FIELDS[9:],
        "REG28 PA": _COMMON_FIELDS[:9] + ["notes", "equity_headroom"] + _COMMON_FIELDS[9:],
        "Global":   _COMMON_FIELDS,
    }

    data_out = {}
    columns_out = {}

    for strategy, fields in fields_by_strategy.items():
        subset = df[df["strategy"] == strategy][fields].copy()
        data_out[strategy]    = subset.to_dict(orient="records")
        columns_out[strategy] = columns_by_strategy[strategy]

    return {"data": data_out, "columns": columns_out}


def equity_beta_positioning(db_engines: dict, val_date: str) -> pd.DataFrame:
    """Return equity beta exposures split by instrument type for each fund.

    Aggregates equity exposures across all Prescient equity portfolios for a
    given valuation date. Instrument types reported are Total Return Swaps
    (TRS), Futures (price futures + total-return futures), Equity Linked Notes
    (ELN / Notes), and Physical Equity. A combined derivatives column and an
    un-equitised residual column are also included.

    Args:
        db_engines: A dictionary of SQLAlchemy engines keyed by name. Required
            keys are 'prime_eagle' and 'prime_equities'. Use
            ``ppym.data.db.create_engine_multi()`` to create it.
        val_date: Valuation date in YYYY-MM-DD format. Required.

    Returns:
        DataFrame with one row per fund and columns portfolio_code,
        portfolio_name, vehicle, fund_size, total_equity, derivatives,
        futures, trs, notes, physical, Un-equitised, strategy, datestamp.
        All exposure columns are rounded to three decimal places.

    Raises:
        RuntimeError: If 'prime_eagle' or 'prime_equities' is missing from
            db_engines.

    Example:
        >>> import ppym.data.db as pimdb
        >>> db_engines = pimdb.create_engine_multi(
        ...     ['prime_eagle', 'prime_equities'], user)
        >>> df = equity_beta_positioning(db_engines, '2025-01-09')
        >>> df[['portfolio_code', 'total_equity', 'futures', 'notes', 'physical']]
          portfolio_code  total_equity  futures  notes  physical
        0            PEQ         0.988    0.214  0.774     0.000
        1         PCEQTF         0.993    0.087  0.749     0.157

    Owner:
        Romelon Chetty
    """
    validate_db_engines(db_engines, ["prime_eagle", "prime_equities"])

    df_base = pd.read_sql(
        "CALL prime_eagle.proc_get_equity_effective_exposure(%(edate)s)",
        con=db_engines["prime_eagle"],
        params={"edate": val_date},
        parse_dates=["datestamp"],
    )
    df_funds = get_fund_list(db_engines)

    df_base["futures"] = df_base["price_futures"] + df_base["total_return_futures"]
    df_base["derivatives"] = df_base["futures"] + df_base["trs"] + df_base["notes"]
    df_base["Un-equitised"] = 1 - df_base["total_equity"]
    df_base.drop(["strategy", "portfolio_name"], axis=1, inplace=True)

    df_base = pd.merge(left=df_base, right=df_funds, how="right", on="portfolio_code")

    return df_base[
        [
            "portfolio_code", "portfolio_name", "vehicle", "fund_size",
            "total_equity", "derivatives", "futures", "trs", "notes",
            "physical", "Un-equitised", "strategy", "datestamp",
        ]
    ].round(3)


def equity_beta_share_level_exposure(db_engines: dict, val_date: str) -> pd.DataFrame:
    """Return look-through share-level effective exposures for each equity fund.

    Computes effective exposures for all equity instrument types (physical
    equity, index futures, TRS, ELN) and expands derivatives into their
    underlying index constituents so that the result represents a pure
    share-level view. The PCGEF feeder fund is handled by scaling PGPCGE
    exposures by the CIS weight held in PCGEF.

    Args:
        db_engines: A dictionary of SQLAlchemy engines keyed by name. Required
            keys are 'prime_eagle', 'prime_equities', 'prime_jse', 'prime_msci',
            and 'prime_compliance'. Use ``ppym.data.db.create_engine_multi()``
            to create it.
        val_date: Valuation date in YYYY-MM-DD format. Required.

    Returns:
        DataFrame with columns [portfolio_code, instrument_code,
        effective_exposure], sorted by portfolio_code ascending and
        effective_exposure descending.

    Raises:
        RuntimeError: If any required DB connection is missing from db_engines.

    Example:
        >>> import ppym.data.db as pimdb
        >>> db_engines = pimdb.create_engine_multi(
        ...     ['prime_eagle', 'prime_equities', 'prime_jse',
        ...      'prime_msci', 'prime_compliance'], user)
        >>> df = equity_beta_share_level_exposure(db_engines, '2025-01-09')
        >>> df.head()
          portfolio_code instrument_code  effective_exposure
        0       ECICBALE             NPN            0.098707
        1       ECICBALE             FSR            0.060474
        2       ECICBALE             SBK            0.046863

    Owner:
        Romelon Chetty
    """
    validate_db_engines(
        db_engines,
        ["prime_eagle", "prime_equities", "prime_jse", "prime_msci", "prime_compliance"],
    )

    df_share_level, _, _, _ = compute_share_level_exposures(db_engines, val_date)

    return df_share_level


def equity_active_sector_exposures(db_engines: dict, val_date: str) -> pd.DataFrame:
    """Return fund vs benchmark sector weights and the active difference for each fund.

    Computes look-through share-level exposures, maps each share to its ICB
    industry sector (JSE shares) or GICS sector (MSCI shares), then aggregates
    to portfolio/sector level. Benchmark constituent weights are fetched from
    the relevant JSE or MSCI index and summed at sector level so that active
    (fund minus benchmark) exposures can be computed.

    Args:
        db_engines: A dictionary of SQLAlchemy engines keyed by name. Required
            keys are 'prime_eagle', 'prime_equities', 'prime_jse', 'prime_msci',
            'prime_compliance', and 'prime_eav'. Use
            ``ppym.data.db.create_engine_multi()`` to create it.
        val_date: Valuation date in YYYY-MM-DD format. Required.

    Returns:
        DataFrame with columns [portfolio_code, sector, fund, benchmark,
        active] where fund and benchmark are decimal weights (e.g. 0.25 = 25%)
        and active = fund - benchmark.

    Raises:
        RuntimeError: If any required DB connection is missing from db_engines.

    Example:
        >>> import ppym.data.db as pimdb
        >>> db_engines = pimdb.create_engine_multi(
        ...     ['prime_eagle', 'prime_equities', 'prime_jse',
        ...      'prime_msci', 'prime_compliance', 'prime_eav'], user)
        >>> df = equity_active_sector_exposures(db_engines, '2025-01-09')
        >>> df[df['portfolio_code'] == 'PCEQTF'].head()
          portfolio_code           sector   fund  benchmark  active
        0         PCEQTF       Financials  0.312      0.285   0.027
        1         PCEQTF  Basic Materials  0.218      0.201   0.017

    Owner:
        Romelon Chetty
    """
    validate_db_engines(
        db_engines,
        [
            "prime_eagle", "prime_equities", "prime_jse", "prime_msci",
            "prime_compliance", "prime_eav",
        ],
    )

    df_share_level, df_indices_combined, df_mdd_bench, df_gics_sector_names = (
        compute_share_level_exposures(db_engines, val_date)
    )

    # Build ICB sector classification for JSE shares via EAV proc_describe_entity
    jse_codes = df_share_level.loc[
        ~df_share_level["portfolio_code"].isin(["PGPCGE", "PGPCEM", "PCGEF"]),
        "instrument_code",
    ].unique()
    jse_codes_eav = pd.Series(jse_codes) + " SJ Equity"

    df_icb_raw = pd.read_sql(
        "CALL prime_eav.proc_describe_entity(%(p_entities)s)",
        con=db_engines["prime_eav"],
        params={"p_entities": ",".join(jse_codes_eav)},
    )
    df_icb = df_icb_raw["json"].apply(lambda x: pd.Series(json.loads(x)))
    df_icb["TICKER_AND_EXCH_CODE"] = (
        df_icb["TICKER_AND_EXCH_CODE"].str.split(" ").str[0]
    )
    df_icb.rename(
        columns={"TICKER_AND_EXCH_CODE": "ticker", "ICB_INDUSTRY_NAME": "sector"},
        inplace=True,
    )

    # Combine JSE (ICB) and MSCI (GICS) sector mappings
    df_sectors_all = pd.concat(
        [df_icb[["ticker", "sector"]], df_gics_sector_names],
        axis=0,
        ignore_index=True,
    ).reset_index(drop=True)

    # Merge sector classifications onto share-level exposures
    df_with_sector = pd.merge(
        left=df_share_level,
        right=df_sectors_all,
        left_on="instrument_code",
        right_on="ticker",
        how="left",
    )

    # Attach benchmark code per portfolio (strip trailing 'T' for JSE indices)
    df_bench_codes = df_mdd_bench.copy()
    df_bench_codes["benchmark_code"] = df_bench_codes["benchmark_code"].str.rstrip("T")

    df_with_sector = pd.merge(
        df_with_sector,
        df_bench_codes[["portfolio_code", "benchmark_code"]],
        on="portfolio_code",
        how="left",
    )

    # Attach benchmark constituent weights per share
    df_with_sector = pd.merge(
        df_with_sector,
        df_indices_combined[["index", "equity_alpha_code", "weight"]],
        left_on=["benchmark_code", "instrument_code"],
        right_on=["index", "equity_alpha_code"],
        how="left",
    )

    # Aggregate to portfolio/sector level
    df_active_sector = (
        df_with_sector.groupby(["portfolio_code", "sector"])[
            ["effective_exposure", "weight"]
        ]
        .sum()
        .reset_index()
    )
    df_active_sector.rename(
        columns={"effective_exposure": "fund", "weight": "benchmark"}, inplace=True
    )
    df_active_sector["active"] = (
        df_active_sector["fund"] - df_active_sector["benchmark"]
    )

    return df_active_sector


def equity_top10_exposures(db_engines: dict, val_date: str) -> pd.DataFrame:
    """Return the top 10 share-level holdings by effective exposure for each fund.

    Computes look-through share-level exposures via compute_share_level_exposures()
    and enriches the top-10 rows per fund with constituent names from the
    relevant index data and a rank column (1 = largest holding).

    Args:
        db_engines: A dictionary of SQLAlchemy engines keyed by name. Required
            keys are 'prime_eagle', 'prime_equities', 'prime_jse', 'prime_msci',
            and 'prime_compliance'. Use ``ppym.data.db.create_engine_multi()``
            to create it.
        val_date: Valuation date in YYYY-MM-DD format. Required.

    Returns:
        DataFrame with columns [portfolio_code, instrument_code,
        constituent_name, effective_exposure, rank], one row per top-10
        holding per fund, sorted by portfolio_code ascending and
        effective_exposure descending within each fund.

    Raises:
        RuntimeError: If any required DB connection is missing from db_engines.

    Example:
        >>> import ppym.data.db as pimdb
        >>> db_engines = pimdb.create_engine_multi(
        ...     ['prime_eagle', 'prime_equities', 'prime_jse',
        ...      'prime_msci', 'prime_compliance'], user)
        >>> df = equity_top10_exposures(db_engines, '2025-01-09')
        >>> df[df['portfolio_code'] == 'PCEQTF']
          portfolio_code instrument_code constituent_name  effective_exposure  rank
        0         PCEQTF             NPN          Naspers            0.098707     1
        1         PCEQTF             FSR   Firstrand Ltd.            0.060474     2

    Owner:
        Romelon Chetty
    """
    validate_db_engines(
        db_engines,
        ["prime_eagle", "prime_equities", "prime_jse", "prime_msci", "prime_compliance"],
    )

    df_share_level, df_indices_combined, _, _ = compute_share_level_exposures(
        db_engines, val_date
    )

    # Take top 10 per fund (data is already sorted descending by effective_exposure)
    df_top10 = (
        df_share_level.groupby("portfolio_code", group_keys=False)
        .head(10)
        .reset_index(drop=True)
    )

    # Enrich with constituent names
    df_names = (
        df_indices_combined[["equity_alpha_code", "constituent_name"]]
        .drop_duplicates("equity_alpha_code")
    )
    df_top10 = pd.merge(
        df_top10,
        df_names,
        left_on="instrument_code",
        right_on="equity_alpha_code",
        how="left",
    )
    df_top10.drop(columns=["equity_alpha_code"], inplace=True)

    # Add rank within each portfolio
    df_top10 = df_top10.sort_values(
        by=["portfolio_code", "effective_exposure"], ascending=[True, False]
    ).reset_index(drop=True)
    df_top10["rank"] = df_top10.groupby("portfolio_code").cumcount() + 1

    df_top10["constituent_name"] = df_top10["constituent_name"].str.title()

    return df_top10[
        ["portfolio_code", "instrument_code", "constituent_name",
         "effective_exposure", "rank"]
    ]
