"""Equity beta positioning and exposure functions for Prescient equity funds.

This module provides four public functions for analysing equity beta exposures
across Prescient equity funds, along with private helper functions that
encapsulate shared data-retrieval and computation logic.

Dependencies:
    pandas, numpy, sqlalchemy
    DB engines: prime_eagle, prime_equities, prime_jse, prime_msci,
                prime_compliance, prime_eav
"""

import json

import numpy as np
import pandas as pd


# =============================================================================
# Private helpers
# =============================================================================


def _validate_db_engines(db_engines: dict, required_dbs: list) -> None:
    """Raise RuntimeError if any required DB engine is absent from db_engines.

    Args:
        db_engines: Mapping of engine name to SQLAlchemy engine.
        required_dbs: List of engine names that must be present.

    Raises:
        RuntimeError: One or more required DB connections are missing.
    """
    missing = [db for db in required_dbs if not db_engines.get(db)]
    if missing:
        raise RuntimeError(
            f"Missing required DB connection(s): {', '.join(missing)}. "
            "Please initialise them before executing this operation."
        )


def _get_fund_list(db_engines: dict) -> pd.DataFrame:
    """Return equity portfolio strategy metadata from prime_equities.

    Args:
        db_engines: Mapping of engine name to SQLAlchemy engine.

    Returns:
        DataFrame containing all rows from equity_portfolio_strategy.
    """
    return pd.read_sql(
        "SELECT * FROM prime_equities.equity_portfolio_strategy",
        con=db_engines["prime_equities"],
    )


def _get_benchmark_data(db_engines: dict) -> pd.DataFrame:
    """Return all benchmark mappings from prime_eagle.s_portfolio_bench.

    Args:
        db_engines: Mapping of engine name to SQLAlchemy engine.

    Returns:
        DataFrame with portfolio-to-benchmark relationships.
    """
    return pd.read_sql(
        "SELECT * FROM prime_eagle.s_portfolio_bench",
        con=db_engines["prime_eagle"],
    )


def _get_instrument_metadata(db_engines: dict) -> pd.DataFrame:
    """Return instrument metadata including underlying codes, types and tags.

    Args:
        db_engines: Mapping of engine name to SQLAlchemy engine.

    Returns:
        DataFrame with columns instrument_code, underlying_instrument_code,
        security_subtype, instrument_type.
    """
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


def _get_holdings_with_effective_exposure(
    db_engines: dict,
    portfolio_codes: list,
    val_date: str,
    df_instrument_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Retrieve holdings and attach an effective_exposure column per instrument type.

    Effective exposure is computed as follows:
        - Index futures (IDXFT): holding × price × multiplier / sum_market_value
        - TRS: holding × price / sum_market_value
        - Physical equity & ELN (EQ, EQUITY - ELN): allin_market_value / sum_market_value

    Args:
        db_engines: Mapping of engine name to SQLAlchemy engine.
        portfolio_codes: List of portfolio codes to retrieve holdings for.
        val_date: Valuation date in YYYY-MM-DD format.
        df_instrument_meta: Output of _get_instrument_metadata(), used to join
            underlying_instrument_code and instrument_type onto holdings.

    Returns:
        Holdings DataFrame with effective_exposure, sum_market_value, and
        multiplier columns appended.
    """
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


def _get_index_constituents(
    db_engines: dict,
    list_of_indices: list,
    val_date: str,
) -> tuple:
    """Fetch JSE and MSCI index constituents and return them as a combined DataFrame.

    Args:
        db_engines: Mapping of engine name to SQLAlchemy engine.
        list_of_indices: Index codes to retrieve (JSE and/or MSCI real_time_tickers).
        val_date: Valuation date in YYYY-MM-DD format.

    Returns:
        Tuple of (df_indices_combined, df_gics_sector_names) where:
            df_indices_combined: columns [index, equity_alpha_code,
                constituent_name, weight] with weights as decimals.
            df_gics_sector_names: columns [ticker, sector] mapping MSCI bb_ticker
                to GICS sector name.
    """
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

    # Prepare and combine
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


def _compute_lookthrough_exposures(
    df_holdings: pd.DataFrame,
    df_holdings_eff_exp: pd.DataFrame,
    df_indices_combined: pd.DataFrame,
) -> pd.DataFrame:
    """Expand derivative positions into underlying share-level exposures.

    For each derivative row (futures, TRS, ELN), multiplies the fund's
    effective exposure by each constituent's index weight to produce a
    synthetic physical position. Also handles the PCGEF feeder fund by
    scaling PGPCGE exposures by the CIS weight in PCGEF.

    Args:
        df_holdings: Full holdings DataFrame returned by
            _get_holdings_with_effective_exposure().
        df_holdings_eff_exp: Subset of df_holdings filtered to equity
            instrument types with non-zero effective_exposure.
        df_indices_combined: Combined index constituent data from
            _get_index_constituents().

    Returns:
        DataFrame with columns [portfolio_code, instrument_code,
        effective_exposure] representing look-through share-level positions.
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


def _compute_share_level_exposures(
    db_engines: dict,
    val_date: str,
) -> tuple:
    """Core computation shared by equity_beta_share_level_exposure, equity_active_sector_exposures and equity_top10_exposures.

    Retrieves holdings, computes effective exposures, expands derivatives into
    their underlying index constituents, and aggregates to portfolio/share level.

    Args:
        db_engines: Mapping of engine name to SQLAlchemy engine. Requires
            prime_eagle, prime_equities, prime_jse, prime_msci,
            prime_compliance.
        val_date: Valuation date in YYYY-MM-DD format.

    Returns:
        Tuple of four DataFrames:
            df_share_level: columns [portfolio_code, instrument_code,
                effective_exposure], sorted by portfolio and descending exposure.
            df_indices_combined: Combined JSE + MSCI constituent data.
            df_mdd_bench: Benchmark codes for each fund (monthly performance).
            df_gics_sector_names: GICS sector name per MSCI bb_ticker.
    """
    df_funds = _get_fund_list(db_engines)
    df_bench_all = _get_benchmark_data(db_engines)
    df_instrument_meta = _get_instrument_metadata(db_engines)

    df_mdd_bench = df_bench_all.loc[
        df_bench_all["portfolio_code"].isin(df_funds["portfolio_code"])
        & (df_bench_all["benchmark_type"] == "monthly performance"),
        ["portfolio_code", "benchmark_code"],
    ]

    df_holdings = _get_holdings_with_effective_exposure(
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

    df_indices_combined, df_gics_sector_names = _get_index_constituents(
        db_engines, list_of_indices, val_date
    )

    df_eq_rows = _compute_lookthrough_exposures(
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
# Public functions
# =============================================================================


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
        DataFrame with one row per fund and columns:
        portfolio_code, portfolio_name, vehicle, fund_size, total_equity,
        derivatives, futures, trs, notes, physical, Un-equitised, strategy,
        datestamp. All exposure columns are rounded to three decimal places.

    Raises:
        RuntimeError: One or more required DB connections are missing.
        ValueError: If val_date is not in YYYY-MM-DD format.

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
    _validate_db_engines(db_engines, ["prime_eagle", "prime_equities"])

    df_base = pd.read_sql(
        "CALL prime_eagle.proc_get_equity_effective_exposure(%(edate)s)",
        con=db_engines["prime_eagle"],
        params={"edate": val_date},
        parse_dates=["datestamp"],
    )
    df_funds = _get_fund_list(db_engines)

    df_base["futures"] = df_base["price_futures"] + df_base["total_return_futures"]
    df_base["derivatives"] = df_base["futures"] + df_base["trs"] + df_base["notes"]
    df_base["Un-equitised"] = 1 - df_base["total_equity"]
    df_base.drop(["strategy", "portfolio_name"], axis=1, inplace=True)

    df_base = pd.merge(left=df_base, right=df_funds, how="right", on="portfolio_code")

    df_beta_positioning = df_base[
        [
            "portfolio_code", "portfolio_name", "vehicle", "fund_size",
            "total_equity", "derivatives", "futures", "trs", "notes",
            "physical", "Un-equitised", "strategy", "datestamp",
        ]
    ].round(3)

    return df_beta_positioning


def equity_beta_share_level_exposure(db_engines: dict, val_date: str) -> pd.DataFrame:
    """Return look-through share-level effective exposures for each equity fund.

    Computes effective exposures for all equity instrument types (physical
    equity, index futures, TRS, ELN) and expands derivatives into their
    underlying index constituents so that the result represents a pure
    share-level view. The PCGEF feeder fund is handled by scaling the PGPCGE
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
        RuntimeError: One or more required DB connections are missing.
        ValueError: If val_date is not in YYYY-MM-DD format.

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
    _validate_db_engines(
        db_engines,
        ["prime_eagle", "prime_equities", "prime_jse", "prime_msci", "prime_compliance"],
    )

    df_share_level, _, _, _ = _compute_share_level_exposures(db_engines, val_date)

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
        RuntimeError: One or more required DB connections are missing.
        ValueError: If val_date is not in YYYY-MM-DD format.

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
    _validate_db_engines(
        db_engines,
        [
            "prime_eagle", "prime_equities", "prime_jse", "prime_msci",
            "prime_compliance", "prime_eav",
        ],
    )

    df_share_level, df_indices_combined, df_mdd_bench, df_gics_sector_names = (
        _compute_share_level_exposures(db_engines, val_date)
    )

    # Build sector classification for JSE shares via EAV proc_describe_entity
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

    Derives results from equity_beta_share_level_exposure and enriches the
    output with constituent names from the relevant index data and a rank
    column (1 = largest holding).

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
        RuntimeError: One or more required DB connections are missing.
        ValueError: If val_date is not in YYYY-MM-DD format.

    Example:
        >>> import ppym.data.db as pimdb
        >>> db_engines = pimdb.create_engine_multi(
        ...     ['prime_eagle', 'prime_equities', 'prime_jse',
        ...      'prime_msci', 'prime_compliance'], user)
        >>> df = equity_top10_exposures(db_engines, '2025-01-09')
        >>> df[df['portfolio_code'] == 'PCEQTF']
          portfolio_code instrument_code constituent_name  effective_exposure  rank
        0         PCEQTF             NPN          Naspers             0.0987     1
        1         PCEQTF             FSR   FirstRand Ltd.             0.0605     2

    Owner:
        Romelon Chetty
    """
    _validate_db_engines(
        db_engines,
        ["prime_eagle", "prime_equities", "prime_jse", "prime_msci", "prime_compliance"],
    )

    df_share_level, df_indices_combined, _, _ = _compute_share_level_exposures(
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
