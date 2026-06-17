-- =====================================================================
-- Query 2 / Part 2 — Borrow + Repay metrics
-- Grain / composite key : (time_bucket, asset)
-- Sources               : aave_v3_ethereum.pool_evt_borrow
--                         aave_v3_ethereum.pool_evt_repay
-- Window                : 2025-11-01 00:00 (incl) .. 2026-02-01 00:00 (excl), UTC
-- Bucket                : fixed 6-hour slots (00 / 06 / 12 / 18 UTC)
--
-- Credit-optimization notes
--   * Borrow + Repay folded into ONE query via UNION ALL -> single GROUP BY.
--   * Partition pruning on evt_block_date.
--   * last_borrow_rate = end-of-period state via max_by(value, key); the key is
--     NULL for repay rows so they are ignored, and among borrow rows it returns
--     borrowRate at the highest (block_number, evt_index) in the bucket.
--   * interestRateMode lives only on pool_evt_borrow (1 = stable, 2 = variable).
--   * Wallet fields used only inside approx_distinct().
--   * Raw values only — avg_borrow_rate etc. are DERIVED downstream, not here.
--   * asset_symbol — on-chain token symbol from tokens.erc20 (blockchain =
--     'ethereum'). Resolved via a LEFT JOIN applied AFTER aggregation, on the small
--     per-bucket result set, so the heavy event scan is unchanged; assets missing
--     from tokens.erc20 get symbol = NULL. The symbol is a readability/QA label —
--     keep joining on `asset` (address) downstream, not on the symbol.
-- =====================================================================
WITH events AS (
    SELECT
        reserve                       AS asset,
        evt_block_time,
        evt_block_number,
        evt_index,
        amount,
        "user"                        AS actor,
        interestRateMode              AS irm,
        borrowRate                    AS borrow_rate,
        'borrow'                      AS kind
    FROM aave_v3_ethereum.pool_evt_borrow
    WHERE evt_block_date >= DATE '2025-11-01'
      AND evt_block_date <  DATE '2026-02-01'

    UNION ALL

    SELECT
        reserve                       AS asset,
        evt_block_time,
        evt_block_number,
        evt_index,
        amount,
        repayer                       AS actor,
        CAST(NULL AS integer)         AS irm,
        CAST(NULL AS uint256)         AS borrow_rate,
        'repay'                       AS kind
    FROM aave_v3_ethereum.pool_evt_repay
    WHERE evt_block_date >= DATE '2025-11-01'
      AND evt_block_date <  DATE '2026-02-01'
),
agg AS (
    SELECT
        date_add('hour',
                 6 * CAST(floor(hour(evt_block_time) / 6) AS bigint),
                 date_trunc('day', evt_block_time))                            AS time_bucket,
        asset,
        -- cumulative raw flows in the bucket
        SUM(CASE WHEN kind = 'borrow' THEN amount END)                         AS borrow_amount_raw,
        SUM(CASE WHEN kind = 'repay'  THEN amount END)                         AS repay_amount_raw,
        -- signed net debt flow: borrow (+), repayment (-)
        SUM(CASE WHEN kind = 'borrow' THEN  CAST(amount AS int256)
                 WHEN kind = 'repay'  THEN -CAST(amount AS int256) END)        AS net_debt_flow_raw,
        -- activity counts
        COUNT(CASE WHEN kind = 'borrow' THEN 1 END)                            AS borrow_tx_count,
        COUNT(CASE WHEN kind = 'repay'  THEN 1 END)                            AS repay_tx_count,
        COUNT(CASE WHEN kind = 'borrow' AND irm = 1 THEN 1 END)                AS stable_borrow_tx_count,
        COUNT(CASE WHEN kind = 'borrow' AND irm = 2 THEN 1 END)                AS variable_borrow_tx_count,
        approx_distinct(CASE WHEN kind = 'borrow' THEN actor END)              AS unique_borrowers,
        approx_distinct(CASE WHEN kind = 'repay'  THEN actor END)              AS unique_repayers,
        -- end-of-period borrow rate (last borrow event in the bucket)
        max_by(borrow_rate,
               CASE WHEN kind = 'borrow' THEN ROW(evt_block_number, evt_index) END)
                                                                               AS last_borrow_rate,
        MAX(CASE WHEN kind = 'borrow' THEN evt_block_number END)               AS latest_borrow_block,
        MAX(CASE WHEN kind = 'repay'  THEN evt_block_number END)               AS latest_repay_block
    FROM events
    GROUP BY 1, 2
)
SELECT
    agg.time_bucket,
    agg.asset,
    tok.symbol                                                                AS asset_symbol,
    agg.borrow_amount_raw,
    agg.repay_amount_raw,
    agg.net_debt_flow_raw,
    agg.borrow_tx_count,
    agg.repay_tx_count,
    agg.stable_borrow_tx_count,
    agg.variable_borrow_tx_count,
    agg.unique_borrowers,
    agg.unique_repayers,
    agg.last_borrow_rate,
    agg.latest_borrow_block,
    agg.latest_repay_block
FROM agg
LEFT JOIN tokens.erc20 tok
       ON tok.blockchain = 'ethereum'
      AND tok.contract_address = agg.asset
ORDER BY 1, 2
