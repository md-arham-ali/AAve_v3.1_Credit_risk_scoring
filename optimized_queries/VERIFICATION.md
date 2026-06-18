# Metric ↔ Table ↔ Field Verification (Dune catalog)

Verified against the live Dune catalog (`aave_v3_ethereum.*`, chain = Ethereum, all
tables `public`) via the Dune MCP `searchTables` schema inspection. No table data was
scanned for this verification — schema/metadata only.

Legend: ✅ confirmed exactly as in `Preferred_metrics.md` · ✏️ confirmed with a
correction/clarification · ⤵ intentionally computed DOWNSTREAM (not in SQL) · 🚫 excluded
by `Dune_Context.md` rules (wallets / hashes / addresses / text).

---

## 1. Source-table field verification (all ✅ unless noted)

| Table | Fields used | Status |
|-------|-------------|--------|
| `pool_evt_supply` | `amount`, `reserve`, `user`, `evt_block_*` | ✅ |
| `pool_evt_withdraw` | `amount`, `reserve`, `user`, `to`, `evt_block_*` | ✅ |
| `pool_evt_borrow` | `amount`, `borrowRate`, `interestRateMode`, `reserve`, `user` | ✅ (`interestRateMode` is `integer`) |
| `pool_evt_repay` | `amount`, `repayer`, `reserve`, `useATokens`, `user` | ✅ |
| `pool_evt_reservedataupdated` | `liquidityRate`, `liquidityIndex`, `stableBorrowRate`, `variableBorrowRate`, `variableBorrowIndex`, `reserve` | ✅ |
| `poolconfigurator_evt_supplycapchanged` | `asset`, `newSupplyCap`, `oldSupplyCap` | ✅ |
| `poolconfigurator_evt_borrowcapchanged` | `asset`, `newBorrowCap`, `oldBorrowCap` | ✅ |
| `poolconfigurator_evt_debtceilingchanged` | `asset`, `newDebtCeiling`, `oldDebtCeiling` | ✅ |
| `poolconfigurator_evt_reservefactorchanged` | `asset`, `newReserveFactor`, `oldReserveFactor` | ✅ |
| `poolconfigurator_evt_collateralconfigurationchanged` | `asset`, `liquidationThreshold`, `liquidationBonus`, `ltv` | ✅ |
| `pool_evt_liquidationcall` | `collateralAsset`, `debtAsset`, `debtToCover`, `liquidatedCollateralAmount`, `user`, `liquidator`, `receiveAToken` | ✏️ see note A |
| `pool_evt_flashloan` | `asset`, `amount`, `premium`, `initiator`, `interestRateMode`, `target` | ✏️ see note B |
| `aaveoracle_call_getassetprice` | `asset` (input), `output_0`, `call_block_*` | ✏️ see note C |
| `pool_call_getuseraccountdata` | `output_totalCollateralBase`, `output_totalDebtBase`, `output_availableBorrowsBase`, `output_currentLiquidationThreshold`, `output_ltv`, `output_healthFactor`, `user` | ✏️ see note D |
| `pool_evt_reserveusedascollateralenabled` / `...disabled` | `reserve`, `user` | ✅ (added: extra query 09) |
| `tokens.erc20` | `blockchain`, `contract_address`, `decimals`, `symbol` | ✅ (decimal + symbol source) |

### Correction / clarification notes

- **A — liquidation collateral field lives only on the EVENT table.**
  `liquidatedCollateralAmount` exists on `pool_evt_liquidationcall` (event) but **not** on
  `pool_call_liquidationcall` (call). Query 05 correctly uses the **event** table. ✅
- **B — flashloan `interestRateMode` has a 3rd value.** Type is `integer`; values are
  `0` = no open debt (loan repaid in-tx — the dominant case), `1` = stable, `2` = variable.
  Query 06 exposes all three counts (Plan draft only listed stable/variable).
  `flashloan_target` → 🚫 excluded (it is an address).
- **C — oracle input column name is `asset`.** `Preferred_metrics.md` maps
  `oracle_asset_price → output_0` ✅. The token key (Plan called it "input parameter") is the
  column literally named **`asset`**. There is also `aaveoracle_call_getassetsprices` (plural,
  array I/O) — **not** used.
- **D — user-account wallet column is `user`, not `input_user`.** `Plan.md` Part 6 referred to
  `input_user`; the actual decoded column is **`user`**. Query 08 uses `user` (inside
  `approx_distinct` only). All six `output_*` fields confirmed present.

---

## 2. Decimal corrections (Part 4 reserve config)

`Plan.md`'s Part-4 decimal draft labelled the caps/ceilings as "token decimals". On-chain
Aave V3 encoding differs — `decimal_reference_part4.sql` uses the corrected values:

| Metric | Plan draft | Corrected | Why |
|--------|-----------|-----------|-----|
| `supply_cap`, `old_supply_cap`, `borrow_cap`, `old_borrow_cap` | token decimals | **0 (whole tokens)** | Aave V3 stores caps as whole-token integer counts, not wei. |
| `debt_ceiling`, `old_debt_ceiling` | token decimals | **2** | Aave V3 `DEBT_CEILING_DECIMALS = 2`. |
| `reserve_factor`, `liquidation_threshold`, `liquidation_bonus`, `ltv` (+ `old_*`) | 4 (bps) | **4 (bps)** | ✅ unchanged. |

---

## 3. Metric coverage map (every metric in `Preferred_metrics.md`)

| Metric | Where |
|--------|-------|
| supply_amount | `01` supply_amount_raw |
| total_supply, total_liquidity, net_flow* | ⤵ derived (raw `net_supply_flow_raw` in `01`) |
| liquidity_rate | `03` |
| avg_liquidity_rate | ⤵ |
| supply_cap, old_supply_cap | `04` |
| liquidity_index | `03` |
| reserve_growth | ⤵ Δ(liquidity_index) |
| borrow_amount | `02` borrow_amount_raw |
| total_borrow, total_borrowed | ⤵ (raw `net_debt_flow_raw` in `02`) |
| borrow_rate | `02` last_borrow_rate (end-of-period) |
| avg_borrow_rate, avg_stable_borrow_rate, avg_variable_borrow_rate | ⤵ |
| stable_borrow_rate, variable_borrow_rate | `03` |
| interest_rate_mode | `02` stable/variable tx counts (categorical → counts) |
| borrow_cap, old_borrow_cap, debt_ceiling, old_debt_ceiling | `04` |
| total_debt_base, avg_debt_base | `08` avg_total_debt_base |
| liquidation_debt_covered | `05` liquidation_debt_covered_raw |
| liquidated_collateral | `05` liquidated_collateral_raw |
| receive_atoken | `05` receive_atoken_count |
| liquidation_threshold | `04` |
| current_liquidation_threshold | `08` avg_current_liquidation_threshold |
| avg_liquidation_threshold | ⤵ |
| liquidation_bonus | `04` |
| liquidation_rate, protocol_liquidation_pressure | ⤵ |
| flashloan_amount | `06` flashloan_amount_raw |
| flashloan_volume | ⤵ |
| flashloan_fee | `06` flashloan_premium_raw |
| flashloan_fee_revenue | ⤵ |
| flashloan_interest_mode | `06` mode 0/1/2 counts |
| flashloan_target | 🚫 address |
| flashloan_transaction_count | `06` flashloan_tx_count |
| unique_flashloan_users | `06` unique_flashloan_initiators |
| reserve_factor, old_reserve_factor | `04` |
| variable_borrow_index | `03` |
| utilization_rate | ⤵ |
| reserve (asset) | composite key in `01–07`, `09` |
| total_collateral_base, avg_collateral_base | `08` avg_total_collateral_base |
| available_borrows_base, total/avg_available_borrows | `08` avg_available_borrows_base |
| health_factor, min/max_health_factor | `08` min/max_health_factor |
| avg_health_factor | ⤵ (use min/max) |
| user_ltv, avg_ltv | `08` avg_ltv |
| max_ltv | ⤵ |
| critical/high/medium/low_risk_users | ⤵ (segment from health factor) |
| active_users, active_risk_users | ⤵ (`08` sampled_user_count; per-asset unique counts in `01/02`) |
| borrower, supplier, repay_user, collateral_enabled_user | unique-count form in `01/02/09`; raw wallets 🚫 |
| reserve_count, active_reserve_count | ⤵ COUNT(DISTINCT asset) downstream |
| transaction_count | ⤵ Σ tx_count columns |
| collateralization_ratio | ⤵ |
| oracle_asset_price | `07` |
| block_number | `latest_*_block` columns | 
| block_time | `time_bucket` |
| tx_hash, tx_from | 🚫 |

\* `net_flow` is defined identically to `total_liquidity` (supply − withdraw) in
`Preferred_metrics.md`; the raw signed flow `net_supply_flow_raw` is extracted in `01`.

---

## 4. Compliance with `Dune_Context.md` / `Plan.md`

- ✅ Raw variables only — no LTV/utilization/HF-trend/rolling/z-score/VaR/etc. in SQL.
- ✅ No wallet addresses, tx hashes, ENS, URLs, or text in any output column. Wallet fields
  appear only inside `approx_distinct()`.
  - ⚠️ Exception (by explicit request): asset-level queries also emit `asset_symbol`
    (`05` emits `collateral_asset_symbol` + `debt_asset_symbol`), a textual token symbol
    resolved on-chain from `tokens.erc20`. This is a human-readable QA/readability label,
    NOT a model feature — downstream joins must still key on the `asset` (address), not the
    symbol. `08` (user-account) has no asset dimension, so it carries no symbol.
- ✅ ≤ 10–15 metrics per query; tables grouped by functional similarity; one file per table.
- ✅ Fixed 6-hour buckets (00/06/12/18 UTC); `time_bucket` primary key; asset-level tables use
  composite `(time_bucket, asset)` (liquidations: `(time_bucket, collateral_asset, debt_asset)`;
  Part 6 / user account: `time_bucket` only).
- ✅ Window `2025-11-01` (incl) → `2026-02-01` (excl).
- ✅ Sign convention applied at extraction: supply/borrow inflow `+`, withdrawal/repay `−` in the
  net-flow columns.
- ✅ Missing buckets are absent / NULL — **no** forward/back fill, no interpolation, no row repeats.
- ✅ A decimal-reference table per part (`decimal_reference_part*.sql`).
- ✅ Deterministic, stable schema, consistent naming, `ORDER BY time_bucket`.
