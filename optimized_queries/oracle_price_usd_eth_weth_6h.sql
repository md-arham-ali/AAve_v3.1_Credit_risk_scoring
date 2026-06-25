-- =====================================================================
-- Asset price extraction (USD / ETH / WETH) for transacted assets
-- Grain / composite key : (time_bucket, asset)
-- Window  : 2025-11-01 00:00 (incl) .. 2026-02-01 00:00 (excl), UTC
-- Bucket  : fixed 6-hour slots (00 / 06 / 12 / 18 UTC)
-- Sources : prices.hour  (hourly USD price feed)
--           tokens.erc20 (token decimals)
--
-- Output columns
--   time_bucket     6-hour UTC bucket
--   asset           token contract address
--   decimals        token decimals (tokens.erc20)
--   avg_price_usd   mean hourly USD price within the bucket
--   avg_price_eth   mean hourly price in ETH within the bucket
--   avg_price_weth  mean hourly price in WETH within the bucket
--   price_points    # hourly observations averaged (coverage indicator)
--
-- Notes
--   * Asset universe = the 58 distinct assets present in query_result_data/
--     (the supply/borrow/reserve/config/liquidation/flashloan/collateral tables).
--   * Price feed is hourly; each 6h bucket averages up to 6 hourly prices.
--   * ETH and WETH denominations both divide by the WETH/USD feed
--     (0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2). On Ethereum mainnet ETH and
--     WETH are redeemable 1:1, so an asset's price in ETH equals its price in
--     WETH. Replace the `weth` CTE with a native-ETH / Chainlink ETH-USD feed if
--     a distinct ETH source is required.
--   * Raw averages only; missing buckets stay absent (no fill). Assets without a
--     price feed (e.g. Pendle PT-* tokens) simply produce no rows.
--   * prices.hour.timestamp is tz-aware; it is filtered with UTC literals and
--     cast to a plain UTC timestamp so the bucket matches the event-based tables.
-- =====================================================================
WITH assets (asset, symbol) AS (
    VALUES
        (0x111111111117dc0aa78b770fa6a738034120c302, '1INCH'),
        (0x14bdc3a3ae09f5518b923b69489cbcafb238e617, 'PT-eUSDE-14AUG2025'),
        (0x18084fba666a33d37592fa2633fd49a74dd93a88, 'tBTC'),
        (0x1abaea1f7c830bd89acc67ec4af516284b1bc33c, 'EURC'),
        (0x1f84a51296691320478c98b8d77f2bbd17d34350, 'PT-USDe-5FEB2026'),
        (0x1f9840a85d5af5bf1d1762f925bdaddc4201f984, 'UNI'),
        (0x2260fac5e5542a773aa44fbcfedf7c193bc2c599, 'WBTC'),
        (0x356b8d89c1e1239cbbb9de4815c39a1474d5ba7d, 'syrupUSDT'),
        (0x3b3fb9c57858ef816833dc91565efcd85d96f634, 'PT-sUSDE-31JUL2025'),
        (0x40d16fc0246ad3160ccc09b8d0d3a2cd28ae6c2f, 'GHO'),
        (0x4c9edd5852cd905f086c759e8383e09bff1e68b3, 'USDe'),
        (0x50d2c7992b802eef16c04feadab310f31866a545, 'PT-eUSDE-29MAY2025'),
        (0x514910771af9ca656af840dff83e8264ecf986ca, 'LINK'),
        (0x5a98fcbea516cf06857215779fd812ca3bef1b32, 'LDO'),
        (0x5f98805a4e8be255a32880fdec7f6728c6568ba0, 'LUSD'),
        (0x62c6e813b9589c3631ba0cdb013acdb8544038b7, 'PT-USDe-27NOV2025'),
        (0x657e8c867d8b37dcc18fa4caead9c45eb088c642, 'eBTC'),
        (0x68749665ff8d2d112fa859aa293f07a622782f38, 'XAUt'),
        (0x6b175474e89094c44da98b954eedeac495271d0f, 'DAI'),
        (0x6c3ea9036406852006290770bedfcaba0e23a0e8, 'PYUSD'),
        (0x7f39c581f595b53c5cb19bd0b3f8da6c935e2ca0, 'wstETH'),
        (0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9, 'AAVE'),
        (0x8236a87084f8b84306f72007f36f2618a5634494, 'LBTC'),
        (0x8292bb45bf1ee4d140127049757c2e0ff06317ed, 'RLUSD'),
        (0x83f20f44975d03b1b09e64809b757c47f942beea, 'sDAI'),
        (0x853d955acef822db058eb8505911ed77f175b99e, 'FRAX'),
        (0x90d2af7d622ca3141efa4d8f1f24d86e5974cc8f, 'eUSDe'),
        (0x917459337caac939d41d7493b3999f571d20d667, 'PT-USDe-31JUL2025'),
        (0x9d39a5de30e57443bff2a8307a4256c8797a3497, 'sUSDe'),
        (0x9f56094c450763769ba0ea9fe2876070c0fd5f77, 'PT-sUSDE-25SEP2025'),
        (0x9f8f72aa9304c8b593d555f12ef6589cc3a579a2, 'MKR'),
        (0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48, 'USDC'),
        (0xa1290d69c65a6fe4df752f95823fae25cb99e5a7, 'rsETH'),
        (0xa35b1b31ce002fbf2058d22f30f95d405200a15b, 'ETHx'),
        (0xaca92e438df0b2401ff60da7e4337b687a2435da, 'mUSD'),
        (0xae78736cd615f374d3085123a210448e74fc6393, 'rETH'),
        (0xba100000625a3754423978a60c9317c58a424e3d, 'BAL'),
        (0xbc6736d346a5ebc0debc997397912cd9b8fae10a, 'PT-USDe-25SEP2025'),
        (0xbe9895146f7af43049ca1c1ae358b0541ea49704, 'cbETH'),
        (0xbf5495efe5db9ce00f80364c8b423567e58d2110, 'ezETH'),
        (0xc011a73ee8576fb46f5e1c5751ca3b9fe0af2a6f, 'SNX'),
        (0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2, 'WETH'),
        (0xc139190f447e929f090edeb554d95abb8b18ac1c, 'USDtb'),
        (0xc18360217d8f7ab5e7c516566761ea12ce7f9d72, 'ENS'),
        (0xc96de26018a54d51c097160568752c4e3bd6c364, 'FBTC'),
        (0xcbb7c0000ab88b473b1f5afd9ef808440eed33bf, 'cbBTC'),
        (0xcd5fe23c85820f7b72d0926fc9b05b43e359b7ee, 'weETH'),
        (0xd11c452fc99cf405034ee446803b6f6c1f6d5ed8, 'tETH'),
        (0xd33526068d116ce69f19a9ee46f0bd304f21a51f, 'RPL'),
        (0xd533a949740bb3306d119cc777fa900ba034cd52, 'CRV'),
        (0xdac17f958d2ee523a2206206994597c13d831ec7, 'USDT'),
        (0xdc035d45d973e3ec169d2276ddab16f1e407384f, 'USDS'),
        (0xdefa4e8a7bcba345f687a2f1456f5edd9ce97202, 'KNC'),
        (0xe343167631d89b6ffc58b88d6b7fb0228795491d, 'USDG'),
        (0xe6a934089bbee34f832060ce98848359883749b3, 'PT-sUSDE-27NOV2025'),
        (0xe8483517077afa11a9b07f849cee2552f040d7b2, 'PT-sUSDE-5FEB2026'),
        (0xf1c9acdc66974dfb6decb12aa385b9cd01190e38, 'osETH'),
        (0xf939e0a03fb07f59a73314e73794be0e57ac1b4e, 'crvUSD')
),
eth_prices AS (
    SELECT
        CAST(timestamp AS timestamp) AS ts,
        price AS eth_usd_price
    FROM prices.hour
    WHERE blockchain = 'ethereum'
      AND contract_address = 0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2
      AND timestamp >= TIMESTAMP '2025-04-01 00:00:00 UTC'
      AND timestamp < TIMESTAMP '2026-03-31 00:00:00 UTC'
),
px AS (
    SELECT
        p.contract_address AS asset,
        CAST(p.timestamp AS timestamp) AS ts,
        p.price AS price_usd,
        p.price / e.eth_usd_price AS price_eth
    FROM prices.hour p
    INNER JOIN eth_prices e ON p.timestamp = e.ts
    WHERE p.blockchain = 'ethereum'
      AND p.timestamp >= TIMESTAMP '2025-04-01 00:00:00 UTC'
      AND p.timestamp < TIMESTAMP '2026-03-31 00:00:00 UTC'
      AND p.contract_address IN (SELECT asset FROM assets)
),
dec AS (
    SELECT contract_address AS asset, decimals
    FROM tokens.erc20
    WHERE blockchain = 'ethereum'
)
SELECT
    date_add('hour',
             2 * CAST(floor(hour(p.ts) / 2) AS bigint),
             date_trunc('day', p.ts)) AS time_bucket,
    p.asset,
    MAX(a.symbol) AS symbol,
    MAX(d.decimals) AS decimals,
    AVG(p.price_usd) AS avg_price_usd,
    AVG(p.price_eth) AS avg_price_eth,
    COUNT(p.price_usd) AS price_points
FROM px p
LEFT JOIN dec d ON d.asset = p.asset
LEFT JOIN assets a ON a.asset = p.asset
GROUP BY 1, 2
ORDER BY 1, 2