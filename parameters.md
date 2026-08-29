# Parameter provenance

Every number baked into `RealisticCostModel`, the contract spec registry, and the synthetic
market generator is recorded here with where it came from and how much to trust it. Code
carries no commentary by design, so this file is the only prose in the repository.

## Commission and fee schedule (`slipstream/costs/presets.py`)

Modeled on a fixed-rate retail futures schedule of the kind published by Interactive
Brokers, NinjaTrader, and AMP, cross-checked against the CME and CBOT exchange fee
schedules for non-member customers. Figures are per contract per side in USD and were
compiled in mid 2026. Broker pricing changes frequently, so treat these as
representative retail figures rather than a quote.

| Root | Broker commission | Exchange fee | Clearing fee | NFA fee | Total per side |
|------|-------------------|--------------|--------------|---------|----------------|
| ES   | 0.85              | 1.40         | 0.10         | 0.02    | 2.37           |
| MES  | 0.25              | 0.37         | 0.05         | 0.02    | 0.69           |
| ZN   | 0.85              | 0.87         | 0.10         | 0.02    | 1.84           |
| ZF   | 0.85              | 0.87         | 0.10         | 0.02    | 1.84           |

Notes.

- The broker commission is the IBKR fixed-tier headline rate for US futures (0.85 for
  full-size contracts, 0.25 for micros).
- Exchange fees follow the CME Globex non-member equity index schedule (1.40 ES, 0.37 MES)
  and the CBOT interest rate schedule (0.87 for ZN and ZF).
- The NFA assessment fee is 0.02 per side for all US futures.
- Clearing is folded into exchange fees at some brokers; it is kept as a separate line here
  so the layer decomposition stays honest.

## Impact model coefficients (`slipstream/costs/layers.py`)

- `SquareRootImpact.coefficient = 0.7`. The square root law
  `impact = c * sigma_daily * sqrt(Q / ADV)` with c between 0.5 and 1.0 is the standard
  empirical finding of the market impact literature (Almgren et al. 2005 on US equities,
  and subsequent replications on futures). 0.7 is the middle of that range. This is a
  cross-asset stylized fact, not an ES-specific calibration.
- `LinearParticipationImpact.coefficient = 0.5` gives impact linear in participation of
  interval volume, scaled by daily vol. This is the pre-square-root-law industry
  heuristic and is included as an alternative functional form, not a calibrated model.

## Latency (`slipstream/costs/presets.py`)

`LognormalLatency(median = 0.25s, sigma = 0.8)` describes a retail order path: platform,
broker risk checks, and Globex acknowledgement. Published Globex round-trip times are
single-digit milliseconds; the retail stack on top of it is the dominant term and is
in the hundreds of milliseconds. The lognormal shape gives the heavy right tail that
outage and requeue events produce.

## Roll cost (`slipstream/costs/layers.py`)

`RollCostLayer.calendar_spread_ticks = 1.0` charges half of a one-tick calendar spread
market per side of the roll. The ES calendar spread trades one tick wide (0.05 index
points on the spread book against 0.25 outright) essentially all day during roll week,
which is why rolling through the spread book is much cheaper than legging two outrights.
One outright tick is used here as a conservative stand-in since the simulator quotes
outright ticks only.

## Financing (`slipstream/costs/presets.py`)

`collateral_opportunity_rate = 0.045` treats posted margin as cash that could otherwise
sit in T-bills, using a mid-2026 short bill yield of roughly 4.5 percent. Futures margin
itself accrues no explicit financing charge at most FCMs, hence `margin_rate = 0`.

## Contract specs (`slipstream/contracts/specs.py`)

Tick sizes, tick values, and multipliers are exchange constants from CME Group contract
specifications: ES 0.25 points at 12.50, MES 0.25 at 1.25, ZN 1/64 at 15.625, ZF 1/128
at 7.8125. Session times are the US day session expressed in UTC. The holiday list covers
the major full-close US holidays for 2025 and 2026. Margins are CME Group maintenance
levels observed mid 2026 rounded to round numbers; exchanges revise them with volatility,
so they are indicative only.

## Synthetic market generator (`slipstream/data/synthetic.py`)

- `daily_vol = 1.1%` is in line with realized ES daily vol in a mid-teens VIX regime.
- The U-shaped intraday vol and spread profile, thinner books in fast markets, and
  clustered trade arrivals (AR(1) log intensity) reproduce well-documented intraday
  stylized facts; the specific coefficients were chosen so that the generated tape looks
  reasonable to an eye trained on ES, and carry no claim of calibration to any dataset.
- `trades_per_second = 1.4` and top-of-book depth around 40 lots are day-session ES
  orders of magnitude.

## Queue model defaults (`slipstream/fills/queue.py`)

`cancel_to_trade_ratio = 2.0` assumes two lots cancelled ahead of you for every lot that
trades, consistent with the order-of-magnitude cancel activity visible in CME market data
studies. It is configurable because this number varies strongly by product and by level
depth.
