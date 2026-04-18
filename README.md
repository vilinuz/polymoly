# Polymarket Arbitrage Scanner (PolyMoly)

An advanced, high-frequency Arbitrage Scanner and Backtesting Engine for [Polymarket](https://polymarket.com), leveraging predictive market structural inefficiencies. This system operates entirely automatically, paginating the Gamma API to discover new markets, querying the CLOB L2 API for sub-second order book depths, and crunching prediction combinatorics using Linear Programming to guarantee risk-free returns.

## Core Capabilities

1. **Auto-Discovery**: Extracts all active markets globally from the network, automatically resolving `negRisk` and conditionally disjoint conditions.
2. **Arithmetic Overround/Underround**: Calculates synthetic combinations of YES/NO limit orders to detect "buy-all-yes" and "sell-all-yes" arbitrages, adjusting dynamically for Polymarket tiered taker/maker fees.
3. **Linear Programming State-Space (LP) Engine**: Uses `scipy.optimize.linprog` to construct "Dutch Book" arrays across deeply nested events. It mathematically searches for vector coordinates where the absolute payout across **every single future state of the world** strictly dominates the upfront portfolio cost.
4. **Auto Executor**: Capable of signing L2 HMAC-SHA256 authenticated orders automatically directly against the Polymarket Layer-2 exchange protocol to lock in arbitrary returns before market makers adapt.
5. **Backtest Snapshot Replay**: Dumps exact orderbook microstates into point-in-time JSON files, and supports offline re-running of strategies via historical replay.

---

## Installation

This project utilizes `uv` (or pip) for ultra-fast dependency management.

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

### Environment Variables

For Live Execution capabilities, you must specify your Layer-2 trading credentials via a local `.env` file:

```env
POLY_API_KEY=your_key
POLY_API_SECRET=your_secret
POLY_API_PASSPHRASE=your_passphrase
```

---

## Command-Line Usage

The executable script `scan_arbs.py` handles terminal invocation.

### 1. Live Market Scan

Scans Polymarket endpoints for active arbitrage matrices in real-time.

```bash
python scan_arbs.py --mode live --min-edge 5.0 --min-liquidity 10.0
```

*(Tip: pass `--limit-events 1` or `5` to test quickly instead of downloading all thousands of markets)*

### 2. Paper Trading Simulation

Scan and "simulate" market execution using your automated rules:

```bash
python scan_arbs.py --mode live --auto-execute
```

*Note: Safety first! Without the `--live-trade` flag, the `AutoExecutor` will **only log "PAPER TRADE" messages** and will NOT spend your USDC.*

### 3. Live Execution

Execute live bets immediately against real balances when algorithmic thresholds are met:

```bash
python scan_arbs.py --mode live --auto-execute --live-trade
```

### 4. Background Monitoring

Continuously run intervals to keep track of micro-arbs (e.g., every 5 minutes):

```bash
python scan_arbs.py --mode monitor --interval 300
```

### 5. Snapshot & Backtest

Save the real-time conditions of all orderbooks to `data/snapshots/`:

```bash
python scan_arbs.py --mode snapshot
```

Replay chronological snapshots, passing them through the LP solver and Arithmetic detectors offline:

```bash
python scan_arbs.py --mode backtest --min-edge 1.0
```

---

## Detailed Python Examples

You can decouple the `LiveArbitrageScanner` from the CLI and use it directly within your own Python automated strategies.

### 1. Initializing the Engine

```python
import os
from dotenv import load_dotenv

from algos.polymarket_client import PolymarketClient
from algos.arb_scanner import LiveArbitrageScanner, EventDiscovery

# 1. Initialize API Client
load_dotenv()
client = PolymarketClient(
    host="https://clob.polymarket.com",
    api_key=os.getenv("POLY_API_KEY"),
    api_secret=os.getenv("POLY_API_SECRET"),
    api_passphrase=os.getenv("POLY_API_PASSPHRASE")
)

# 2. Build the Arbitrage Scanner
# Require 5 bps minimum edge, and at least $50 of usable liquidity volume
scanner = LiveArbitrageScanner(client, min_edge_bps=5.0, min_liquidity=50.0)
```

### 2. Manual Cross-Market LP Solve

You don't need to scan all of Polymarket. If you have specific mutually exclusive generic tokens (like predicting who will win an election from 5 candidates), you can ask the mathematical solver if a combination yields free returns.

```python
from algos.combinatorial_arb import CombinatorialArbitrageDetector

# Array of token IDs that sum to 1.0 probability (Mutually Exclusive Outcomes)
token_group = [
    "935929492127981211272131173049126255058...", 
    "307453934715274863285897854516655533254...", 
    "128839019283748291010303847583909101830..."
]

detector = CombinatorialArbitrageDetector(client)
opportunity = detector.detect_lp_arb(token_group)

if opportunity:
    print(f"Mathematical Arbitrage exists!")
    print(f"   Guaranteed Return: ${opportunity.guaranteed_return}")
    print(f"   Implied Cost to size limit: ${opportunity.implied_cost}")
    print(f"   Required Trades:")
    
    for leg in opportunity.legs:
        print(f"     Order -> {leg.side} {leg.size} of {leg.token_id[:8]} at {leg.price}")
```

### 3. Safely Executing a Parsed Output

Feed the constructed `ArbOpportunity` object seamlessly into the `AutoExecutor`.

```python
from algos.auto_executor import AutoExecutor

# Make sure you wrap in paper_trade=True for sandbox validation
executor = AutoExecutor(client, paper_trade=True)

# Checks constraints, guarantees atomic flow tracking, and builds the correct signatures
did_execute = executor.execute_arbitrage(opportunity)

if did_execute:
    print("Execution pipeline passed successfully.")
```

### 4. Running the full Auto-Discovery API Hook

If you want the full sequence: Discovered -> CLOB fetch -> Arb Math -> Execution

```python
# Execute an automatic 1-page loop extraction
stats_dict = scanner.run_scan_cycle(limit_events=1) 

print(f"Found {stats_dict['events_scanned']} active live events!")
print(f"Out of these, extracted {stats_dict['groups_scanned']} mutually exclusive groups.")

for result in stats_dict['opportunities']:
    print(f"Identified Arbitrage in: {result['title']}")
    print(f"   -> Edge Yield: {result['edge_bps']} bps")
```
