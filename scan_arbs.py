import argparse
import logging
import time
from tabulate import tabulate
from tqdm import tqdm

from algos.polymarket_client import PolymarketClient
from algos.arb_scanner import LiveArbitrageScanner, EventDiscovery, get_fee_rate
from algos.arb_backtest import ArbBacktestEngine
from algos.auto_executor import AutoExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ScanArbs")

def format_opportunities(opportunities, min_edge_bps):
    table_data = []
    for opp in opportunities:
        if opp['edge_bps'] >= min_edge_bps:
            table_data.append([
                opp['title'][:50],
                opp['strategy'],
                f"${opp['edge']:.4f}",
                f"{opp['edge_bps']:.1f}",
                f"${opp['cost']:.2f}",
                f"{opp['max_size']:.1f}"
            ])
            
    if table_data:
        print("\n" + tabulate(
            table_data, 
            headers=["Event", "Strategy", "Edge ($)", "Edge (bps)", "Implied Cost ($)", "Max Size"],
            tablefmt="grid"
        ) + "\n")
    else:
        print(f"\nNo opportunities found above {min_edge_bps} bps edge.\n")

def run_live_scan(min_edge_bps: float, min_liquidity: float, snapshot: bool = False, limit_events: int = None, auto_execute: bool = False, paper_trade: bool = True):
    client = PolymarketClient()
    scanner = LiveArbitrageScanner(client, min_edge_bps=min_edge_bps, min_liquidity=min_liquidity)
    backtester = ArbBacktestEngine() if snapshot else None
    executor = AutoExecutor(client=client, paper_trade=paper_trade) if auto_execute else None
    
    # We will override the scanner's run_scan_cycle slightly here just to add tqdm
    # for a better CLI experience
    
    logger.info("Starting live scan...")
    events = scanner.discovery.fetch_all_active_events(max_pages=limit_events)
    market_groups = scanner.discovery.extract_conditionally_exclusive_groups(events)
    logger.info(f"Extracted {len(market_groups)} mutually exclusive market groups.")
    
    all_token_ids = set()
    for g in market_groups:
        all_token_ids.update(g['tokens'])
    all_token_ids = list(all_token_ids)
    
    print(f"\nFetching CLOB snapshots for {len(all_token_ids)} tokens...")
    valid_books = 0
    for i in tqdm(range(len(all_token_ids)), desc="Fetching Order Books"):
        tid = all_token_ids[i]
        try:
            scanner.client.fetch_orderbook_snapshot(tid)
            valid_books += 1
        except Exception:
            pass
        if i > 0 and i % 50 == 0:
            time.sleep(0.5) # rate limit
            
    logger.info(f"Successfully fetched {valid_books} active order books.")
    
    scanner.detector.opportunities_found.clear()
    
    print(f"\nScanning {len(market_groups)} event groups for arbitrage...")
    for group in tqdm(market_groups, desc="Detecting Arbitrage"):
        fee_rate = get_fee_rate(group.get('category', 'Other'))
        scanner.detector.fee_rate = fee_rate
        scanner.detector.detect_overround_arb(group['tokens'])
        scanner.detector.detect_underround_arb(group['tokens'])
        
    # Gather results
    opportunities = []
    for opp in scanner.detector.opportunities_found:
        title = "Unknown"
        for g in market_groups:
            if set(opp.legs[0].token_id).issubset(set(g['tokens'])):
                 title = g['title']
                 break
        opp_dict = {
            'timestamp': opp.timestamp,
            'strategy': opp.strategy,
            'title': title,
            'edge': opp.edge,
            'edge_bps': (opp.edge / opp.implied_cost) * 10000 if opp.implied_cost > 0 else 0,
            'max_size': opp.max_size,
            'cost': opp.implied_cost,
            'legs': [{'token': l.token_id, 'side': l.side, 'price': l.price, 'size': l.size} for l in opp.legs]
        }
        opportunities.append(opp_dict)
        
    results = {
        'timestamp': time.time(),
        'events_scanned': len(events),
        'groups_scanned': len(market_groups),
        'tokens_scanned': len(all_token_ids),
        'opportunities': opportunities
    }
        
    format_opportunities(opportunities, min_edge_bps)
    
    if executor:
        for opp in scanner.detector.opportunities_found:
            # Check edge explicitly again before executing
            implied_cost = opp.implied_cost
            if implied_cost > 0:
                edge_bps = (opp.edge / implied_cost) * 10000
                if edge_bps >= min_edge_bps:
                    logger.info("AutoExecutor: Arbitrage threshold met, executing...")
                    executor.execute_arbitrage(opp)
    
    if snapshot:
        backtester.save_snapshot(scanner, results)

def main():
    parser = argparse.ArgumentParser(description="Polymarket Arbitrage Scanner")
    parser.add_argument("--mode", choices=["live", "monitor", "snapshot", "backtest"], required=True, help="Operating mode")
    parser.add_argument("--min-edge", type=float, default=5.0, help="Minimum edge in basis points")
    parser.add_argument("--min-liquidity", type=float, default=10.0, help="Minimum book depth")
    parser.add_argument("--interval", type=int, default=300, help="Monitor interval in seconds")
    parser.add_argument("--limit-events", type=int, default=100, help="Limit fetch to this many event pages (1 page = 100 events)")
    parser.add_argument("--auto-execute", action="store_true", help="Enable automatic opportunity execution")
    parser.add_argument("--live-trade", action="store_true", help="Execute real orders! Uses .env credentials. Default is paper trade.")
    
    args = parser.parse_args()
    paper_trade = not args.live_trade
    
    if args.mode == "live":
        run_live_scan(args.min_edge, args.min_liquidity, snapshot=False, limit_events=args.limit_events, auto_execute=args.auto_execute, paper_trade=paper_trade)
        
    elif args.mode == "snapshot":
        run_live_scan(args.min_edge, args.min_liquidity, snapshot=True, limit_events=args.limit_events, auto_execute=args.auto_execute, paper_trade=paper_trade)
        
    elif args.mode == "monitor":
        logger.info(f"Starting continuous monitoring (interval: {args.interval}s)")
        try:
            while True:
                run_live_scan(args.min_edge, args.min_liquidity, snapshot=True, limit_events=args.limit_events, auto_execute=args.auto_execute, paper_trade=paper_trade)
                logger.info(f"Sleeping for {args.interval}s...")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user.")
            
    elif args.mode == "backtest":
        engine = ArbBacktestEngine()
        engine.run_replay(min_edge_bps=args.min_edge, min_liquidity=args.min_liquidity)

if __name__ == "__main__":
    main()
