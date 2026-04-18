import os
import json
import time
import glob
from typing import List, Dict, Any
import logging

from algos.arb_scanner import LiveArbitrageScanner
from algos.polymarket_client import PolymarketClient, OrderBook, OrderBookLevel

logger = logging.getLogger("ArbBacktest")

class ArbBacktestEngine:
    """Handles snapshot persistence and replaying historical data."""
    def __init__(self, data_dir: str = "data/snapshots"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        
    def save_snapshot(self, scanner: LiveArbitrageScanner, results: Dict[str, Any]):
        """Persist the current state of all orderbooks and the discovery groups."""
        timestamp = int(results['timestamp'])
        filename = f"{self.data_dir}/snapshot_{timestamp}.json"
        
        # We need to save the orderbooks we fetched
        books_data = {}
        for tid, book in scanner.client.books.items():
            books_data[tid] = {
                'bids': [{'price': b.price, 'size': b.size} for b in book.bids],
                'asks': [{'price': a.price, 'size': a.size} for a in book.asks],
                'timestamp': book.timestamp
            }
            
        data = {
            'timestamp': timestamp,
            'results': results, # The opportunities found
            'books': books_data
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f)
            
        logger.info(f"Saved snapshot to {filename} ({len(books_data)} books)")
        return filename
        
    def load_snapshot(self, filename: str) -> Dict[str, Any]:
        """Load a single snapshot."""
        with open(filename, 'r') as f:
            return json.load(f)
            
    def run_replay(self, min_edge_bps: float = 10.0, min_liquidity: float = 50.0):
        """Replay all snapshots in the data directory through the detector logic."""
        files = glob.glob(f"{self.data_dir}/snapshot_*.json")
        files.sort() # chronological
        
        if not files:
            logger.warning(f"No snapshots found in {self.data_dir}")
            return
            
        logger.info(f"Replaying {len(files)} snapshots...")
        
        total_opps = 0
        total_edge = 0.0
        
        for file in files:
            data = self.load_snapshot(file)
            timestamp = data['timestamp']
            books_data = data.get('books', {})
            
            # Reconstruct dummy client with historical books
            client = PolymarketClient()
            for tid, bdata in books_data.items():
                book = OrderBook(token_id=tid, timestamp=bdata['timestamp'])
                book.bids = [OrderBookLevel(b['price'], b['size']) for b in bdata['bids']]
                book.asks = [OrderBookLevel(a['price'], a['size']) for a in bdata['asks']]
                client.books[tid] = book
                
            # Run scanner manually on the reconstructed state
            # NOTE: For true backtest we'd also need the groups, but the original
            # groups are stored implicitly in the 'results' if we saved them, or we can just 
            # use the stored results. Here we just analyze the stored results for demonstration.
            
            results = data.get('results', {})
            opps = results.get('opportunities', [])
            
            # Filter based on backtest parameters
            valid_opps = []
            for opp in opps:
                if opp['edge_bps'] >= min_edge_bps and opp['max_size'] >= min_liquidity:
                    valid_opps.append(opp)
                    
            if valid_opps:
                logger.info(f"[{timestamp}] Found {len(valid_opps)} opportunities.")
                for opp in valid_opps:
                    logger.debug(f"  - {opp['title'][:50]}... Edge: {opp['edge']:.4f} ({opp['edge_bps']:.1f} bps)")
                    total_edge += opp['edge']
                    total_opps += 1
                    
        logger.info("--- Backtest Summary ---")
        logger.info(f"Total Snapshots: {len(files)}")
        logger.info(f"Total Opportunities Found: {total_opps}")
        logger.info(f"Cumulative Edge (Assuming 100% Fill): ${total_edge:.2f}")
