import time
import asyncio
import logging
from typing import Dict, List, Any
import requests
from requests.exceptions import RequestException

from algos.polymarket_client import PolymarketClient
from algos.combinatorial_arb import CombinatorialArbitrageDetector

logger = logging.getLogger("ArbScanner")

# Fee rates per Gamma API documentation
FEE_RATES = {
    "Crypto": 0.0005,
    "Sports": 0.0005,
    "Politics": 0.0005, # Some categories are zero or lower depending on exact promos, 
    # but using baseline estimate to be safe unless fetchable dynamically
}

def get_fee_rate(category: str) -> float:
    return FEE_RATES.get(category, 0.001)  # default 10 bps

class EventDiscovery:
    """Discovers active events and groups mutually exclusive markets."""
    def __init__(self, gamma_api_url: str = "https://gamma-api.polymarket.com"):
        self.gamma_api_url = gamma_api_url
        
    def fetch_all_active_events(self, limit: int = 100, max_pages: int = None) -> List[Dict[str, Any]]:
        """Paginate through all active events on Polymarket."""
        events = []
        offset = 0
        pages = 0
        logger.info("Fetching active events from Gamma API...")
        
        while True:
            try:
                response = requests.get(
                    f"{self.gamma_api_url}/events", 
                    params={"active": "true", "closed": "false", "limit": limit, "offset": offset},
                    timeout=10
                )
                response.raise_for_status()
                data = response.json()
                
                if not data:
                    break
                    
                events.extend(data)
                offset += limit
                pages += 1
                
                if max_pages and pages >= max_pages:
                    break
                    
                # Respect rate limits
                time.sleep(0.1)
                
            except RequestException as e:
                logger.error(f"Failed to fetch events at offset {offset}: {e}")
                break
                
        logger.info(f"Discovered {len(events)} active events.")
        return events
        
    def extract_conditionally_exclusive_groups(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract market groups that must sum to 1.0 probability.
        Returns a list of dicts: {'event_id': str, 'title': str, 'category': str, 'tokens': List[str]}
        """
        groups = []
        for event in events:
            markets = event.get('markets', [])
            if not markets:
                continue
                
            # Usually events have markets grouped by conditionId
            condition_groups = {}
            for m in markets:
                # We only want Active markets
                if not m.get('active', False) or m.get('closed', False):
                    continue
                    
                # We need the Yes token ID
                clob_tokens_str = m.get('clobTokenIds', '[]')
                
                try:
                    import json
                    tokens = json.loads(clob_tokens_str)
                except Exception:
                    tokens = []
                    
                if not tokens or len(tokens) == 0:
                    continue
                    
                token_id = tokens[0]  # The YES token is almost always the first one in the list for a binary market
                if not token_id:
                    continue
                    
                condition_id = m.get('conditionId', 'unknown')
                if condition_id not in condition_groups:
                    condition_groups[condition_id] = []
                condition_groups[condition_id].append(token_id)
                
            # If an event has multiple outcomes but only one market object with multiple tokens 
            # (common in negRisk markets), handle it.
            if event.get('enableNegRisk', False) or event.get('negRisk', False):
                # NegRisk markets often have multiple markets, but they are all mutually exclusive 
                # against the same event outcome set.
                group_tokens = []
                for m in markets:
                    if m.get('active', False) and not m.get('closed', False):
                        clob_tokens_str = m.get('clobTokenIds', '[]')
                        try:
                            import json
                            tokens = json.loads(clob_tokens_str)
                        except Exception:
                            tokens = []
                        if tokens and len(tokens) > 0:
                            yes_token = tokens[0]  # The YES token is almost always the first one
                            if yes_token:
                                group_tokens.append(yes_token)
                
                if len(group_tokens) > 1:
                    groups.append({
                        'event_id': event.get('id'),
                        'title': event.get('title'),
                        'category': event.get('category'),
                        'tokens': group_tokens,
                        'is_neg_risk': True
                    })
                    continue # Skip condition logic as we grouped them all
            
            # Standard conditional logic
            for condition_id, token_ids in condition_groups.items():
                if len(token_ids) > 1: # Only care if there's more than 1 outcome
                    groups.append({
                        'event_id': event.get('id'),
                        'title': event.get('title'),
                        'category': event.get('category'),
                        'tokens': token_ids,
                        'condition_id': condition_id,
                        'is_neg_risk': False
                    })
                    
        return groups

class LiveArbitrageScanner:
    def __init__(self, client: PolymarketClient, min_edge_bps: float = 10.0, min_liquidity: float = 50.0):
        self.client = client
        self.detector = CombinatorialArbitrageDetector(
            client=client, 
            min_edge_bps=min_edge_bps,
            min_liquidity=min_liquidity
        )
        self.discovery = EventDiscovery()
        
    def run_scan_cycle(self) -> Dict[str, Any]:
        """Full cycle: discover events -> fetch CLOB books -> detect arb."""
        start_time = time.time()
        
        # 1. Discover
        events = self.discovery.fetch_all_active_events(max_pages=20) # cap at 2000 events to be safe
        market_groups = self.discovery.extract_conditionally_exclusive_groups(events)
        logger.info(f"Extracted {len(market_groups)} mutually exclusive market groups.")
        
        # 2. Extract uniquely to prep CLOB cache
        # To avoid making N*M api calls, we fetch snapshots sequentially with rate limit backing off
        all_token_ids = set()
        for g in market_groups:
            all_token_ids.update(g['tokens'])
            
        all_token_ids = list(all_token_ids)
        logger.info(f"Fetching CLOB snapshots for {len(all_token_ids)} unique tokens...")
        
        # We process them in chunks so we can actually find arbs before they disappear,
        # but for a backtesting scanner, completing the whole state first is also fine.
        # Let's Seed the client's books
        self._seed_books(all_token_ids)
        
        # 3. Detect Arbitrage
        self.detector.opportunities_found.clear()
        
        for idx, group in enumerate(market_groups):
            # Try to grab fee rate specifically for this category
            fee_rate = get_fee_rate(group.get('category', 'Other'))
            self.detector.fee_rate = fee_rate
            
            self.detector.detect_overround_arb(group['tokens'])
            self.detector.detect_underround_arb(group['tokens'])
            
            # Use LP Advanced Solver for deep/complex arbitrage dependencies
            self.detector.detect_lp_arb(group['tokens'])
            
            if idx % 100 == 0 and idx > 0:
                logger.info(f"Processed {idx}/{len(market_groups)} groups...")
                
        elapsed = time.time() - start_time
        
        # Gather results
        opportunities = []
        for opp in self.detector.opportunities_found:
            # Try to attach title context
            title = "Unknown"
            for g in market_groups:
                if opp.legs[0].token_id in g['tokens']:
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
            
        # Optional Cross-market checks would go here...
        
        return {
            'timestamp': time.time(),
            'scan_duration': elapsed,
            'events_scanned': len(events),
            'groups_scanned': len(market_groups),
            'tokens_scanned': len(all_token_ids),
            'opportunities': opportunities
        }
        
    def _seed_books(self, token_ids: List[str]):
        """Fetches books for all tokens sequentially with rate limit protection."""
        valid_books = 0
        for i, tid in enumerate(token_ids):
            try:
                self.client.fetch_orderbook_snapshot(tid)
                valid_books += 1
            except Exception as e:
                # usually means market doesn't have open book yet
                pass
            
            # Simple rate limit for CLOB REST is ~100/sec public, but let's be nicer
            if i > 0 and i % 50 == 0:
                time.sleep(0.5)
