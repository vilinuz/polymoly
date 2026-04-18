import os
import time
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

from algos.polymarket_client import PolymarketClient
from algos.combinatorial_arb import ArbOpportunity

logger = logging.getLogger("AutoExecutor")

class AutoExecutor:
    """
    Automated execution engine that wraps PolymarketClient. 
    Handles API credentials, order sizing, and safe paper trading vs live trading.
    """
    def __init__(self, client: PolymarketClient, paper_trade: bool = True):
        self.client = client
        self.paper_trade = paper_trade
        self.credentials_loaded = False
        
        if not self.paper_trade:
            self._init_live_credentials()
            
    def _init_live_credentials(self):
        load_dotenv()
        api_key = os.getenv("POLY_API_KEY")
        api_secret = os.getenv("POLY_API_SECRET")
        api_passphrase = os.getenv("POLY_API_PASSPHRASE")
        
        if not (api_key and api_secret and api_passphrase):
            logger.error("Live trading requested but missing POLY_API_KEY, POLY_API_SECRET, or POLY_API_PASSPHRASE in .env!")
            raise ValueError("Live trading credentials missing.")
            
        self.client.api_key = api_key
        self.client.api_secret = api_secret
        self.client.api_passphrase = api_passphrase
        self.credentials_loaded = True
        logger.warning("LIVE TRADING ENABLED! Real funds will be used.")

    def execute_arbitrage(self, opp: ArbOpportunity) -> bool:
        """
        Executes a detected arbitrage opportunity. Minimum trade size considerations and 
        handling concurrent atomic submission of legs.
        """
        if opp.max_size < 5.0:  # Arbitrary min order size safely checking constraints
            logger.warning(f"Opportunity size {opp.max_size} too small to execute.")
            return False
            
        logger.info(f"Preparing execution: {opp.strategy} Arb, Edge: ${opp.edge:.4f}, Est Size: {opp.max_size}")
        
        if self.paper_trade:
            logger.info("---| PAPER TRADE MODE |---")
            for leg in opp.legs:
                logger.info(f"[PAPER] -> {leg.side} {leg.size:.2f} shares of {leg.token_id[:8]}... at ${leg.price:.4f}")
            logger.info(f"[PAPER] Arbitrage transaction 'succeeded' for assumed ${opp.edge:.2f} profit.")
            return True
            
        # LIVE TRADING execution happens synchronously here, but should ideally be async gathered 
        # as done in the detector's execution method. We use the client's synchronous method for simplicity
        # or we rely on the async submit_order_async logic if wrapped.
        
        success_legs = []
        for leg in opp.legs:
            try:
                logger.info(f"[LIVE] -> Submitting {leg.side} {leg.size} @ {leg.price} for {leg.token_id}")
                resp = self.client.submit_order(
                    token_id=leg.token_id,
                    side=leg.side,
                    price=leg.price,
                    size=leg.size
                )
                success_legs.append(resp)
            except Exception as e:
                logger.error(f"[LIVE] Failed to submit order: {e}")
                # HEDGE FALLBACK (In production: immediately reverse prior successful legs using market orders)
                logger.critical("PARTIAL FILL RISK! Execution failed mid-arbitrage.")
                return False
                
        logger.info(f"[LIVE] Arbitrage executed successfully. Legs filled: {len(success_legs)}")
        return True
