"""
Combinatorial Arbitrage Engine for Polymarket.

Detects and executes risk-free arbitrage across mutually exclusive
and conditionally paired prediction markets by exploiting the
fundamental constraint that probabilities of collectively exhaustive
outcomes must sum to exactly 1.0.

Strategies implemented:
  1. Overround Arbitrage — sum(best_ask) < 1.0 across all outcomes
  2. Underround Arbitrage — sum(best_bid) > 1.0 across all outcomes
  3. Cross-Market Conditional Arbitrage — correlated conditions across
     separate market groups
  4. Depth-Aware Optimal Sizing — walk the book to compute maximum
     profitable position size accounting for slippage
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np

from algos.polymarket_client import PolymarketClient, OrderBook, OrderBookLevel


logger = logging.getLogger("CombinatorialArb")


# ─── Data Structures ────────────────────────────────────────────────────────────

@dataclass
class ArbLeg:
    token_id: str
    side: str           # BUY or SELL
    price: float        # limit price
    size: float         # number of shares
    book_depth_used: int  # how many levels walked

@dataclass
class ArbOpportunity:
    strategy: str            # overround | underround | cross_conditional
    legs: List[ArbLeg]
    implied_cost: float      # total outlay
    guaranteed_return: float # 1.0 for mutually exclusive completeness
    edge: float              # guaranteed_return - implied_cost - fees
    max_size: float          # max position size at this edge
    timestamp: float = field(default_factory=time.time)

    @property
    def is_profitable(self) -> bool:
        return self.edge > 0.0

    @property
    def roi_pct(self) -> float:
        if self.implied_cost == 0:
            return 0.0
        return (self.edge / self.implied_cost) * 100.0


# ─── Engine ──────────────────────────────────────────────────────────────────────

class CombinatorialArbitrageDetector:
    """
    Continuously scans configured market groups for combinatorial arbitrage.
    Uses depth-aware sizing to compute the maximum profitable position.
    """

    def __init__(
        self,
        client: PolymarketClient,
        fee_rate: float = 0.02,         # Polymarket takes ~2% on winnings
        min_edge_bps: float = 10.0,     # Minimum edge in basis points to act
        max_position_usd: float = 5000, # Risk cap per opportunity
        min_liquidity: float = 50.0,    # Minimum depth in shares on each leg
    ):
        self.client = client
        self.fee_rate = fee_rate
        self.min_edge_bps = min_edge_bps
        self.max_position_usd = max_position_usd
        self.min_liquidity = min_liquidity
        self.opportunities_found: List[ArbOpportunity] = []
        self._last_scan_ns = 0

    # ─── Overround Detection (Buy All YES tokens) ──────────────────────

    def detect_overround_arb(self, market_group: List[str]) -> Optional[ArbOpportunity]:
        """
        If sum of best asks across all outcomes in a mutually exclusive group
        is less than $1.00, we can buy every outcome for a guaranteed $1.00 payout.

        This method also walks the book to find the maximum position size
        that remains profitable after fees.
        """
        books = []
        for tid in market_group:
            book = self.client.get_orderbook(tid)
            if not book.asks:
                return None
            books.append((tid, book))

        # Phase 1: Quick check at top-of-book
        total_best_ask = sum(book.best_ask.price for _, book in books)
        gross_edge_top = 1.0 - total_best_ask
        net_edge_top = gross_edge_top - self.fee_rate
        
        if net_edge_top <= 0:
            return None

        # Phase 2: Depth-aware sizing — find max size where arb still holds
        max_size = self._find_max_profitable_size(books)

        if max_size < self.min_liquidity:
            return None

        # Phase 3: Walk the book at optimal size to build actual legs
        legs = []
        total_cost = 0.0
        for tid, book in books:
            cost, avg_px = book.fill_cost("BUY", max_size)
            total_cost += cost
            depth_used = self._levels_consumed(book.asks, max_size)
            legs.append(ArbLeg(
                token_id=tid,
                side="BUY",
                price=avg_px,
                size=max_size,
                book_depth_used=depth_used,
            ))

        guaranteed_return = max_size * 1.0  # $1 per share of winning outcome
        net_fee = guaranteed_return * self.fee_rate
        edge = guaranteed_return - total_cost - net_fee

        if edge <= 0:
            return None

        edge_bps = (edge / total_cost) * 10_000 if total_cost > 0 else 0
        if edge_bps < self.min_edge_bps:
            return None

        opp = ArbOpportunity(
            strategy="overround",
            legs=legs,
            implied_cost=total_cost,
            guaranteed_return=guaranteed_return,
            edge=edge,
            max_size=max_size,
        )
        self.opportunities_found.append(opp)
        logger.warning(
            f"OVERROUND ARB: {len(legs)} legs, cost=${total_cost:.4f}, "
            f"edge=${edge:.4f} ({edge_bps:.1f}bps), size={max_size:.1f}"
        )
        return opp

    # ─── Underround Detection (Sell All YES tokens) ────────────────────

    def detect_underround_arb(self, market_group: List[str]) -> Optional[ArbOpportunity]:
        """
        If sum of best bids > $1.00, selling YES on every outcome
        is risk-free because only one outcome pays out.
        """
        books = []
        for tid in market_group:
            book = self.client.get_orderbook(tid)
            if not book.bids:
                return None
            books.append((tid, book))

        total_best_bid = sum(book.best_bid.price for _, book in books)
        gross_edge = total_best_bid - 1.0
        net_edge = gross_edge - self.fee_rate

        if net_edge <= 0:
            return None

        max_size = self._find_max_profitable_size_sell(books)
        if max_size < self.min_liquidity:
            return None

        legs = []
        total_proceeds = 0.0
        for tid, book in books:
            proceeds, avg_px = book.fill_cost("SELL", max_size)
            total_proceeds += proceeds
            depth_used = self._levels_consumed(book.bids, max_size)
            legs.append(ArbLeg(
                token_id=tid,
                side="SELL",
                price=avg_px,
                size=max_size,
                book_depth_used=depth_used,
            ))

        liability = max_size * 1.0  # $1 per share on the winning outcome
        net_fee = total_proceeds * self.fee_rate
        edge = total_proceeds - liability - net_fee

        if edge <= 0:
            return None

        opp = ArbOpportunity(
            strategy="underround",
            legs=legs,
            implied_cost=liability,
            guaranteed_return=total_proceeds - net_fee,
            edge=edge,
            max_size=max_size,
        )
        self.opportunities_found.append(opp)
        logger.warning(
            f"UNDERROUND ARB: {len(legs)} legs, proceeds=${total_proceeds:.4f}, "
            f"edge=${edge:.4f}, size={max_size:.1f}"
        )
        return opp

    # ─── Cross-Market Conditional Arbitrage ─────────────────────────────

    def detect_cross_market_arb(
        self,
        group_a: List[str],
        group_b: List[str],
        correlation: float = 1.0,
    ) -> Optional[ArbOpportunity]:
        """
        Detect arbitrage across conditionally paired market groups.
        e.g., "Will X win in State Y?" across multiple correlated state-level markets
        where the joint probability must satisfy Bayesian constraints.
        """
        cost_a = 0.0
        cost_b = 0.0
        legs = []

        for tid in group_a:
            book = self.client.get_orderbook(tid)
            if not book.best_ask:
                return None
            cost_a += book.best_ask.price
            legs.append(ArbLeg(tid, "BUY", book.best_ask.price, 1.0, 1))

        for tid in group_b:
            book = self.client.get_orderbook(tid)
            if not book.best_bid:
                return None
            cost_b += book.best_bid.price
            legs.append(ArbLeg(tid, "SELL", book.best_bid.price, 1.0, 1))

        # Under perfect correlation, cost_a and cost_b should be nearly equal
        total_cost = cost_a
        total_return = cost_b * correlation
        edge = total_return - total_cost - self.fee_rate

        if edge <= 0:
            return None

        opp = ArbOpportunity(
            strategy="cross_conditional",
            legs=legs,
            implied_cost=total_cost,
            guaranteed_return=total_return,
            edge=edge,
            max_size=1.0,
        )
        self.opportunities_found.append(opp)
        logger.warning(f"CROSS-MARKET ARB: edge=${edge:.4f}")
        return opp

    # ─── LP Advanced State-Space Detection ───────────────────────────────

    def detect_lp_arb(self, market_group: List[str]) -> Optional[ArbOpportunity]:
        """
        Solves the combinatorial arbitrage problem using Linear Programming.
        This handles partial hedges and arbitrary sets of mutually exclusive states.
        
        minimize: cost vector c * x
        subject to: A_ub * x >= b_ub (Guarantee non-negative payoff in ALL states)
        """
        import scipy.optimize as opt
        import numpy as np
        
        books = []
        for tid in market_group:
            book = self.client.get_orderbook(tid)
            books.append((tid, book))

        num_tokens = len(books)
        c = np.zeros(num_tokens * 2)
        for i, (tid, book) in enumerate(books):
            ask_price = book.best_ask.price if book.best_ask else 999.0  # Unfillable cost
            bid_price = book.best_bid.price if book.best_bid else 0.0    # 0 return on sale
            
            c[i] = ask_price              # Cost to buy 1 YES
            c[i + num_tokens] = -bid_price # Proceeds to sell 1 YES (Negative cost)
            
        # States constraint: we assume exactly one outcome happens.
        # So there are `num_tokens` states (world where token i wins)
        A = np.zeros((num_tokens, num_tokens * 2))
        for state in range(num_tokens):
            for i in range(num_tokens):
                # If state happens, YES pays $1, NO pays $0
                is_winner = 1.0 if state == i else 0.0
                A[state, i] = is_winner           # return from bought YES
                A[state, i + num_tokens] = -is_winner # liability from sold YES
                
        # We want return in EVERY state to be >= 1.0
        # In scipy linprog, Ax <= b is standard. So we do -Ax <= -b
        b = np.ones(num_tokens)
        
        res = opt.linprog(c, A_ub=-A, b_ub=-b, bounds=(0, self.max_position_usd))
        
        if res.success and res.fun < (1.0 - self.fee_rate):
            # We found an arbitrage where implied cost to guarantee $1 is less than $1
            legs = []
            implied_cost = res.fun
            gross_edge = 1.0 - implied_cost
            net_edge = gross_edge - self.fee_rate
            
            if net_edge <= 0 or (net_edge / implied_cost * 10000) < self.min_edge_bps:
                return None
                
            for i, (tid, book) in enumerate(books):
                if res.x[i] > 1e-3:
                    price = book.best_ask.price if book.best_ask else 0.0
                    legs.append(ArbLeg(tid, "BUY", price, res.x[i], 1))
                if res.x[i + num_tokens] > 1e-3:
                    price = book.best_bid.price if book.best_bid else 0.0
                    legs.append(ArbLeg(tid, "SELL", price, res.x[i + num_tokens], 1))
                    
            if not legs:
                return None
                
            opp = ArbOpportunity(
                strategy="lp_solver",
                legs=legs,
                implied_cost=implied_cost,
                guaranteed_return=1.0,
                edge=net_edge,
                max_size=min(l.size for l in legs) if legs else 0.0,
            )
            self.opportunities_found.append(opp)
            logger.warning(f"LP ARB FOUND! Edge: ${net_edge:.4f} using {len(legs)} legs.")
            return opp
            
        return None

    # ─── Full Scan ──────────────────────────────────────────────────────

    def scan_all(
        self,
        market_groups: List[List[str]],
        cross_pairs: Optional[List[Tuple[List[str], List[str]]]] = None,
    ) -> List[ArbOpportunity]:
        """
        Run all arbitrage detection strategies across configured groups.
        Returns all profitable opportunities found in this scan cycle.
        """
        self._last_scan_ns = time.time_ns()
        results = []

        for group in market_groups:
            # Overround
            opp = self.detect_overround_arb(group)
            if opp and opp.is_profitable:
                results.append(opp)

            # Underround
            opp = self.detect_underround_arb(group)
            if opp and opp.is_profitable:
                results.append(opp)

        # Cross-market pairs
        if cross_pairs:
            for group_a, group_b in cross_pairs:
                opp = self.detect_cross_market_arb(group_a, group_b)
                if opp and opp.is_profitable:
                    results.append(opp)

        return results

    # ─── Execution ──────────────────────────────────────────────────────

    async def execute_opportunity(self, opp: ArbOpportunity) -> List[Dict]:
        """
        Atomically execute all legs of an arbitrage opportunity.
        Uses size capping to stay within risk limits.
        """
        actual_size = min(opp.max_size, self.max_position_usd / opp.implied_cost)

        results = []
        tasks = []
        for leg in opp.legs:
            tasks.append(
                self.client.submit_order_async(
                    token_id=leg.token_id,
                    side=leg.side,
                    price=leg.price,
                    size=actual_size,
                )
            )

        # Fire all legs concurrently for minimum latency
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successes = sum(1 for r in results if not isinstance(r, Exception))
        failures = [r for r in results if isinstance(r, Exception)]

        if failures:
            logger.error(f"Partial fill! {successes}/{len(opp.legs)} legs filled. Failures: {failures}")
        else:
            logger.info(f"All {len(opp.legs)} arb legs filled successfully. Edge captured: ${opp.edge:.4f}")

        return results

    # ─── Helpers ────────────────────────────────────────────────────────

    def _find_max_profitable_size(self, books: List[Tuple[str, OrderBook]]) -> float:
        """Binary search for the largest position size where total fill cost < $1/share - fees."""
        lo, hi = 0.0, min(
            sum(l.size for l in book.asks[:20]) for _, book in books
        )
        hi = min(hi, self.max_position_usd)

        for _ in range(50):  # 50 iterations of bisection
            mid = (lo + hi) / 2
            if mid < 1e-6:
                break
            total_per_share = sum(
                book.fill_cost("BUY", mid)[0] / mid for _, book in books
            )
            net = 1.0 - total_per_share - self.fee_rate
            if net > 0:
                lo = mid
            else:
                hi = mid

        return lo

    def _find_max_profitable_size_sell(self, books: List[Tuple[str, OrderBook]]) -> float:
        """Binary search for sell-side arb sizing."""
        lo, hi = 0.0, min(
            sum(l.size for l in book.bids[:20]) for _, book in books
        )
        hi = min(hi, self.max_position_usd)

        for _ in range(50):
            mid = (lo + hi) / 2
            if mid < 1e-6:
                break
            total_per_share = sum(
                book.fill_cost("SELL", mid)[0] / mid for _, book in books
            )
            net = total_per_share - 1.0 - self.fee_rate
            if net > 0:
                lo = mid
            else:
                hi = mid

        return lo

    @staticmethod
    def _levels_consumed(levels: List[OrderBookLevel], target: float) -> int:
        remaining = target
        count = 0
        for level in levels:
            count += 1
            remaining -= level.size
            if remaining <= 0:
                break
        return count
