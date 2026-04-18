"""
Polymarket CLOB Client — Production-grade websocket streaming and REST order execution.

Handles:
  - Real-time L2 order book streaming via Polymarket websocket subscriptions
  - Signed order creation and submission to the CLOB endpoint
  - In-memory order book reconstruction from delta messages
  - Tick-by-tick history buffer for downstream model consumption
"""

import asyncio
import json
import logging
import time
import os
import hmac
import hashlib
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import requests
import websockets
import numpy as np


logger = logging.getLogger("PolymarketClient")

# ─── Configuration ──────────────────────────────────────────────────────────────

CLOB_REST_BASE = "https://clob.polymarket.com"
WS_BASE = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
GAMMA_REST_BASE = "https://gamma-api.polymarket.com"


@dataclass
class OrderBookLevel:
    price: float
    size: float


@dataclass
class OrderBook:
    """Full L2 state for a single token."""
    token_id: str
    bids: List[OrderBookLevel] = field(default_factory=list)   # sorted desc by price
    asks: List[OrderBookLevel] = field(default_factory=list)   # sorted asc by price
    timestamp: float = 0.0

    @property
    def best_bid(self) -> Optional[OrderBookLevel]:
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self) -> Optional[OrderBookLevel]:
        return self.asks[0] if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return (self.best_bid.price + self.best_ask.price) / 2.0
        return None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid and self.best_ask:
            return self.best_ask.price - self.best_bid.price
        return None

    def depth_at(self, price_levels: int = 5) -> Dict:
        """Return aggregated depth for top-N levels on each side."""
        bid_depth = sum(l.size for l in self.bids[:price_levels])
        ask_depth = sum(l.size for l in self.asks[:price_levels])
        return {"bid_depth": bid_depth, "ask_depth": ask_depth, "imbalance": bid_depth / (bid_depth + ask_depth + 1e-12)}

    def fill_cost(self, side: str, target_size: float) -> Tuple[float, float]:
        """
        Walk the book to compute realized cost and average fill price
        for a market order of `target_size`.
        Returns (total_cost, avg_price).
        """
        levels = self.asks if side == "BUY" else self.bids
        remaining = target_size
        total_cost = 0.0
        filled = 0.0

        for level in levels:
            take = min(remaining, level.size)
            total_cost += take * level.price
            filled += take
            remaining -= take
            if remaining <= 0:
                break

        avg_price = total_cost / filled if filled > 0 else 0.0
        return total_cost, avg_price


@dataclass
class TickRecord:
    """Single tick snapshot for model ingestion."""
    timestamp: float
    bid_px: float
    bid_sz: float
    ask_px: float
    ask_sz: float
    mid: float
    spread: float
    bid_depth_5: float
    ask_depth_5: float
    imbalance: float


class PolymarketClient:
    """
    Asynchronous Polymarket CLOB client with:
      1. Websocket-based order book streaming with delta application
      2. REST-based order submission
      3. Tick history ring buffer for ML model consumption
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        api_passphrase: Optional[str] = None,
        ws_url: str = WS_BASE,
        rest_url: str = CLOB_REST_BASE,
        tick_buffer_size: int = 100_000,
    ):
        self.api_key = api_key or os.environ.get("POLY_API_KEY", "")
        self.api_secret = api_secret or os.environ.get("POLY_API_SECRET", "")
        self.api_passphrase = api_passphrase or os.environ.get("POLY_API_PASSPHRASE", "")
        self.ws_url = ws_url
        self.rest_url = rest_url

        # In-memory order books keyed by token_id
        self.books: Dict[str, OrderBook] = {}

        # Ring buffer of tick records per token
        self.tick_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=tick_buffer_size))

        # Websocket handle
        self._ws = None
        self._running = False
        self._reconnect_delay = 1.0

    # ─── Authentication ─────────────────────────────────────────────────

    def _sign_request(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        timestamp = str(int(time.time()))
        message = timestamp + method.upper() + path + body
        signature = hmac.new(
            self.api_secret.encode(), message.encode(), hashlib.sha256
        ).hexdigest()
        return {
            "POLY-API-KEY": self.api_key,
            "POLY-SIGNATURE": signature,
            "POLY-TIMESTAMP": timestamp,
            "POLY-PASSPHRASE": self.api_passphrase,
            "Content-Type": "application/json",
        }

    # ─── REST: Market Discovery ─────────────────────────────────────────

    def fetch_markets(self, limit: int = 100, active: bool = True) -> List[Dict]:
        """Fetch active markets from the Gamma API."""
        params = {"limit": limit, "active": str(active).lower(), "closed": "false"}
        resp = requests.get(f"{GAMMA_REST_BASE}/markets", params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()

    def fetch_condition_tokens(self, condition_id: str) -> List[Dict]:
        """Fetch all tokens under a condition (mutually exclusive group)."""
        resp = requests.get(
            f"{GAMMA_REST_BASE}/markets",
            params={"condition_id": condition_id},
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    # ─── REST: Order Book Snapshot ──────────────────────────────────────

    def fetch_orderbook_snapshot(self, token_id: str) -> OrderBook:
        """Pull a full L2 snapshot and build an OrderBook in memory."""
        resp = requests.get(
            f"{self.rest_url}/book",
            params={"token_id": token_id},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        book = OrderBook(token_id=token_id, timestamp=time.time())
        for bid in data.get("bids", []):
            book.bids.append(OrderBookLevel(price=float(bid["price"]), size=float(bid["size"])))
        for ask in data.get("asks", []):
            book.asks.append(OrderBookLevel(price=float(ask["price"]), size=float(ask["size"])))

        book.bids.sort(key=lambda l: l.price, reverse=True)
        book.asks.sort(key=lambda l: l.price)
        self.books[token_id] = book
        return book

    # ─── REST: Order Submission ─────────────────────────────────────────

    def submit_order(self, token_id: str, side: str, price: float, size: float) -> Dict:
        """Submit a limit order to Polymarket CLOB."""
        path = "/order"
        body = json.dumps({
            "tokenID": token_id,
            "side": side.upper(),
            "price": str(round(price, 4)),
            "size": str(round(size, 2)),
            "type": "GTC",
        })
        headers = self._sign_request("POST", path, body)
        resp = requests.post(f"{self.rest_url}{path}", headers=headers, data=body, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        logger.info(f"Order submitted: {side} {size}@{price} for {token_id} → {result}")
        return result

    async def submit_order_async(self, token_id: str, side: str, price: float, size: float) -> Dict:
        """Non-blocking order submission via event loop executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.submit_order, token_id, side, price, size)

    # ─── Websocket: Streaming ───────────────────────────────────────────

    async def connect_and_listen(self, token_ids: List[str]):
        """
        Open persistent websocket, subscribe to order book channels,
        and apply deltas to in-memory books indefinitely.
        """
        self._running = True

        # Seed books with REST snapshots first
        for tid in token_ids:
            try:
                self.fetch_orderbook_snapshot(tid)
                logger.info(f"Seeded L2 snapshot for {tid}")
            except Exception as e:
                logger.warning(f"Failed to seed snapshot for {tid}: {e}")
                self.books[tid] = OrderBook(token_id=tid)

        while self._running:
            try:
                async with websockets.connect(self.ws_url, ping_interval=20, ping_timeout=10) as ws:
                    self._ws = ws
                    self._reconnect_delay = 1.0
                    logger.info("Websocket connected.")

                    # Subscribe to each market
                    for tid in token_ids:
                        sub_msg = json.dumps({
                            "type": "subscribe",
                            "channel": "book",
                            "markets": [tid],
                        })
                        await ws.send(sub_msg)

                    # Message processing loop
                    async for raw in ws:
                        msg = json.loads(raw)
                        self._process_ws_message(msg)

            except (websockets.ConnectionClosed, ConnectionError, OSError) as e:
                logger.warning(f"WS disconnected: {e}. Reconnecting in {self._reconnect_delay}s...")
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 30.0)
            except Exception as e:
                logger.error(f"WS fatal error: {e}", exc_info=True)
                break

    def _process_ws_message(self, msg: Dict):
        """Apply incoming delta to in-memory order book and record a tick."""
        msg_type = msg.get("type", "")
        if msg_type not in ("book_update", "book_snapshot", "price_change"):
            return

        token_id = msg.get("market", msg.get("asset_id", ""))
        if not token_id or token_id not in self.books:
            return

        book = self.books[token_id]
        now = time.time()

        if msg_type == "book_snapshot":
            # Full replacement
            book.bids = [OrderBookLevel(float(b["price"]), float(b["size"])) for b in msg.get("bids", [])]
            book.asks = [OrderBookLevel(float(a["price"]), float(a["size"])) for a in msg.get("asks", [])]
        else:
            # Incremental delta
            for change in msg.get("changes", []):
                side = change.get("side", "").lower()
                price = float(change["price"])
                size = float(change["size"])
                levels = book.bids if side == "bid" else book.asks

                # Remove existing level at this price
                levels[:] = [l for l in levels if abs(l.price - price) > 1e-9]

                # If size > 0, add back
                if size > 0:
                    levels.append(OrderBookLevel(price, size))

            book.bids.sort(key=lambda l: l.price, reverse=True)
            book.asks.sort(key=lambda l: l.price)

        book.timestamp = now

        # Record tick
        depth = book.depth_at(5)
        mid = book.mid_price or 0.0
        spread = book.spread or 0.0
        bb = book.best_bid
        ba = book.best_ask
        self.tick_history[token_id].append(TickRecord(
            timestamp=now,
            bid_px=bb.price if bb else 0.0,
            bid_sz=bb.size if bb else 0.0,
            ask_px=ba.price if ba else 0.0,
            ask_sz=ba.size if ba else 0.0,
            mid=mid,
            spread=spread,
            bid_depth_5=depth["bid_depth"],
            ask_depth_5=depth["ask_depth"],
            imbalance=depth["imbalance"],
        ))

    # ─── Data Export for ML Models ──────────────────────────────────────

    def get_tick_matrix(self, token_id: str, lookback: int = 1000) -> np.ndarray:
        """
        Return last `lookback` ticks as a (lookback, 10) numpy matrix,
        ready for ingestion into the Mamba architecture.
        Columns: [timestamp, bid_px, bid_sz, ask_px, ask_sz, mid, spread,
                   bid_depth_5, ask_depth_5, imbalance]
        """
        history = self.tick_history.get(token_id)
        if not history or len(history) == 0:
            return np.zeros((lookback, 10))

        rows = []
        start = max(0, len(history) - lookback)
        for i in range(start, len(history)):
            t = history[i]
            rows.append([
                t.timestamp, t.bid_px, t.bid_sz, t.ask_px,
                t.ask_sz, t.mid, t.spread, t.bid_depth_5,
                t.ask_depth_5, t.imbalance,
            ])

        mat = np.array(rows, dtype=np.float64)
        # Pad to `lookback` if needed
        if mat.shape[0] < lookback:
            pad = np.zeros((lookback - mat.shape[0], 10))
            mat = np.vstack([pad, mat])
        return mat

    def get_orderbook(self, token_id: str) -> OrderBook:
        """Return in-memory L2 order book for a token."""
        if token_id not in self.books:
            return self.fetch_orderbook_snapshot(token_id)
        return self.books[token_id]

    def stop(self):
        self._running = False
