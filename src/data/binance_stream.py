"""
Binance WebSocket client for streaming real-time Limit Order Book (LOB) data.

This module provides async WebSocket clients for connecting to Binance's public
market data streams. It handles automatic reconnection, connection management,
and provides callbacks for processing incoming order book updates.

Example:
    Basic usage with a single symbol::

        async def process_data(data):
            print(f"Received update for {data['s']}")

        stream = BinanceDepthStream('BTCUSDT', process_data)
        await stream.start()
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Callable

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger(__name__)


class BinanceDepthStream:
    """
    Async WebSocket client for Binance order book depth streams.

    Connects to Binance's depth@<levels> stream to receive real-time
    order book updates with configurable depth levels.

    Args:
        symbol (str): Trading pair symbol (e.g., 'BTCUSDT').
        callback (Callable): Async callback function to process LOB updates.
        levels (int, optional): Order book depth (5, 10, or 20). Defaults to 5.
        use_testnet (bool, optional): Use Binance testnet. Defaults to False.

    Attributes:
        symbol (str): Lowercase trading pair symbol.
        levels (int): Validated order book depth.
        callback (Callable): User-provided message handler.
        ws_url (str): Complete WebSocket URL.
    """

    BINANCE_WS_BASE = "wss://stream.binance.com:443/ws"
    BINANCE_TESTNET_WS_BASE = "wss://testnet.binance.vision/ws"

    def __init__(
        self,
        symbol: str,
        callback: Callable[[dict[str, Any]], Any],
        levels: int = 5,
        use_testnet: bool = False,
    ):
        """Initialize Binance depth stream."""
        self.symbol = symbol.lower()
        self.symbol_upper = symbol.upper()
        self.levels = self._validate_levels(levels)
        self.callback = callback
        self.use_testnet = use_testnet
        self._stop_event = asyncio.Event()

        # Construct WebSocket URL
        base_url = self.BINANCE_TESTNET_WS_BASE if use_testnet else self.BINANCE_WS_BASE
        stream_name = f"{self.symbol}@depth{self.levels}"
        self.ws_url = f"{base_url}/{stream_name}"

        logger.info(
            f"Initialized BinanceDepthStream for {symbol.upper()} "
            f"with {levels} levels (testnet={use_testnet})"
        )

    @staticmethod
    def _validate_levels(levels: int) -> int:
        """Validate order book depth levels."""
        valid_levels = [5, 10, 20]
        if levels not in valid_levels:
            raise ValueError(f"Invalid depth levels {levels}. Must be one of {valid_levels}.")
        return levels

    async def _process_message(self, message: str) -> None:
        """Parse and handle incoming WebSocket message."""
        try:
            data = json.loads(message)

            # Add symbol if not present (snapshot streams don't include it)
            if "s" not in data:
                data["s"] = self.symbol_upper

            # Add timestamps
            data["local_time"] = datetime.now(timezone.utc)

            # Call user callback
            if asyncio.iscoroutinefunction(self.callback):
                await self.callback(data)
            else:
                self.callback(data)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
        except Exception as e:
            logger.error(f"Error in callback: {e}", exc_info=True)

    async def start(self) -> None:
        """
        Start listening to WebSocket messages.

        This method runs until stop() is called or an error occurs.
        Automatically reconnects on disconnection.
        """
        reconnect_delay = 1
        max_delay = 60

        while not self._stop_event.is_set():
            try:
                logger.info(f"Connecting to {self.ws_url}")

                async with websockets.connect(
                    self.ws_url,
                    ping_interval=20,
                    ping_timeout=10,
                ) as websocket:
                    logger.info(f"Connected to Binance for {self.symbol.upper()}")
                    reconnect_delay = 1  # Reset on successful connection

                    # Receive messages until disconnection or stop
                    while not self._stop_event.is_set():
                        try:
                            message = await asyncio.wait_for(
                                websocket.recv(), timeout=1.0  # Check stop event every second
                            )
                            await self._process_message(message)
                        except asyncio.TimeoutError:
                            continue  # No message, check stop event

            except (ConnectionClosed, WebSocketException) as e:
                if self._stop_event.is_set():
                    break
                logger.warning(f"Connection lost: {e}. Reconnecting in {reconnect_delay}s...")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_delay)

            except Exception as e:
                if self._stop_event.is_set():
                    break
                logger.error(f"Unexpected error: {e}", exc_info=True)
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, max_delay)

        logger.info(f"Stream stopped for {self.symbol.upper()}")

    async def stop(self) -> None:
        """Stop the WebSocket stream gracefully."""
        logger.info(f"Stopping stream for {self.symbol.upper()}")
        self._stop_event.set()


class BinanceMultiStream:
    """
    Manage multiple Binance depth streams concurrently.

    Args:
        symbols (list[str]): List of trading pairs.
        callback (Callable): Async callback for all streams.
        levels (int, optional): Order book depth. Defaults to 5.
        use_testnet (bool, optional): Use testnet. Defaults to False.
    """

    def __init__(
        self,
        symbols: list[str],
        callback: Callable[[dict[str, Any]], Any],
        levels: int = 5,
        use_testnet: bool = False,
    ):
        """Initialize multi-stream manager."""
        self.streams = [
            BinanceDepthStream(symbol, callback, levels, use_testnet) for symbol in symbols
        ]
        logger.info(f"Initialized multi-stream for {len(symbols)} symbols")

    async def start(self) -> None:
        """Start all streams concurrently."""
        tasks = [stream.start() for stream in self.streams]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def stop(self) -> None:
        """Stop all streams."""
        for stream in self.streams:
            await stream.stop()


# =============================================================================
# Example Usage & Testing
# =============================================================================


async def example_callback(data: dict[str, Any]) -> None:
    """Example callback that prints LOB updates."""
    symbol = data.get("s", "UNKNOWN")
    event_time = data.get("local_time", datetime.now(timezone.utc))

    bids = data.get("bids", [])
    asks = data.get("asks", [])

    if bids and asks:
        logger.info(f"[{symbol}] @ {event_time.strftime('%H:%M:%S.%f')[:-3]}")
        logger.info(f"  Best Bid: ${bids[0][0]} (qty: {bids[0][1]})")
        logger.info(f"  Best Ask: ${asks[0][0]} (qty: {asks[0][1]})")

        # Calculate spread
        spread = float(asks[0][0]) - float(bids[0][0])
        spread_bps = (spread / float(bids[0][0])) * 10000
        logger.info(f"  Spread: ${spread:.2f} ({spread_bps:.2f} bps)")
        logger.info("---")


async def main() -> None:
    """Test the Binance WebSocket client."""
    from config.logging_config import setup_logging

    setup_logging(log_level="INFO")

    logger.info("=== Testing Binance WebSocket Stream ===")
    stream = BinanceDepthStream(
        symbol="BTCUSDT",
        callback=example_callback,
        levels=settings.lob_depth_levels,
        use_testnet=settings.binance_testnet,
    )

    # Start stream
    task = asyncio.create_task(stream.start())

    try:
        logger.info("Streaming for 30 seconds... (Ctrl+C to stop)")
        await asyncio.sleep(30)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await stream.stop()
        await task

    logger.info("Test completed")


if __name__ == "__main__":
    asyncio.run(main())
