"""Simple execution simulator for event-driven research."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ExecutionConfig:
    """Execution assumptions for converting target orders into fills."""

    slippage_bps: float = 1.0
    commission_bps: float = 2.0
    max_participation_rate: float = 0.10


@dataclass(frozen=True)
class Order:
    """Target order in shares."""

    timestamp: pd.Timestamp
    symbol: str
    quantity: float
    side: str


@dataclass(frozen=True)
class Fill:
    """Simulated order fill."""

    timestamp: pd.Timestamp
    symbol: str
    quantity: float
    price: float
    commission: float
    slippage: float


class ExecutionSimulator:
    """Convert target share orders into cost-aware fills."""

    def __init__(self, config: ExecutionConfig | None = None) -> None:
        self.config = config or ExecutionConfig()

    def simulate_orders(
        self,
        orders: list[Order],
        prices: pd.DataFrame,
        volumes: pd.DataFrame | None = None,
    ) -> list[Fill]:
        """Simulate fills using close prices, slippage, commissions, and volume caps."""

        fills: list[Fill] = []
        for order in orders:
            if order.symbol not in prices.columns or order.timestamp not in prices.index:
                continue
            price = float(prices.loc[order.timestamp, order.symbol])
            quantity = float(order.quantity)
            if volumes is not None and order.symbol in volumes.columns:
                available = float(volumes.loc[order.timestamp, order.symbol])
                cap = abs(available * self.config.max_participation_rate)
                quantity = max(-cap, min(cap, quantity))
            slippage = price * abs(quantity) * self.config.slippage_bps / 10_000.0
            commission = price * abs(quantity) * self.config.commission_bps / 10_000.0
            signed_slippage_price = price * (1.0 + self.config.slippage_bps / 10_000.0)
            if order.side.lower() == "sell":
                signed_slippage_price = price * (1.0 - self.config.slippage_bps / 10_000.0)
            fills.append(
                Fill(
                    timestamp=order.timestamp,
                    symbol=order.symbol,
                    quantity=quantity,
                    price=signed_slippage_price,
                    commission=commission,
                    slippage=slippage,
                )
            )
        return fills

