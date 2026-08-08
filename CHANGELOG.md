# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.26] - 2026-08-08

### Added
- Track and print the funds available for opening new positions during live runs.

### Changed
- Match order status updates to managed orders exclusively by stable permanent IDs.

## [0.2.25] - 2026-08-07

### Added
- Log full, partial, and liquidation executions for individual market-maker backtests with source filenames and Unix timestamps.

## [0.2.24] - 2026-08-07

### Changed
- Reconcile open orders once per Interactive Brokers connection and rely on streaming updates thereafter.

## [0.2.23] - 2026-08-07

### Fixed
- Prefer stable permanent IDs when matching order status updates to persisted orders.

## [0.2.22] - 2026-08-07

### Added
- Persist and reconcile the open orders managed by Stockholm.

### Changed
- Load persisted run state once before reconnecting to Interactive Brokers.
- Organize concurrent live-run functions consistently.

## [0.2.21] - 2026-08-07

### Added
- Stream and log order updates during live runs.
- Provide reusable limit buy and sell helpers with extended and overnight session support.

### Changed
- Tag Stockholm orders with UUID references and list only Stockholm open orders.

## [0.2.20] - 2026-08-07

### Added
- Track and print the latest positive bid and ask prices during live runs.

### Changed
- Request open orders and positions concurrently.
- Separate completed grid-search progress from the winning configurations.

## [0.2.19] - 2026-08-07

### Changed
- Print human-readable market-maker reports with daily returns and complete configurations.

## [0.2.18] - 2026-08-07

### Added
- Calculate the market-maker Sharpe ratio from daily returns.
- Report separate market-maker grid winners for total return and Sharpe ratio.

### Changed
- Simulate market-maker backtests one day at a time and print the effective configuration.

## [0.2.17] - 2026-08-06

### Added
- Persist extensible run state in the platform-specific local data directory.
- Configure the share of account value available to market-maker buy orders and search it in grid backtests.

### Changed
- Report final account values from buy-and-hold and market-maker backtests.

## [0.2.16] - 2026-08-06

### Changed
- Simulate partial market-maker fills with a configurable per-bar share limit.

## [0.2.15] - 2026-08-06

### Added
- Backtest a configurable limit-order market-making strategy.
- Search a configurable grid of market-maker parameters for the most profitable combination with progress reporting.

### Changed
- Liquidate market-maker inventory during the final fifteen minutes of each trading day.

## [0.2.14] - 2026-08-06

### Added
- Backtest a buy-and-hold strategy over filename-ordered historical CSV files.
- Bundle example market data with training and validation splits.
- Run a single trained-model inference from a CSV price window with the `infer` subcommand.

## [0.2.13] - 2026-08-05

### Changed
- Report per-epoch training and validation RMSE in basis points.
- Validate short training inputs alongside the requested model window dimensions.

## [0.2.12] - 2026-08-04

### Changed
- Read opening prices for training and validation from separate groups of independent historical CSV files.
- Configure the training batch size, epoch count, learning rate, and random seed.

## [0.2.11] - 2026-08-04

### Added
- Train a Burn neural network on historical stock prices with configurable input and output window sizes.

## [0.2.10] - 2026-08-03

### Added
- Select the historical bar size with the `historical` subcommand's `--interval` option.

## [0.2.9] - 2026-08-03

### Added
- Select the live market data symbol with the run subcommand's `--symbol` option.

## [0.2.8] - 2026-08-03

### Added
- Stream real-time five-second bars from the SMART and OVERNIGHT venues alongside live market data in the run command.

## [0.2.7] - 2026-08-02

### Fixed
- Pad short historical data requests to Interactive Brokers' minimum supported duration.

## [0.2.6] - 2026-08-02

### Added
- Fetch historical one-second bars as CSV for an explicit datetime range.
- List current positions alongside open orders.

### Changed
- Prefix order and market data output to identify its source.
- Organize the run and historical commands into separate modules.

## [0.2.5] - 2026-07-31

### Added
- List open orders once per second while concurrently streaming real-time SOXL market data.

### Changed
- Reuse each Interactive Brokers connection until an order request or market data stream fails.

## [0.2.4] - 2026-07-29

### Fixed
- Propagate market data stream failures so the top-level loop reconnects.

## [0.2.3] - 2026-07-29

### Changed
- Stream real-time SOXL market data instead of periodically requesting AAPL snapshots.

## [0.2.2] - 2026-07-27

### Changed
- Propagate snapshot failures so the top-level loop reconnects after every attempt.

### Fixed
- Stop the IB Gateway container when its wrapper exits or starts a replacement instance.

## [0.2.1] - 2026-07-27

### Fixed
- Retry top-level application failures after a ten-second delay.

## [0.2.0] - 2026-07-27

### Changed
- Log an AAPL market data snapshot every minute and continue after individual snapshot failures.

## [0.1.0] - 2026-07-26

### Added
- Print the latest available delayed AAPL trade price from Interactive Brokers Gateway.
- Support custom Gateway addresses and API client IDs.

## [0.0.0] - 2026-07-25

### Added
- Initial project scaffold.
