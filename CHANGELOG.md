# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
