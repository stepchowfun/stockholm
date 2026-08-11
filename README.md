# Stockholm

[![Build status](https://github.com/stepchowfun/stockholm/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/stepchowfun/stockholm/actions?query=branch%3Amain)

*Stockholm* is a laboratory for algorithmic trading.

## Usage

Once Stockholm is [installed](#installation-instructions), you can run it from the command line as follows:

```sh
stockholm
```

Here are the supported command-line options:

```
Usage: stockholm [OPTIONS] [COMMAND]

Commands:
  backtest    Backtest a trading strategy
  run         Run the trading bot (default)
  historical  Fetch historical market data as CSV
  infer       Run one inference with a trained neural network
  train       Train a neural network on historical stock data
  help        Print this message or the help of the given subcommand(s)

Options:
      --address <ADDRESS>      Address of the running TWS or IB Gateway API [default: 127.0.0.1:4001]
      --client-id <CLIENT_ID>  Client ID to use for the API connection [default: 100]
  -v, --version                Print version
  -h, --help                   Print help
```

You can evaluate a strategy over historical CSV files with the `backtest` subcommand. Every strategy excludes records from 4:00 a.m. through 4:14:59 a.m. Eastern time because delayed overnight trade reports make that window unreliable for backtesting. Files are ordered lexicographically by filename; the buy-and-hold strategy invests the initial cash at the first remaining opening price and prints the resulting account value at the final closing price:

```sh
stockholm backtest --strategy buy-and-hold --data-paths data/validation/*.csv
```

The `market-maker` strategy treats each input CSV as one trading day and repeatedly places whole-share limit orders below and above each one-second bar's closing price. Existing orders fill when a subsequent bar trades through their limit and expire after their configured lifetime. Each eligible order fills at most `--bar-volume-limit` shares per one-second bar (1,000 by default), with the remainder staying open. During the final rows of each day selected by `--liquidation-seconds` (900 by default), the strategy cancels all buy and sell orders, places no new ordinary orders, and liquidates up to the same volume limit at the current closing price each second. Account value compounds across days. The reported CSV includes the final marked-to-market account value and the unannualized Sharpe ratio calculated as mean daily return divided by its population standard deviation:

```sh
stockholm backtest --strategy market-maker --data-paths data/validation/*.csv --initial-cash 1000000 --buy-ttl 3600 --sell-ttl 14400 --discount-percent 0.25 --markup-percent 0.25
```

The `market-maker-grid` strategy evaluates buy and sell TTLs of 5, 15, 30, 60, 120, 300, 900, 3,600, 7,200, 14,400, 43,200, and 86,400 seconds, discount and markup percentages of 0.01, 0.03, 0.1, 0.3, 1, 3, and 10, and bet sizes of 80, 90, and 100 percent by default. It prints separate CSV records for the configurations with the highest final return and highest Sharpe ratio among the resulting 21,168 combinations, including the daily return series for each winner. Override each part of the search space independently with comma-separated `--buy-ttls`, `--sell-ttls`, `--discount-percentages`, `--markup-percentages`, and `--bet-sizes` values. Search progress is written to standard error, leaving standard output as machine-readable CSV. Initial cash and liquidation duration remain configurable, while the single-run TTL and percentage options do not affect grid mode:

```sh
stockholm backtest --strategy market-maker-grid --data-paths data/training/*.csv
```

You can select the symbol whose live market data and five-second bars should be streamed with the `run` subcommand:

```sh
stockholm run --symbol AAPL
```

The live strategy runs its control loop once per second. Between 7 p.m. and 8:05 p.m. Eastern time, it suppresses ordinary market-making orders, immediately cancels its buy orders for the selected symbol, gives matching sell orders a five-minute lifetime, and repeatedly offers every unreserved whole share at the current ask. Stockholm-managed orders for other symbols and orders placed outside Stockholm are unaffected.

You can fetch historical bars as CSV with the `historical` subcommand. The start and end are ISO 8601 datetimes with explicit UTC offsets, and the interval defaults to `SEC`:

```sh
stockholm historical --start 2026-07-31T13:30:00Z --end 2026-07-31T20:00:00Z --symbol SOXL --interval MIN5
```

The `train` subcommand trains a simple neural network to predict multiple future samples from a fixed window of historical stock prices. Each input must be a CSV file produced by the `historical` subcommand; only its `open` column is used. Training and validation files are specified separately, and every file is treated as an independent time series so windows never span file boundaries:

```sh
stockholm train --training-paths monday.csv tuesday.csv --validation-paths wednesday.csv
```

Training uses log returns and every available overlapping window. Normalization is fitted exclusively from the training files. By default, the model uses 20 inputs to predict 5 outputs, a batch size of 64, 50 epochs, a learning rate of 0.001, and a seed of 42, and writes the trained Burn model and its preprocessing metadata to `model`; these settings can be changed with `--inputs`, `--outputs`, `--batch-size`, `--epochs`, `--learning-rate`, `--seed`, and `--model-directory`.

The `infer` subcommand loads those artifacts and forecasts opening prices from one CSV input window. Because prices are converted to log returns, the CSV must contain one more raw price than the model's input count; the default 20-input model therefore consumes 21 prices:

```sh
stockholm infer
```

You'll also need to start an Interactive Brokers Gateway that Stockholm can talk to. You can run that in a Docker container via the provided `run-ib-gateway.sh` script:

```sh
./run-ib-gateway.sh
```

The script expects the Interactive Brokers username to be in a file called `$HOME/ib-username`, and it looks for the password in `$HOME/ib-password`.

You probably want to run Stockholm and IB Gateway as daemons, e.g., with [launchd](https://www.launchd.info/) or [systemd](https://www.freedesktop.org/wiki/Software/systemd/). See the [Configuring your operating system to run Stockholm as a daemon](#configuring-your-operating-system-to-run-stockholm-as-a-daemon) section below for instructions.

For debugging, you can connect to the VNC server at `vnc://127.0.0.1:5900` using the default password `vnc_password`. In macOS, you can do this from Finder with Cmd+k.

Stockholm assumes a *US Equity and Options Add-On Streaming Bundle (NP)* market data subscription.

## Installation instructions

### Installation on macOS or Linux (AArch64 or x86-64)

If you're running macOS or Linux (AArch64 or x86-64), you can install Stockholm with this command:

```sh
curl https://raw.githubusercontent.com/stepchowfun/stockholm/main/install.sh -LSfs | sh
```

The same command can be used again to update to the latest version.

The installation script supports the following optional environment variables:

- `VERSION=x.y.z` (defaults to the latest version)
- `PREFIX=/path/to/install` (defaults to `/usr/local/bin`)

For example, the following will install Stockholm into the working directory:

```sh
curl https://raw.githubusercontent.com/stepchowfun/stockholm/main/install.sh -LSfs | PREFIX=. sh
```

If you prefer not to use this installation method, you can download the binary from the [releases page](https://github.com/stepchowfun/stockholm/releases), make it executable (e.g., with `chmod`), and place it in some directory in your [`PATH`](https://en.wikipedia.org/wiki/PATH_\(variable\)) (e.g., `/usr/local/bin`).

### Installation on Windows (AArch64 or x86-64)

If you're running Windows (AArch64 or x86-64), download the latest binary from the [releases page](https://github.com/stepchowfun/stockholm/releases) and rename it to `stockholm` (or `stockholm.exe` if you have file extensions visible). Create a directory called `Stockholm` in your `%PROGRAMFILES%` directory (e.g., `C:\Program Files\Stockholm`), and place the renamed binary in there. Then, in the "Advanced" tab of the "System Properties" section of Control Panel, click on "Environment Variables..." and add the full path to the new `Stockholm` directory to the `PATH` variable under "System variables". Note that the `Program Files` directory might have a different name if Windows is configured for a language other than English.

To update an existing installation, simply replace the existing binary.

### Installation with Cargo

If you have [Cargo](https://doc.rust-lang.org/cargo/), you can install Stockholm as follows:

```sh
cargo install stockholm
```

You can run that command with `--force` to update an existing installation.

### Configuring your operating system to run Stockholm as a daemon

Stockholm depends on IB Gateway, so the repository provides sample service definitions for running both programs as daemons. Adjust the paths, arguments, and other settings in these files as needed before installing them.

#### Creating launchd services on macOS

On macOS, [launchd](https://www.launchd.info/) can be used to run Stockholm and IB Gateway as daemons. Copy [`local.stockholm.plist`](service_configs/local.stockholm.plist) and [`local.ib-gateway.plist`](service_configs/local.ib-gateway.plist) from the `service_configs` directory to `/Library/LaunchDaemons/`, copy [`run-ib-gateway.sh`](run-ib-gateway.sh) to `/usr/local/bin/`, and make sure all three files are owned by root.

Run the following commands to start the services:

```sh
sudo launchctl load /Library/LaunchDaemons/local.ib-gateway.plist
sudo launchctl load /Library/LaunchDaemons/local.stockholm.plist
```

You can view the logs with `tail -F /var/log/ib-gateway.log /var/log/stockholm.log`.

#### Creating systemd services on Linux

On most Linux distributions, [systemd](https://www.freedesktop.org/wiki/Software/systemd/) can be used to run Stockholm and IB Gateway as daemons. Copy [`stockholm.service`](service_configs/stockholm.service) and [`ib-gateway.service`](service_configs/ib-gateway.service) from the `service_configs` directory to `/etc/systemd/system/`, copy [`run-ib-gateway.sh`](run-ib-gateway.sh) to `/usr/local/bin/`, and make sure all three files are owned by root.

Run `sudo systemctl enable ib-gateway stockholm --now` to enable and start the services. You can view the logs with `sudo journalctl --follow --unit ib-gateway --unit stockholm`.
