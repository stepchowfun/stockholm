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
  run         Run the trading bot (default)
  historical  Fetch historical market data as CSV
  train       Train a neural network on historical stock data
  help        Print this message or the help of the given subcommand(s)

Options:
      --address <ADDRESS>      Address of the running TWS or IB Gateway API [default: 127.0.0.1:4001]
      --client-id <CLIENT_ID>  Client ID to use for the API connection [default: 100]
  -v, --version                Print version
  -h, --help                   Print help
```

You can select the symbol whose live market data and five-second bars should be streamed with the `run` subcommand:

```sh
stockholm run --symbol AAPL
```

You can fetch historical bars as CSV with the `historical` subcommand. The start and end are ISO 8601 datetimes with explicit UTC offsets, and the interval defaults to `SEC`:

```sh
stockholm historical --start 2026-07-31T13:30:00Z --end 2026-07-31T20:00:00Z --symbol SOXL --interval MIN5
```

The `train` subcommand trains a simple neural network to predict multiple future samples from a fixed window of historical stock prices. Each input must be a CSV file produced by the `historical` subcommand; only its `open` column is used. Training and validation files are specified separately, and every file is treated as an independent time series so windows never span file boundaries:

```sh
stockholm train --training-paths monday.csv tuesday.csv --validation-paths wednesday.csv
```

Training uses log returns and every available overlapping window. Normalization is fitted exclusively from the training files. By default, the model uses 20 inputs to predict 5 outputs, a batch size of 64, 50 epochs, a learning rate of 0.001, and a seed of 42, and writes the trained Burn model and its preprocessing metadata to `model`; these settings can be changed with `--inputs`, `--outputs`, `--batch-size`, `--epochs`, `--learning-rate`, `--seed`, and `--model-directory`.

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
