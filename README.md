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
Usage: stockholm [OPTIONS]

Options:
      --address <ADDRESS>      Address of the running TWS or IB Gateway API [default: 127.0.0.1:4001]
      --client-id <CLIENT_ID>  Client ID to use for the API connection [default: 100]
  -v, --version                Print version
  -h, --help                   Print help
```

You'll also need to start an Interactive Brokers Gateway that Stockholm can talk to. You can do that via:

```sh
docker container run \
  --env 'READ_ONLY_API=no' \
  --env 'TRADING_MODE=live' \
  --env 'TWS_USERID=your_ibkr_username' \
  --env 'TWS_PASSWORD=your_ibkr_password' \
  --env 'TWOFA_TIMEOUT_ACTION=exit' \
  --env 'RELOGIN_AFTER_TWOFA_TIMEOUT=no' \
  --env 'TWOFA_EXIT_INTERVAL=60' \
  --publish 127.0.0.1:4001:4003 \
  ghcr.io/gnzsnz/ib-gateway@sha256:8b1106efea6c27c14d1a53c881e149a124224e90b4565575334b7f305f7d35b3 # :10.45.1i
```

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
