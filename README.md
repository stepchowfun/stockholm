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

You'll also need to start an Interactive Brokers Gateway that Stockholm can talk to. You can do that via the provided `run-ib-gateway.sh` script:

```sh
TWS_USERID='your_ibkr_username' TWS_PASSWORD='your_ibkr_password' ./run-ib-gateway.sh
```

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

Stockholm depends on IB Gateway, so the repository provides sample service definitions for running both programs as daemons. Adjust the paths, credentials, and other settings in these files as needed before installing them.

#### Creating launchd services on macOS

On macOS, [launchd](https://www.launchd.info/) can be used to run Stockholm and IB Gateway as daemons. Copy [`local.stockholm.plist`](local.stockholm.plist) and [`local.ib-gateway.plist`](local.ib-gateway.plist) to `/Library/LaunchDaemons/`, copy [`run-ib-gateway.sh`](run-ib-gateway.sh) to `/usr/local/bin/`, and make sure all three files are owned by root.

Run the following commands to start the services:

```sh
sudo launchctl load /Library/LaunchDaemons/local.ib-gateway.plist
sudo launchctl load /Library/LaunchDaemons/local.stockholm.plist
```

You can view the logs with `tail -F /var/log/ib-gateway.log /var/log/stockholm.log`.

#### Creating systemd services on Linux

On most Linux distributions, [systemd](https://www.freedesktop.org/wiki/Software/systemd/) can be used to run Stockholm and IB Gateway as daemons. Copy [`stockholm.service`](stockholm.service) and [`ib-gateway.service`](ib-gateway.service) to `/etc/systemd/system/`, copy [`run-ib-gateway.sh`](run-ib-gateway.sh) to `/root/run-ib-gateway.sh`, and make sure all three files are owned by root.

Run `sudo systemctl enable ib-gateway stockholm --now` to enable and start the services. You can view the logs with `sudo journalctl --follow --unit ib-gateway --unit stockholm`.
