mod backtest;
mod historical;
mod infer;
mod run;
mod state;
mod train;

#[macro_use]
extern crate log;

use chrono::Local;
use clap::{ArgAction, Parser, Subcommand as ClapSubcommand};
use env_logger::{Builder, fmt::style::Effects};
use log::LevelFilter;
use std::{env, error::Error, io::Write, str::FromStr};

// These defaults select the instrument and operational log verbosity.
const DEFAULT_SYMBOL: &str = "SOXL";
const DEFAULT_LOG_LEVEL: LevelFilter = LevelFilter::Debug;

// Set up timestamped, leveled logging for operational output.
fn set_up_logging() {
    Builder::new()
        .filter_module(
            module_path!(),
            LevelFilter::from_str(
                &env::var("LOG_LEVEL").unwrap_or_else(|_| DEFAULT_LOG_LEVEL.to_string()),
            )
            .unwrap_or(DEFAULT_LOG_LEVEL),
        )
        .format(|buf, record| {
            let style = buf
                .default_level_style(record.level())
                .effects(Effects::BOLD);

            writeln!(
                buf,
                "{style}[{} {}]{style:#} {}",
                Local::now().format("%Y-%m-%d %H:%M:%S %:z"),
                record.level(),
                record.args(),
            )
        })
        .init();
}

// This struct represents the command-line arguments.
#[derive(Parser)]
#[command(
    about = concat!(
        env!("CARGO_PKG_DESCRIPTION"),
        "\n\n",
        "More information can be found at: ",
        env!("CARGO_PKG_HOMEPAGE"),
    ),
    version,
    disable_version_flag = true
)]
struct Cli {
    #[arg(short, long, help = "Print version", action = ArgAction::Version)]
    _version: Option<bool>,

    /// Address of the running TWS or IB Gateway API.
    #[arg(long, default_value = "127.0.0.1:4001")]
    address: String,

    /// Client ID to use for the API connection.
    #[arg(long, default_value_t = 100)]
    client_id: i32,

    #[command(subcommand)]
    command: Option<Subcommand>,
}

// These subcommands select the program's operating mode.
#[derive(ClapSubcommand)]
enum Subcommand {
    #[command(about = "Backtest a trading strategy")]
    Backtest(backtest::Args),

    #[command(about = "Run the trading bot (default)")]
    Run(run::Args),

    #[command(about = "Fetch historical market data as CSV")]
    Historical(historical::Args),

    #[command(about = "Run inference with a trained neural network")]
    Infer(infer::Args),

    #[command(about = "Train a neural network on historical stock data")]
    Train(train::Args),
}

// Parse the configuration and run the selected operating mode.
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Set up logging before performing any fallible application work.
    set_up_logging();

    // Parse the command-line arguments.
    let cli = Cli::parse();

    // Decide what to do based on the subcommand.
    match cli.command {
        Some(Subcommand::Backtest(args)) => backtest::run(&args),
        Some(Subcommand::Run(args)) => run::run(&cli.address, cli.client_id, &args).await,
        Some(Subcommand::Historical(args)) => {
            historical::run(&cli.address, cli.client_id, &args).await
        }
        Some(Subcommand::Infer(args)) => infer::run(&args),
        Some(Subcommand::Train(args)) => train::run(&args),
        None => run::run(&cli.address, cli.client_id, &run::Args::default()).await,
    }
}

#[cfg(test)]
mod tests {
    use super::Cli;
    use clap::CommandFactory;

    #[test]
    fn verify_cli() {
        Cli::command().debug_assert();
    }
}
