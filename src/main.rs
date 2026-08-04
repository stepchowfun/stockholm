mod historical;
mod run;

use clap::{ArgAction, Parser, Subcommand as ClapSubcommand};
use std::error::Error;

// This symbol is used when the user does not select an instrument.
const DEFAULT_SYMBOL: &str = "SOXL";

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
    #[command(about = "Run the trading bot (default)")]
    Run(run::Args),

    #[command(about = "Fetch historical market data as CSV")]
    Historical(historical::Args),
}

// Parse the configuration and run the selected operating mode.
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Parse the command-line arguments.
    let cli = Cli::parse();

    // Decide what to do based on the subcommand.
    match cli.command {
        Some(Subcommand::Run(args)) => run::run(&cli.address, cli.client_id, &args).await,
        Some(Subcommand::Historical(args)) => {
            historical::run(&cli.address, cli.client_id, &args).await
        }
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
