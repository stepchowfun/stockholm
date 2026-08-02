mod historical;
mod run;

use clap::{ArgAction, Parser, Subcommand as ClapSubcommand};
use ibapi::Client;
use std::error::Error;

// This delay controls recovery from connection and top-level runtime failures.
const RETRY_DELAY: tokio::time::Duration = tokio::time::Duration::from_secs(10);

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
    Run,

    #[command(about = "Fetch historical market data as CSV")]
    Historical(historical::Args),
}

// Parse the configuration and run the selected operating mode.
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Parse the command-line arguments.
    let cli = Cli::parse();

    // Decide what to do based on the subcommand.
    match cli.command.unwrap_or(Subcommand::Run) {
        Subcommand::Run => run(&cli.address, cli.client_id).await,
        Subcommand::Historical(args) => {
            let client = Client::connect(&cli.address, cli.client_id).await?;
            historical::run(&client, &args).await
        }
    }
}

// Reconnect and restart the trading bot after top-level failures.
async fn run(address: &str, client_id: i32) -> Result<(), Box<dyn Error>> {
    // Restart the application after a delay whenever a top-level operation completes.
    loop {
        // Connect to the configured TWS or IB Gateway instance for this attempt.
        match Client::connect(address, client_id).await {
            Ok(client) => {
                if let Err(error) = run::run(&client).await {
                    eprintln!("Error: {error}");
                }
            }
            Err(error) => eprintln!("Connection to Interactive Brokers Gateway failed: {error}"),
        }

        tokio::time::sleep(RETRY_DELAY).await;
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
