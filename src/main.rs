use clap::{ArgAction, Parser};
use ibapi::{Client, contracts::Contract, market_data::MarketDataType, prelude::StreamExt};
use std::error::Error;
use tokio::time::{self, Duration};

// These constants identify the demonstration instrument and configure failure recovery.
const SYMBOL: &str = "SOXL";
const RETRY_DELAY: Duration = Duration::from_secs(10);

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
}

// Parse the configuration and retry the application after top-level failures.
#[tokio::main]
async fn main() {
    // Parse the command-line arguments.
    let cli = Cli::parse();

    // Restart the application after a delay whenever a top-level operation fails.
    loop {
        if let Err(error) = run(&cli).await {
            eprintln!("Application failed: {error}");
        }

        time::sleep(RETRY_DELAY).await;
    }
}

// Connect to Interactive Brokers and stream the raw SOXL market data.
async fn run(cli: &Cli) -> Result<(), Box<dyn Error>> {
    // Connect to the configured TWS or IB Gateway instance.
    let client = Client::connect(&cli.address, cli.client_id).await?;

    // Configure subsequent requests to use subscribed real-time market data.
    client
        .switch_market_data_type(MarketDataType::Realtime)
        .await?;

    // Prepare the demonstration instrument.
    let contract = Contract::stock(SYMBOL).build();

    // Mark the start of the request before streaming its ticks.
    println!("Streaming {SYMBOL} market data…\n");

    // Subscribe to a continuous stream and propagate setup failures to the retry loop.
    let mut subscription = client
        .market_data(&contract)
        .streaming()
        .subscribe()
        .await?;

    // Print every tick and propagate stream failures to the retry loop.
    while let Some(tick) = subscription.next().await {
        println!("{:?}", tick?);
    }

    Ok(())
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
