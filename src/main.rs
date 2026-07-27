use clap::{ArgAction, Parser};
use ibapi::{Client, contracts::Contract, market_data::MarketDataType};
use std::error::Error;
use tokio::time::{self, Duration, MissedTickBehavior};

// These constants identify the demonstration instrument and bound the market data request.
const SYMBOL: &str = "AAPL";
const SNAPSHOT_TIMEOUT: Duration = Duration::from_secs(5);
const SNAPSHOT_INTERVAL: Duration = Duration::from_mins(1);

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

// Connect to Interactive Brokers and print a snapshot of the raw AAPL market data.
#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Parse the command-line arguments.
    let cli = Cli::parse();

    // Connect to the configured TWS or IB Gateway instance.
    let client = Client::connect(&cli.address, cli.client_id).await?;

    // Configure subsequent requests to use subscribed real-time market data.
    client
        .switch_market_data_type(MarketDataType::Realtime)
        .await?;

    // Prepare the demonstration instrument and schedule the first snapshot immediately.
    let contract = Contract::stock(SYMBOL).build();
    let mut interval = time::interval(SNAPSHOT_INTERVAL);
    interval.set_missed_tick_behavior(MissedTickBehavior::Delay);

    // Request and print a bounded snapshot every minute until the process is stopped.
    loop {
        interval.tick().await;

        // Mark the start of each iteration before requesting its snapshot.
        println!("Requesting {SYMBOL} market data snapshot…\n");

        // Collect a complete snapshot, logging failures before the next scheduled attempt.
        let ticks = match client
            .market_data(&contract)
            .snapshot_once(SNAPSHOT_TIMEOUT)
            .await
        {
            Ok(ticks) => ticks,
            Err(error) => {
                eprintln!("Failed to request {SYMBOL} market data snapshot: {error}");
                continue;
            }
        };

        // Print every tick without interpreting or filtering the market data.
        for tick in ticks {
            println!("{tick:?}");
        }
        println!();
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
