use clap::{ArgAction, Parser};
use ibapi::{
    Client,
    contracts::Contract,
    market_data::MarketDataType,
    orders::Orders,
    prelude::{StreamExt, SubscriptionItemStreamExt},
};
use std::error::Error;
use tokio::time::{self, Duration};

// These constants configure the instrument and failure recovery.
const SYMBOL: &str = "SOXL";
const RUN_DELAY: Duration = Duration::from_secs(1);
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

    // Restart the application after a delay whenever a top-level operation completes.
    loop {
        // Connect to the configured TWS or IB Gateway instance for this attempt.
        match Client::connect(&cli.address, cli.client_id).await {
            Ok(client) => {
                if let Err(error) = run_connection(&client).await {
                    eprintln!("Error: {error}");
                }
            }
            Err(error) => eprintln!("Connection to Interactive Brokers Gateway failed: {error}"),
        }

        time::sleep(RETRY_DELAY).await;
    }
}

// Run the order loop and market data stream concurrently on one connection.
async fn run_connection(client: &Client) -> Result<(), Box<dyn Error>> {
    tokio::try_join!(run_steps(client), stream_live_data(client))?;

    Ok(())
}

// Repeat order-processing steps until one fails.
async fn run_steps(client: &Client) -> Result<(), Box<dyn Error>> {
    loop {
        run_step(client).await?;
        time::sleep(RUN_DELAY).await;
    }
}

// Stream live market data for the configured symbol.
async fn stream_live_data(client: &Client) -> Result<(), Box<dyn Error>> {
    // Configure subsequent requests to use subscribed real-time market data.
    client
        .switch_market_data_type(MarketDataType::Realtime)
        .await?;

    // Subscribe to the default SMART-routed contract for consolidated data.
    let contract = Contract::stock(SYMBOL).build();
    let mut subscription = client
        .market_data(&contract)
        .streaming()
        .subscribe()
        .await?;
    println!("Streaming {SYMBOL} market data…");

    // Print every tick and propagate stream failures to the connection loop.
    while let Some(tick) = subscription.next().await {
        println!("{SYMBOL} market data: {:?}", tick?);
    }

    Err(ibapi::Error::UnexpectedEndOfStream.into())
}

// Print all current open orders.
async fn run_step(client: &Client) -> Result<(), Box<dyn Error>> {
    // List the current orders for this step.
    list_orders(client).await?;

    Ok(())
}

// Print all current open orders.
async fn list_orders(client: &Client) -> Result<(), Box<dyn Error>> {
    // Mark the start of the request before collecting its results.
    println!("Requesting all open orders…");

    // Request every current open order across associated accounts and API clients.
    let subscription = client.all_open_orders().await?;
    let mut orders = subscription.filter_data();
    let mut order_count: usize = 0;

    // Print order details and statuses while propagating request failures.
    while let Some(order) = orders.next().await {
        match order? {
            Orders::OrderData(data) => {
                order_count += 1;
                println!("- {data:?}");
            }
            Orders::OrderStatus(status) => println!("{status:?}"),
        }
    }

    // Confirm that the complete response arrived even when it contained no orders.
    if order_count == 0 {
        println!("No open orders found.");
    } else if order_count == 1 {
        println!("Finished listing 1 open order.");
    } else {
        println!("Finished listing {order_count} open orders.");
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
