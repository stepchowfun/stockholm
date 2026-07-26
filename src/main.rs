use clap::{ArgAction, Parser};
use ibapi::{
    client::blocking::Client,
    contracts::{Contract, tick_types::TickType},
    market_data::{MarketDataType, realtime::TickTypes},
};
use std::{error::Error, io, time::Duration};

// These constants identify the demonstration instrument and bound the market data request.
const SYMBOL: &str = "AAPL";
const SNAPSHOT_TIMEOUT: Duration = Duration::from_secs(10);

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

// Connect to Interactive Brokers and print the latest available AAPL trade price.
fn main() -> Result<(), Box<dyn Error>> {
    // Parse the command-line arguments.
    let cli = Cli::parse();

    // Request a bounded snapshot so the program exits after receiving the current quote.
    let client = Client::connect(&cli.address, cli.client_id)?;
    let contract = Contract::stock(SYMBOL).build();
    client.switch_market_data_type(MarketDataType::Delayed)?;
    let ticks = client
        .market_data(&contract)
        .snapshot_once(SNAPSHOT_TIMEOUT)?;

    // Prefer a real-time last trade, while accepting IBKR's delayed equivalent when supplied.
    let price = latest_price(&ticks).ok_or_else(|| {
        io::Error::other(format!(
            "no latest trade price for {SYMBOL} arrived within {} seconds",
            SNAPSHOT_TIMEOUT.as_secs(),
        ))
    })?;
    println!("{SYMBOL}: ${price:.2}");

    Ok(())
}

// Extract a real-time or delayed last-trade price from a market data snapshot.
fn latest_price(ticks: &[TickTypes]) -> Option<f64> {
    // Prefer the real-time value even if a delayed tick appeared earlier in the snapshot.
    price_for_tick_type(ticks, &TickType::Last)
        .or_else(|| price_for_tick_type(ticks, &TickType::DelayedLast))
}

// Find the price carried by a particular IBKR tick type.
fn price_for_tick_type(ticks: &[TickTypes], wanted: &TickType) -> Option<f64> {
    ticks.iter().find_map(|tick| match tick {
        TickTypes::Price(price) if &price.tick_type == wanted => Some(price.price),
        TickTypes::PriceSize(price) if &price.price_tick_type == wanted => Some(price.price),
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::{Cli, latest_price};
    use clap::CommandFactory;
    use ibapi::{
        contracts::tick_types::TickType,
        market_data::realtime::{TickPrice, TickTypes},
    };

    #[test]
    fn verify_cli() {
        Cli::command().debug_assert();
    }

    #[test]
    fn extracts_latest_price() {
        // Supply both variants to confirm that the real-time price takes precedence.
        let ticks = [
            TickTypes::Price(TickPrice {
                tick_type: TickType::DelayedLast,
                price: 198.50,
                ..TickPrice::default()
            }),
            TickTypes::Price(TickPrice {
                tick_type: TickType::Last,
                price: 200.25,
                ..TickPrice::default()
            }),
        ];

        // Verify the market-data response is reduced to the price shown to the user.
        assert_eq!(latest_price(&ticks), Some(200.25_f64));
    }
}
