use crate::{DEFAULT_SYMBOL, state};
use clap::Args as ClapArgs;
use ibapi::{
    Client,
    accounts::{AccountSummaryResult, AccountSummaryTags, PositionUpdate, types::AccountGroup},
    contracts::{Contract, tick_types::TickType},
    market_data::{MarketDataType, TradingHours, realtime::TickTypes},
    orders::Orders,
    prelude::{StreamExt, SubscriptionItemStreamExt},
};
use std::{error::Error, io, sync::RwLock};

// These constants configure the instrument and failure recovery.
const RUN_DELAY: tokio::time::Duration = tokio::time::Duration::from_secs(1);
const RETRY_DELAY: tokio::time::Duration = tokio::time::Duration::from_secs(10);
const SMART_EXCHANGE: &str = "SMART";
const OVERNIGHT_EXCHANGE: &str = "OVERNIGHT";

// These arguments configure the trading bot.
#[derive(ClapArgs)]
pub struct Args {
    /// Symbol whose live market data should be streamed.
    #[arg(long, default_value = DEFAULT_SYMBOL)]
    symbol: String,
}

// This connection-local state tracks values that do not need to survive a restart.
struct VolatileState {
    // The most recently observed bid price, if one is available.
    bid_price: Option<f64>,

    // The most recently observed ask price, if one is available.
    ask_price: Option<f64>,
}

// Supply the same defaults when the run subcommand is omitted.
impl Default for Args {
    fn default() -> Self {
        Self {
            symbol: DEFAULT_SYMBOL.to_string(),
        }
    }
}

// Run the main trading loop.
pub async fn run(address: &str, client_id: i32, args: &Args) -> Result<(), Box<dyn Error>> {
    // Restart the application after a delay whenever a top-level operation completes.
    loop {
        // Connect to the configured TWS or IB Gateway instance for this attempt.
        match Client::connect(address, client_id).await {
            Ok(client) => {
                if let Err(error) = run_with_connection(&client, &args.symbol).await {
                    eprintln!("Error: {error}");
                }
            }
            Err(error) => eprintln!("Connection to Interactive Brokers Gateway failed: {error}"),
        }

        tokio::time::sleep(RETRY_DELAY).await;
    }
}

// Run the order loop and both market data streams concurrently on one connection.
async fn run_with_connection(client: &Client, symbol: &str) -> Result<(), Box<dyn Error>> {
    // Load persisted state, falling back to a fresh state when no usable file exists.
    let _state = state::load().unwrap_or_else(|error| {
        eprintln!(
            "Unable to load state from disk. Proceeding with initial state. Details: {error}",
        );
        state::initial()
    });

    // Start connection-local market state without any observed prices.
    let volatile_state = RwLock::new(VolatileState {
        bid_price: None,
        ask_price: None,
    });

    // Configure subsequent requests to use subscribed real-time market data.
    client
        .switch_market_data_type(MarketDataType::Realtime)
        .await?;

    // Keep every operating loop alive until any one of them requires a reconnect.
    tokio::try_join!(
        run_steps(client, &volatile_state),
        account_summary(client),
        stream_live_data(client, symbol, &volatile_state),
        stream_realtime_bars(client, symbol, SMART_EXCHANGE),
        stream_realtime_bars(client, symbol, OVERNIGHT_EXCHANGE),
    )?;

    Ok(())
}

// Repeat order-processing steps until one fails.
async fn run_steps(client: &Client, state: &RwLock<VolatileState>) -> Result<(), Box<dyn Error>> {
    loop {
        run_step(client, state).await?;
        tokio::time::sleep(RUN_DELAY).await;
    }
}

// Stream live market data for the configured symbol.
async fn stream_live_data(
    client: &Client,
    symbol: &str,
    state: &RwLock<VolatileState>,
) -> Result<(), Box<dyn Error>> {
    // Subscribe to the default SMART-routed contract for consolidated data.
    let contract = Contract::stock(symbol).build();
    let subscription = client
        .market_data(&contract)
        .streaming()
        .subscribe()
        .await?;
    let mut ticks = subscription.filter_data();
    println!("[market data] Streaming {symbol} market data…");

    // Print every tick and propagate stream failures to the connection loop.
    while let Some(tick) = ticks.next().await {
        let tick = tick?;

        // Retain positive bid and ask prices from either form of price update.
        match &tick {
            TickTypes::Price(tick) => {
                update_locked_price(state, &tick.tick_type, tick.price)?;
            }
            TickTypes::PriceSize(tick) => {
                update_locked_price(state, &tick.price_tick_type, tick.price)?;
            }
            _ => {}
        }

        // Keep logging the complete market-data stream for visibility.
        println!("[market data] {symbol}: {tick:?}");
    }

    Err(ibapi::Error::UnexpectedEndOfStream.into())
}

// Update one quote while holding the connection-local state lock briefly.
fn update_locked_price(
    state: &RwLock<VolatileState>,
    tick_type: &TickType,
    price: f64,
) -> io::Result<()> {
    // Fail the connection attempt if another task poisoned the shared state.
    let mut state = state
        .write()
        .map_err(|_| io::Error::other("Volatile state lock was poisoned."))?;
    update_price(&mut state, tick_type, price);

    Ok(())
}

// Update one side of the market when a usable price arrives.
fn update_price(state: &mut VolatileState, tick_type: &TickType, price: f64) {
    // Ignore sentinel and otherwise invalid prices reported by the data source.
    if price > 0.0_f64 {
        match tick_type {
            TickType::Bid => state.bid_price = Some(price),
            TickType::Ask => state.ask_price = Some(price),
            _ => {}
        }
    }
}

// Stream real-time five-second bars for the configured symbol and exchange.
async fn stream_realtime_bars(
    client: &Client,
    symbol: &str,
    exchange: &str,
) -> Result<(), Box<dyn Error>> {
    // Subscribe to trade bars for the requested routing venue across all sessions.
    let contract = Contract::stock(symbol).on_exchange(exchange).build();
    let subscription = client
        .realtime_bars(&contract)
        .trading_hours(TradingHours::Extended)
        .subscribe()
        .await?;
    let mut bars = subscription.filter_data();
    println!("[bars] Streaming {symbol} five-second bars from {exchange}…");

    // Print every completed bar and propagate stream failures to the connection loop.
    while let Some(bar) = bars.next().await {
        println!("[bars] {symbol} ({exchange}): {:?}", bar?);
    }

    Err(ibapi::Error::UnexpectedEndOfStream.into())
}

// Run one set of independent account checks.
async fn run_step(client: &Client, state: &RwLock<VolatileState>) -> Result<(), Box<dyn Error>> {
    // Fetch the current orders and positions concurrently for this step.
    tokio::try_join!(list_orders(client), list_positions(client))?;

    // Print the latest quote snapshot after the account checks finish.
    let state = state
        .read()
        .map_err(|_| io::Error::other("Volatile state lock was poisoned."))?;
    println!(
        "[market data] Current bid: {}; current ask: {}",
        state
            .bid_price
            .map_or_else(|| "unavailable".to_string(), |price| price.to_string()),
        state
            .ask_price
            .map_or_else(|| "unavailable".to_string(), |price| price.to_string()),
    );

    Ok(())
}

// Stream account summary updates across all accessible accounts.
async fn account_summary(client: &Client) -> Result<(), Box<dyn Error>> {
    // Mark the start of the request before collecting its results.
    println!("[account summary] Requesting account summary…");

    // Request every supported summary field across all accessible accounts.
    let subscription = client
        .account_summary(&AccountGroup("All".to_string()), AccountSummaryTags::ALL)
        .await?;
    let mut summaries = subscription.filter_data();

    // Print the initial summary and subsequent updates for the life of the connection.
    while let Some(update) = summaries.next().await {
        match update? {
            AccountSummaryResult::Summary(summary) => {
                if summary.currency.is_empty() {
                    println!(
                        "[account summary] {}: {} = {}",
                        summary.account, summary.tag, summary.value,
                    );
                } else {
                    println!(
                        "[account summary] {}: {} = {} {}",
                        summary.account, summary.tag, summary.value, summary.currency,
                    );
                }
            }
            AccountSummaryResult::End => {
                println!("[account summary] Finished listing initial account summary.");
            }
        }
    }

    Err(ibapi::Error::UnexpectedEndOfStream.into())
}

// Print all current open orders.
async fn list_orders(client: &Client) -> Result<(), Box<dyn Error>> {
    // Mark the start of the request before collecting its results.
    println!("[orders] Requesting all open orders…");

    // Request every current open order across associated accounts and API clients.
    let subscription = client.all_open_orders().await?;
    let mut orders = subscription.filter_data();
    let mut order_count: usize = 0;

    // Print order details and statuses while propagating request failures.
    while let Some(order) = orders.next().await {
        match order? {
            Orders::OrderData(data) => {
                order_count += 1;
                println!("[orders] - {data:?}");
            }
            Orders::OrderStatus(status) => println!("{status:?}"),
        }
    }

    // Confirm that the complete response arrived even when it contained no orders.
    if order_count == 0 {
        println!("[orders] No open orders found.");
    } else if order_count == 1 {
        println!("[orders] Finished listing 1 open order.");
    } else {
        println!("[orders] Finished listing {order_count} open orders.");
    }

    Ok(())
}

// Print all current positions.
async fn list_positions(client: &Client) -> Result<(), Box<dyn Error>> {
    // Mark the start of the request before collecting its results.
    println!("[positions] Requesting all positions…");

    // Request every current position across accessible accounts.
    let subscription = client.positions().await?;
    let mut positions = subscription.filter_data();
    let mut position_count: usize = 0;

    // Print position details until the complete initial snapshot arrives.
    while let Some(update) = positions.next().await {
        match update? {
            PositionUpdate::Position(position) => {
                position_count += 1;
                println!("[positions] - {position:?}");
            }
            PositionUpdate::PositionEnd => break,
        }
    }

    // Confirm that the complete response arrived even when it contained no positions.
    if position_count == 0 {
        println!("[positions] No positions found.");
    } else if position_count == 1 {
        println!("[positions] Finished listing 1 position.");
    } else {
        println!("[positions] Finished listing {position_count} positions.");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Args, VolatileState, update_price};
    use clap::Parser;
    use ibapi::contracts::tick_types::TickType;

    // This parser exposes the run arguments for focused tests.
    #[derive(Parser)]
    struct TestCli {
        #[command(flatten)]
        args: Args,
    }

    #[test]
    fn default_symbol() {
        // Confirm the run command falls back to the shared default symbol.
        let cli = TestCli::try_parse_from(["run"]).unwrap();

        assert_eq!(cli.args.symbol, "SOXL");
    }

    #[test]
    fn explicit_symbol() {
        // Confirm the run command accepts an explicit symbol.
        let cli = TestCli::try_parse_from(["run", "--symbol", "AAPL"]).unwrap();

        assert_eq!(cli.args.symbol, "AAPL");
    }

    #[test]
    fn retain_only_positive_bid_and_ask_prices() {
        // Confirm valid quote updates replace their side without accepting invalid prices.
        let mut state = VolatileState {
            bid_price: None,
            ask_price: None,
        };
        update_price(&mut state, &TickType::Bid, 100.0);
        update_price(&mut state, &TickType::Ask, 101.0);
        update_price(&mut state, &TickType::Bid, 0.0);
        update_price(&mut state, &TickType::Ask, f64::NAN);

        assert_eq!(state.bid_price, Some(100.0_f64));
        assert_eq!(state.ask_price, Some(101.0_f64));
    }
}
