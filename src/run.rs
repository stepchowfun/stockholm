use crate::DEFAULT_SYMBOL;
use ibapi::{
    Client,
    accounts::PositionUpdate,
    contracts::Contract,
    market_data::{MarketDataType, TradingHours},
    orders::Orders,
    prelude::{StreamExt, SubscriptionItemStreamExt},
};
use std::error::Error;

// These constants configure the instrument and failure recovery.
const RUN_DELAY: tokio::time::Duration = tokio::time::Duration::from_secs(1);
const RETRY_DELAY: tokio::time::Duration = tokio::time::Duration::from_secs(10);
const SMART_EXCHANGE: &str = "SMART";
const OVERNIGHT_EXCHANGE: &str = "OVERNIGHT";

// Run the main trading loop.
pub async fn run(address: &str, client_id: i32) -> Result<(), Box<dyn Error>> {
    // Restart the application after a delay whenever a top-level operation completes.
    loop {
        // Connect to the configured TWS or IB Gateway instance for this attempt.
        match Client::connect(address, client_id).await {
            Ok(client) => {
                if let Err(error) = run_with_connection(&client).await {
                    eprintln!("Error: {error}");
                }
            }
            Err(error) => eprintln!("Connection to Interactive Brokers Gateway failed: {error}"),
        }

        tokio::time::sleep(RETRY_DELAY).await;
    }
}

// Run the order loop and both market data streams concurrently on one connection.
async fn run_with_connection(client: &Client) -> Result<(), Box<dyn Error>> {
    // Configure subsequent requests to use subscribed real-time market data.
    client
        .switch_market_data_type(MarketDataType::Realtime)
        .await?;

    // Keep every operating loop alive until any one of them requires a reconnect.
    tokio::try_join!(
        run_steps(client),
        stream_live_data(client),
        stream_realtime_bars(client, SMART_EXCHANGE),
        stream_realtime_bars(client, OVERNIGHT_EXCHANGE),
    )?;

    Ok(())
}

// Repeat order-processing steps until one fails.
async fn run_steps(client: &Client) -> Result<(), Box<dyn Error>> {
    loop {
        run_step(client).await?;
        tokio::time::sleep(RUN_DELAY).await;
    }
}

// Stream live market data for the configured symbol.
async fn stream_live_data(client: &Client) -> Result<(), Box<dyn Error>> {
    // Subscribe to the default SMART-routed contract for consolidated data.
    let contract = Contract::stock(DEFAULT_SYMBOL).build();
    let mut subscription = client
        .market_data(&contract)
        .streaming()
        .subscribe()
        .await?;
    println!("[market data] Streaming {DEFAULT_SYMBOL} market data…");

    // Print every tick and propagate stream failures to the connection loop.
    while let Some(tick) = subscription.next().await {
        println!("[market data] {DEFAULT_SYMBOL}: {:?}", tick?);
    }

    Err(ibapi::Error::UnexpectedEndOfStream.into())
}

// Stream real-time five-second bars for the configured symbol and exchange.
async fn stream_realtime_bars(client: &Client, exchange: &str) -> Result<(), Box<dyn Error>> {
    // Subscribe to trade bars for the requested routing venue across all sessions.
    let contract = Contract::stock(DEFAULT_SYMBOL)
        .on_exchange(exchange)
        .build();
    let subscription = client
        .realtime_bars(&contract)
        .trading_hours(TradingHours::Extended)
        .subscribe()
        .await?;
    let mut bars = subscription.filter_data();
    println!("[bars] Streaming {DEFAULT_SYMBOL} five-second bars from {exchange}…");

    // Print every completed bar and propagate stream failures to the connection loop.
    while let Some(bar) = bars.next().await {
        println!("[bars] {DEFAULT_SYMBOL} ({exchange}): {:?}", bar?);
    }

    Err(ibapi::Error::UnexpectedEndOfStream.into())
}

// Print all current open orders.
async fn run_step(client: &Client) -> Result<(), Box<dyn Error>> {
    // List the current orders and positions for this step.
    list_orders(client).await?;
    list_positions(client).await?;

    Ok(())
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
