use crate::{DEFAULT_SYMBOL, state};
use clap::Args as ClapArgs;
use ibapi::{
    Client,
    accounts::{AccountSummaryResult, AccountSummaryTags, PositionUpdate, types::AccountGroup},
    contracts::{Contract, tick_types::TickType},
    market_data::{MarketDataType, TradingHours, realtime::TickTypes},
    orders::{Action, OrderData, OrderUpdate, Orders},
    prelude::{StreamExt, Subscription, SubscriptionItemStreamExt},
};
use std::{
    collections::{HashMap, HashSet},
    error::Error,
    io,
    sync::RwLock,
};
use time::OffsetDateTime;
use uuid::Uuid;

// These constants configure the instrument and failure recovery.
const RUN_DELAY: tokio::time::Duration = tokio::time::Duration::from_secs(1);
const RETRY_DELAY: tokio::time::Duration = tokio::time::Duration::from_secs(10);
const ORDER_REF_PREFIX: &str = "stockholm:";
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
    // Details for open orders, keyed by Stockholm's stable order reference.
    open_orders: HashMap<String, VolatileOrder>,

    // The most recently reported funds available for opening new positions.
    available_funds: Option<f64>,

    // The most recently observed bid price, if one is available.
    bid_price: Option<f64>,

    // The most recently observed ask price, if one is available.
    ask_price: Option<f64>,
}

// These connection-local details describe one open Stockholm order.
#[allow(dead_code)]
struct VolatileOrder {
    // The connection-specific order identifier assigned by TWS.
    order_id: i32,

    // The instrument being traded.
    symbol: String,

    // The order's limit price.
    price: f64,

    // Whether the order buys or sells shares.
    side: Side,

    // The number of shares filled so far.
    filled: f64,

    // The number of shares still awaiting execution.
    remaining: f64,
}

// This direction distinguishes buy orders from sell orders.
#[derive(Clone, Copy, Eq, PartialEq)]
enum Side {
    Buy,
    Sell,
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
    // Load persisted state once, falling back to a fresh state when no usable file exists.
    let persistent_state = RwLock::new(state::load().unwrap_or_else(|error| {
        warn!("Unable to load state from disk. Proceeding with initial state. Details: {error}");
        state::initial()
    }));

    // Restart the application after a delay whenever a top-level operation completes.
    loop {
        // Connect to the configured TWS or IB Gateway instance for this attempt.
        match Client::connect(address, client_id).await {
            Ok(client) => {
                if let Err(error) =
                    run_with_connection(&client, &args.symbol, &persistent_state).await
                {
                    error!("{error}");
                }
            }
            Err(error) => error!("Connection to Interactive Brokers Gateway failed: {error}"),
        }

        tokio::time::sleep(RETRY_DELAY).await;
    }
}

// Run every control and streaming task concurrently on one connection.
async fn run_with_connection(
    client: &Client,
    symbol: &str,
    persistent_state: &RwLock<state::State>,
) -> Result<(), Box<dyn Error>> {
    // Start connection-local state without account, quote, or order details.
    let volatile_state = RwLock::new(VolatileState {
        open_orders: HashMap::new(),
        available_funds: None,
        bid_price: None,
        ask_price: None,
    });

    // Configure subsequent requests to use subscribed real-time market data.
    client
        .switch_market_data_type(MarketDataType::Realtime)
        .await?;

    // Subscribe before reconciling the initial snapshot so intervening updates are buffered.
    let order_updates = client.order_update_stream().await?;
    list_orders(client, persistent_state, &volatile_state).await?;

    // Keep every operating loop alive until any one of them requires a reconnect.
    tokio::try_join!(
        control_loop(client, &volatile_state),
        stream_account_summary(client, &volatile_state),
        stream_live_data(client, symbol, &volatile_state),
        stream_order_updates(order_updates, persistent_state, &volatile_state),
        stream_realtime_bars(client, symbol, OVERNIGHT_EXCHANGE),
        stream_realtime_bars(client, symbol, SMART_EXCHANGE),
    )?;

    Ok(())
}

// Repeat order-processing steps until one fails.
async fn control_loop(
    client: &Client,
    volatile_state: &RwLock<VolatileState>,
) -> Result<(), Box<dyn Error>> {
    loop {
        run_step(client, volatile_state).await?;
        tokio::time::sleep(RUN_DELAY).await;
    }
}

// Stream account summary updates across all accessible accounts.
async fn stream_account_summary(
    client: &Client,
    state: &RwLock<VolatileState>,
) -> Result<(), Box<dyn Error>> {
    // Mark the start of the request before collecting its results.
    info!("Requesting account summary…");

    // Request every supported summary field across all accessible accounts.
    let subscription = client
        .account_summary(&AccountGroup("All".to_string()), AccountSummaryTags::ALL)
        .await?;
    let mut summaries = subscription.filter_data();

    // Log the initial summary and subsequent updates for the life of the connection.
    while let Some(update) = summaries.next().await {
        match update? {
            AccountSummaryResult::Summary(summary) => {
                // Retain valid available-funds updates for use by the control loop.
                update_available_funds(state, &summary.tag, &summary.value)?;

                if summary.currency.is_empty() {
                    debug!(
                        "Account summary for {}: {} = {}",
                        summary.account,
                        summary.tag,
                        summary.value,
                    );
                } else {
                    debug!(
                        "Account summary for {}: {} = {} {}",
                        summary.account,
                        summary.tag,
                        summary.value,
                        summary.currency,
                    );
                }
            }
            AccountSummaryResult::End => {
                info!("Finished listing initial account summary.");
            }
        }
    }

    Err(ibapi::Error::UnexpectedEndOfStream.into())
}

// Update available funds from the matching account-summary field.
fn update_available_funds(state: &RwLock<VolatileState>, tag: &str, value: &str) -> io::Result<()> {
    // Ignore unrelated, nonnumeric, and nonfinite account-summary values.
    if tag == AccountSummaryTags::AVAILABLE_FUNDS
        && let Ok(value) = value.parse::<f64>()
        && value.is_finite()
    {
        let mut state = state
            .write()
            .map_err(|_| io::Error::other("Volatile state lock was poisoned."))?;
        state.available_funds = Some(value);
    }

    Ok(())
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
    info!("Streaming {symbol} market data…");

    // Process and log every tick while propagating stream failures to the connection loop.
    while let Some(tick) = ticks.next().await {
        let tick = tick?;

        // Refresh bid and ask availability from either form of price update.
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
        debug!("Market data for {symbol}: {tick:?}");
    }

    Err(ibapi::Error::UnexpectedEndOfStream.into())
}

// Stream order updates for the life of the connection.
async fn stream_order_updates(
    subscription: Subscription<OrderUpdate>,
    persistent_state: &RwLock<state::State>,
    volatile_state: &RwLock<VolatileState>,
) -> Result<(), Box<dyn Error>> {
    // Consume the subscription established before the initial order reconciliation.
    let mut updates = subscription.filter_data();
    info!("Streaming order updates…");

    // Process and log every order-related event while propagating stream failures.
    while let Some(update) = updates.next().await {
        let update: OrderUpdate = update?;

        // Keep persistent identities and volatile details synchronized for managed orders.
        match &update {
            OrderUpdate::OpenOrder(data) if data.order.order_ref.starts_with(ORDER_REF_PREFIX) => {
                update_open_order(persistent_state, volatile_state, data)?;
            }
            OrderUpdate::OrderStatus(status) => {
                update_order_status(
                    persistent_state,
                    volatile_state,
                    status.order_id,
                    status.perm_id,
                    status.filled,
                    status.remaining,
                    status.status.is_terminal(),
                )?;
            }
            OrderUpdate::OpenOrder(_)
            | OrderUpdate::ExecutionData(_)
            | OrderUpdate::CommissionReport(_) => {}
        }

        debug!("Order update: {update:?}");
    }

    Err(ibapi::Error::UnexpectedEndOfStream.into())
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
    info!("Streaming {symbol} five-second bars from {exchange}…");

    // Log every completed bar and propagate stream failures to the connection loop.
    while let Some(bar) = bars.next().await {
        debug!("Five-second bar for {symbol} ({exchange}): {:?}", bar?);
    }

    Err(ibapi::Error::UnexpectedEndOfStream.into())
}

// Run one control-loop step.
async fn run_step(
    client: &Client,
    volatile_state: &RwLock<VolatileState>,
) -> Result<(), Box<dyn Error>> {
    // Fetch the current positions for this step.
    list_positions(client).await?;

    // Log the latest account and quote snapshot after the positions request finishes.
    let state = volatile_state
        .read()
        .map_err(|_| io::Error::other("Volatile state lock was poisoned."))?;
    info!(
        "Available funds: {}; current bid: {}; current ask: {}",
        state
            .available_funds
            .map_or_else(|| "unavailable".to_string(), |funds| funds.to_string()),
        state
            .bid_price
            .map_or_else(|| "unavailable".to_string(), |price| price.to_string()),
        state
            .ask_price
            .map_or_else(|| "unavailable".to_string(), |price| price.to_string()),
    );

    Ok(())
}

// Place a limit order to buy the requested number of shares.
#[allow(dead_code)]
async fn place_limit_buy(
    client: &Client,
    symbol: &str,
    shares: i32,
    limit: f64,
    persistent_state: &RwLock<state::State>,
    volatile_state: &RwLock<VolatileState>,
) -> Result<(), Box<dyn Error>> {
    // Build the order with a stable Stockholm-generated correlation reference.
    let contract = Contract::stock(symbol).build();
    let order_ref = format!("{ORDER_REF_PREFIX}{}", Uuid::new_v4().simple());
    let mut order = client
        .order(&contract)
        .buy(shares)
        .limit(limit)
        .outside_rth()
        .build()?;
    order.order_ref.clone_from(&order_ref);
    order.include_overnight = true;

    // Persist the pending order before submitting it to Interactive Brokers.
    let order_id = client.next_order_id();
    {
        let mut state = persistent_state
            .write()
            .map_err(|_| io::Error::other("The persistent state lock is poisoned."))?;
        state.open_orders.push(state::OpenOrder {
            order_ref: order_ref.clone(),
            perm_id: None,
            created_at: OffsetDateTime::now_utc(),
        });
        state::save(&state)?;
    }

    // Retain the order details for this connection before submitting the order.
    volatile_state
        .write()
        .map_err(|_| io::Error::other("Volatile state lock was poisoned."))?
        .open_orders
        .insert(
            order_ref.clone(),
            VolatileOrder {
                order_id,
                symbol: symbol.to_string(),
                price: limit,
                side: Side::Buy,
                filled: 0.0_f64,
                remaining: f64::from(shares),
            },
        );

    // Submit the order only after its state has been safely persisted.
    client.submit_order(order_id, &contract, &order).await?;
    info!("Submitted limit buy {order_id} ({order_ref}): {shares} {symbol} @ ${limit:.2}");

    Ok(())
}

// Place a limit order to sell the requested number of shares.
#[allow(dead_code)]
async fn place_limit_sell(
    client: &Client,
    symbol: &str,
    shares: i32,
    limit: f64,
    persistent_state: &RwLock<state::State>,
    volatile_state: &RwLock<VolatileState>,
) -> Result<(), Box<dyn Error>> {
    // Build the order with a stable Stockholm-generated correlation reference.
    let contract = Contract::stock(symbol).build();
    let order_ref = format!("{ORDER_REF_PREFIX}{}", Uuid::new_v4().simple());
    let mut order = client
        .order(&contract)
        .sell(shares)
        .limit(limit)
        .outside_rth()
        .build()?;
    order.order_ref.clone_from(&order_ref);
    order.include_overnight = true;

    // Persist the pending order before submitting it to Interactive Brokers.
    let order_id = client.next_order_id();
    {
        let mut state = persistent_state
            .write()
            .map_err(|_| io::Error::other("The persistent state lock is poisoned."))?;
        state.open_orders.push(state::OpenOrder {
            order_ref: order_ref.clone(),
            perm_id: None,
            created_at: OffsetDateTime::now_utc(),
        });
        state::save(&state)?;
    }

    // Retain the order details for this connection before submitting the order.
    volatile_state
        .write()
        .map_err(|_| io::Error::other("Volatile state lock was poisoned."))?
        .open_orders
        .insert(
            order_ref.clone(),
            VolatileOrder {
                order_id,
                symbol: symbol.to_string(),
                price: limit,
                side: Side::Sell,
                filled: 0.0_f64,
                remaining: f64::from(shares),
            },
        );

    // Submit the order only after its state has been safely persisted.
    client.submit_order(order_id, &contract, &order).await?;
    info!("Submitted limit sell {order_id} ({order_ref}): {shares} {symbol} @ ${limit:.2}");

    Ok(())
}

// Record the latest details for an open Stockholm order.
fn update_open_order(
    persistent_state: &RwLock<state::State>,
    volatile_state: &RwLock<VolatileState>,
    data: &OrderData,
) -> io::Result<()> {
    // Extract the connection-local fields from Interactive Brokers' open-order details.
    let price = data
        .order
        .limit_price
        .ok_or_else(|| io::Error::other("A Stockholm order is missing its limit price."))?;
    let side = match data.order.action {
        Action::Buy => Side::Buy,
        Action::Sell | Action::SellShort | Action::SellLong => Side::Sell,
    };
    let perm_id = (data.order.perm_id != 0).then_some(data.order.perm_id);
    let symbol = data.contract.symbol.to_string();

    // Update persistent identifiers by stable reference before retaining volatile details.
    {
        let mut state = persistent_state
            .write()
            .map_err(|_| io::Error::other("The persistent state lock is poisoned."))?;
        let changed = if let Some(order) = state
            .open_orders
            .iter_mut()
            .find(|order| order.order_ref == data.order.order_ref)
        {
            let changed = order.perm_id != perm_id;
            order.perm_id = perm_id;
            changed
        } else {
            state.open_orders.push(state::OpenOrder {
                order_ref: data.order.order_ref.clone(),
                perm_id,
                created_at: OffsetDateTime::now_utc(),
            });
            true
        };
        if changed {
            state::save(&state)?;
        }
    }

    // Refresh volatile details while retaining the latest filled quantity.
    let mut state = volatile_state
        .write()
        .map_err(|_| io::Error::other("Volatile state lock was poisoned."))?;
    let filled = state
        .open_orders
        .get(&data.order.order_ref)
        .map_or(0.0_f64, |order| order.filled);
    state.open_orders.insert(
        data.order.order_ref.clone(),
        VolatileOrder {
            order_id: data.order_id,
            symbol,
            price,
            side,
            filled,
            remaining: data.order.total_quantity - filled,
        },
    );

    Ok(())
}

// Apply an order status when it identifies an order already managed by Stockholm.
fn update_order_status(
    persistent_state: &RwLock<state::State>,
    volatile_state: &RwLock<VolatileState>,
    order_id: i32,
    perm_id: i64,
    filled: f64,
    remaining: f64,
    is_terminal: bool,
) -> io::Result<()> {
    // Match only the stable permanent ID and persist terminal removal.
    let order_ref = {
        let mut state = persistent_state
            .write()
            .map_err(|_| io::Error::other("The persistent state lock is poisoned."))?;
        let Some(index) = state
            .open_orders
            .iter()
            .position(|order| order.perm_id == Some(perm_id))
        else {
            return Ok(());
        };
        let order_ref = state.open_orders[index].order_ref.clone();
        if is_terminal {
            state.open_orders.remove(index);
            state::save(&state)?;
        }
        order_ref
    };

    // Remove terminal orders or refresh execution quantities in volatile state.
    let mut state = volatile_state
        .write()
        .map_err(|_| io::Error::other("Volatile state lock was poisoned."))?;
    if is_terminal {
        state.open_orders.remove(&order_ref);
    } else if let Some(order) = state.open_orders.get_mut(&order_ref) {
        order.order_id = order_id;
        order.filled = filled;
        order.remaining = remaining;
    }

    Ok(())
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

// Update one side of the market, clearing it when the latest price is unusable.
fn update_price(state: &mut VolatileState, tick_type: &TickType, price: f64) {
    // Represent sentinel and otherwise invalid prices as unavailable.
    let price = (price > 0.0_f64).then_some(price);
    match tick_type {
        TickType::Bid => state.bid_price = price,
        TickType::Ask => state.ask_price = price,
        _ => {}
    }
}

// Reconcile current open orders placed by Stockholm.
async fn list_orders(
    client: &Client,
    persistent_state: &RwLock<state::State>,
    volatile_state: &RwLock<VolatileState>,
) -> Result<(), Box<dyn Error>> {
    // Mark the start of the request before collecting its results.
    info!("Requesting Stockholm open orders…");

    // Request every current open order across associated accounts and API clients.
    let subscription = client.all_open_orders().await?;
    let mut orders = subscription.filter_data();
    let mut order_count: usize = 0;
    let mut open_order_refs = HashSet::new();

    // Process and log only orders carrying Stockholm's reference prefix.
    while let Some(order) = orders.next().await {
        match order? {
            Orders::OrderData(data) if data.order.order_ref.starts_with(ORDER_REF_PREFIX) => {
                order_count += 1;
                open_order_refs.insert(data.order.order_ref.clone());
                update_open_order(persistent_state, volatile_state, &data)?;
                debug!("Stockholm open order: {data:?}");
            }
            Orders::OrderStatus(status) => {
                update_order_status(
                    persistent_state,
                    volatile_state,
                    status.order_id,
                    status.perm_id,
                    status.filled,
                    status.remaining,
                    status.status.is_terminal(),
                )?;
            }
            Orders::OrderData(_) => {}
        }
    }

    // Remove every local record absent from IB's complete open-order snapshot.
    {
        let mut state = persistent_state
            .write()
            .map_err(|_| io::Error::other("The persistent state lock is poisoned."))?;
        let previous_len = state.open_orders.len();
        state
            .open_orders
            .retain(|order| open_order_refs.contains(&order.order_ref));
        if state.open_orders.len() != previous_len {
            state::save(&state)?;
        }
    }

    // Keep volatile details aligned with the same complete snapshot.
    volatile_state
        .write()
        .map_err(|_| io::Error::other("Volatile state lock was poisoned."))?
        .open_orders
        .retain(|order_ref, _| open_order_refs.contains(order_ref));

    // Confirm that the complete response arrived even when it contained no orders.
    if order_count == 0 {
        info!("No Stockholm open orders found.");
    } else if order_count == 1 {
        info!("Finished listing 1 Stockholm open order.");
    } else {
        info!("Finished listing {order_count} Stockholm open orders.");
    }

    Ok(())
}

// List all current positions.
async fn list_positions(client: &Client) -> Result<(), Box<dyn Error>> {
    // Mark the start of the request before collecting its results.
    info!("Requesting all positions…");

    // Request every current position across accessible accounts.
    let subscription = client.positions().await?;
    let mut positions = subscription.filter_data();
    let mut position_count: usize = 0;

    // Log position details until the complete initial snapshot arrives.
    while let Some(update) = positions.next().await {
        match update? {
            PositionUpdate::Position(position) => {
                position_count += 1;
                debug!("Position: {position:?}");
            }
            PositionUpdate::PositionEnd => break,
        }
    }

    // Confirm that the complete response arrived even when it contained no positions.
    if position_count == 0 {
        info!("No positions found.");
    } else if position_count == 1 {
        info!("Finished listing 1 position.");
    } else {
        info!("Finished listing {position_count} positions.");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Args, VolatileState, update_available_funds, update_price};
    use clap::Parser;
    use ibapi::accounts::AccountSummaryTags;
    use ibapi::contracts::tick_types::TickType;
    use std::{collections::HashMap, sync::RwLock};

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
    fn clear_nonpositive_bid_and_ask_prices() {
        // Confirm unusable quote updates clear previously valid prices on their side.
        let mut state = VolatileState {
            open_orders: HashMap::new(),
            available_funds: None,
            bid_price: None,
            ask_price: None,
        };
        update_price(&mut state, &TickType::Bid, 100.0);
        update_price(&mut state, &TickType::Ask, 101.0);
        update_price(&mut state, &TickType::Bid, 0.0);
        update_price(&mut state, &TickType::Ask, f64::NAN);

        assert_eq!(state.bid_price, None);
        assert_eq!(state.ask_price, None);
    }

    #[test]
    fn retain_only_valid_available_funds() {
        // Confirm only finite numeric available-funds summaries update volatile state.
        let state = RwLock::new(VolatileState {
            open_orders: HashMap::new(),
            available_funds: None,
            bid_price: None,
            ask_price: None,
        });
        update_available_funds(&state, AccountSummaryTags::AVAILABLE_FUNDS, "1234.5").unwrap();
        update_available_funds(&state, AccountSummaryTags::AVAILABLE_FUNDS, "NaN").unwrap();
        update_available_funds(&state, AccountSummaryTags::NET_LIQUIDATION, "9999").unwrap();

        assert_eq!(state.read().unwrap().available_funds, Some(1234.5_f64));
    }
}
