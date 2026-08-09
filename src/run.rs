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
const DEFAULT_BUYING_POWER_BUFFER: f64 = 90.0_f64;
const DEFAULT_INITIAL_MARGIN_REQUIREMENT: f64 = 75.0_f64;
const ORDER_REF_PREFIX: &str = "stockholm:";
const SMART_EXCHANGE: &str = "SMART";
const OVERNIGHT_EXCHANGE: &str = "OVERNIGHT";

// These arguments configure the trading bot.
#[derive(ClapArgs)]
pub struct Args {
    /// Symbol whose live market data should be streamed.
    #[arg(long, default_value = DEFAULT_SYMBOL)]
    symbol: String,

    /// Percentage of equity withheld when calculating buying power.
    #[arg(
        long,
        default_value_t = DEFAULT_BUYING_POWER_BUFFER,
        value_parser = parse_percent
    )]
    buying_power_buffer: f64,

    /// Initial margin requirement as a percentage in the range (0, 100].
    #[arg(
        long,
        default_value_t = DEFAULT_INITIAL_MARGIN_REQUIREMENT,
        value_parser = parse_positive_percent
    )]
    initial_margin_requirement: f64,
}

// This connection-local state tracks values that do not need to survive a restart.
struct VolatileState {
    // Details for open orders, keyed by their connection-specific order identifier.
    open_orders: HashMap<i32, VolatileOrder>,

    // The most recently reported account equity including loan value.
    equity_with_loan_value: Option<f64>,

    // The most recently reported initial margin requirement.
    init_margin_req: Option<f64>,

    // The most recently observed bid price, if one is available.
    bid_price: Option<f64>,

    // The most recently observed ask price, if one is available.
    ask_price: Option<f64>,
}

// These connection-local details describe one open Stockholm order.
#[allow(dead_code)]
struct VolatileOrder {
    // The stable Stockholm-generated reference attached to the order.
    order_ref: String,

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
            buying_power_buffer: DEFAULT_BUYING_POWER_BUFFER,
            initial_margin_requirement: DEFAULT_INITIAL_MARGIN_REQUIREMENT,
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
                if let Err(error) = run_with_connection(&client, args, &persistent_state).await {
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
    args: &Args,
    persistent_state: &RwLock<state::State>,
) -> Result<(), Box<dyn Error>> {
    // Start connection-local state without account, quote, or order details.
    let volatile_state = RwLock::new(VolatileState {
        open_orders: HashMap::new(),
        equity_with_loan_value: None,
        init_margin_req: None,
        bid_price: None,
        ask_price: None,
    });

    // Configure subsequent requests to use subscribed real-time market data.
    client
        .switch_market_data_type(MarketDataType::Realtime)
        .await?;

    // Subscribe before reconciling initial snapshots so intervening order updates are buffered.
    let order_updates = client.order_update_stream().await?;
    list_orders(client, persistent_state, &volatile_state).await?;
    list_positions(client).await?;

    // Keep every operating loop alive until any one of them requires a reconnect.
    tokio::try_join!(
        control_loop(
            &volatile_state,
            args.buying_power_buffer,
            args.initial_margin_requirement,
        ),
        stream_account_summary(client, &volatile_state),
        stream_live_data(client, &args.symbol, &volatile_state),
        stream_order_updates(order_updates, persistent_state, &volatile_state),
        stream_realtime_bars(client, &args.symbol, OVERNIGHT_EXCHANGE),
        stream_realtime_bars(client, &args.symbol, SMART_EXCHANGE),
    )?;

    Ok(())
}

// Repeat control steps until one fails.
async fn control_loop(
    volatile_state: &RwLock<VolatileState>,
    buying_power_buffer: f64,
    initial_margin_requirement: f64,
) -> Result<(), Box<dyn Error>> {
    loop {
        run_step(
            volatile_state,
            buying_power_buffer,
            initial_margin_requirement,
        )?;
        tokio::time::sleep(RUN_DELAY).await;
    }
}

// Stream account summary updates across all accessible accounts.
async fn stream_account_summary(
    client: &Client,
    state: &RwLock<VolatileState>,
) -> Result<(), Box<dyn Error>> {
    // Mark the start of the request before collecting its results.
    debug!("Requesting account summary…");

    // Request every supported summary field across all accessible accounts.
    let subscription = client
        .account_summary(&AccountGroup("All".to_string()), AccountSummaryTags::ALL)
        .await?;
    let mut summaries = subscription.filter_data();

    // Log the initial summary and subsequent updates for the life of the connection.
    while let Some(update) = summaries.next().await {
        match update? {
            AccountSummaryResult::Summary(summary) => {
                // Retain valid account metrics for use by the control loop.
                update_account_metric(state, &summary.tag, &summary.value)?;

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
                debug!("Finished listing initial account summary.");
            }
        }
    }

    Err(ibapi::Error::UnexpectedEndOfStream.into())
}

// Update tracked metrics from matching account-summary fields.
fn update_account_metric(state: &RwLock<VolatileState>, tag: &str, value: &str) -> io::Result<()> {
    // Ignore nonnumeric and nonfinite account-summary values.
    let Ok(value) = value.parse::<f64>() else {
        return Ok(());
    };
    if !value.is_finite() {
        return Ok(());
    }

    // Ignore unrelated fields and retain recognized metrics.
    let mut state = state
        .write()
        .map_err(|_| io::Error::other("Volatile state lock was poisoned."))?;
    match tag {
        AccountSummaryTags::EQUITY_WITH_LOAN_VALUE => state.equity_with_loan_value = Some(value),
        AccountSummaryTags::INIT_MARGIN_REQ => state.init_margin_req = Some(value),
        _ => {}
    }

    Ok(())
}

// Parse a finite percentage that is strictly positive and at most one hundred.
fn parse_positive_percent(value: &str) -> Result<f64, String> {
    // Reject invalid percentages before they reach the trading control loop.
    match value.parse::<f64>() {
        Ok(value) if value > 0.0_f64 && value <= 100.0_f64 => Ok(value),
        _ => Err("The percentage must be a finite number in the range (0, 100].".to_string()),
    }
}

// Parse a finite percentage in the inclusive range from zero to one hundred.
fn parse_percent(value: &str) -> Result<f64, String> {
    // Reject invalid percentages before they reach the trading control loop.
    match value.parse::<f64>() {
        Ok(value) if (0.0_f64..=100.0_f64).contains(&value) => Ok(value),
        _ => Err("The percentage must be a finite number in the range [0, 100].".to_string()),
    }
}

// Calculate buffered buying power and round it down to the nearest cent.
fn calculate_buying_power(
    equity_with_loan_value: f64,
    init_margin_req: f64,
    buying_power_buffer: f64,
    initial_margin_requirement: f64,
) -> f64 {
    // Reduce effective equity by the configured buffer before applying the margin ratio.
    let effective_equity = equity_with_loan_value * (1.0_f64 - buying_power_buffer / 100.0_f64);
    let margin_capacity = (effective_equity - init_margin_req).max(0.0_f64);
    (margin_capacity / (initial_margin_requirement / 100.0_f64) * 100.0_f64).floor() / 100.0_f64
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
    debug!("Streaming {symbol} market data…");

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
    debug!("Streaming order updates…");

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
    debug!("Streaming {symbol} five-second bars from {exchange}…");

    // Log every completed bar and propagate stream failures to the connection loop.
    while let Some(bar) = bars.next().await {
        debug!("Five-second bar for {symbol} ({exchange}): {:?}", bar?);
    }

    Err(ibapi::Error::UnexpectedEndOfStream.into())
}

// Run one control-loop step.
fn run_step(
    volatile_state: &RwLock<VolatileState>,
    buying_power_buffer: f64,
    initial_margin_requirement: f64,
) -> io::Result<()> {
    // Calculate buying power and log the latest account and quote snapshot.
    let state = volatile_state
        .read()
        .map_err(|_| io::Error::other("Volatile state lock was poisoned."))?;
    let buying_power =
        state
            .equity_with_loan_value
            .zip(state.init_margin_req)
            .map(|(equity, margin)| {
                calculate_buying_power(
                    equity,
                    margin,
                    buying_power_buffer,
                    initial_margin_requirement,
                )
            });
    info!(
        "Equity with loan value: {}; buying power: {}; current bid: {}; current ask: {}",
        state
            .equity_with_loan_value
            .map_or_else(|| "unavailable".to_string(), |equity| equity.to_string()),
        buying_power.map_or_else(|| "unavailable".to_string(), |power| power.to_string()),
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
            order_id,
            VolatileOrder {
                order_ref: order_ref.clone(),
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
            order_id,
            VolatileOrder {
                order_ref: order_ref.clone(),
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
    // Refresh volatile details under the current connection-specific order ID.
    {
        let mut state = volatile_state
            .write()
            .map_err(|_| io::Error::other("Volatile state lock was poisoned."))?;
        state.open_orders.insert(
            data.order_id,
            VolatileOrder {
                order_ref: data.order.order_ref.clone(),
                symbol: data.contract.symbol.to_string(),
                price: data.order.limit_price.ok_or_else(|| {
                    io::Error::other("A Stockholm order is missing its limit price.")
                })?,
                side: match data.order.action {
                    Action::Buy => Side::Buy,
                    Action::Sell => Side::Sell,
                    Action::SellShort | Action::SellLong => {
                        return Err(io::Error::other(
                            "A Stockholm order has an unsupported institutional side.",
                        ));
                    }
                },
                filled: data.order.filled_quantity,
                remaining: data.order.total_quantity - data.order.filled_quantity,
            },
        );
    }

    // Synchronize persistent identifiers by stable reference and save any changes.
    {
        let perm_id = (data.order.perm_id != 0).then_some(data.order.perm_id);
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

    Ok(())
}

// Synchronize volatile and persistent state for a status on a Stockholm-managed order.
fn update_order_status(
    persistent_state: &RwLock<state::State>,
    volatile_state: &RwLock<VolatileState>,
    order_id: i32,
    filled: f64,
    remaining: f64,
    is_terminal: bool,
) -> io::Result<()> {
    // Refresh an active order's quantities, or remove and return a terminal order.
    let terminal_order = {
        let mut state = volatile_state
            .write()
            .map_err(|_| io::Error::other("Volatile state lock was poisoned."))?;
        if is_terminal {
            state.open_orders.remove(&order_id)
        } else {
            if let Some(order) = state.open_orders.get_mut(&order_id) {
                order.filled = filled;
                order.remaining = remaining;
            }
            None
        }
    };

    // Remove the terminal order's stable record and persist only when it was present.
    if let Some(order) = terminal_order {
        let mut state = persistent_state
            .write()
            .map_err(|_| io::Error::other("The persistent state lock is poisoned."))?;
        let previous_len = state.open_orders.len();
        state
            .open_orders
            .retain(|persistent_order| persistent_order.order_ref != order.order_ref);
        if state.open_orders.len() != previous_len {
            state::save(&state)?;
        }
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
    debug!("Requesting Stockholm open orders…");

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
        .retain(|_, order| open_order_refs.contains(&order.order_ref));

    // Confirm that the complete response arrived even when it contained no orders.
    if order_count == 0 {
        debug!("No Stockholm open orders found.");
    } else if order_count == 1 {
        debug!("Finished listing 1 Stockholm open order.");
    } else {
        debug!("Finished listing {order_count} Stockholm open orders.");
    }

    Ok(())
}

// List all current positions.
async fn list_positions(client: &Client) -> Result<(), Box<dyn Error>> {
    // Mark the start of the request before collecting its results.
    debug!("Requesting all positions…");

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
        debug!("No positions found.");
    } else if position_count == 1 {
        debug!("Finished listing 1 position.");
    } else {
        debug!("Finished listing {position_count} positions.");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Args, VolatileState, calculate_buying_power, update_account_metric, update_price};
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
        assert!((cli.args.buying_power_buffer - 90.0_f64).abs() < f64::EPSILON);
        assert!((cli.args.initial_margin_requirement - 75.0_f64).abs() < f64::EPSILON);
    }

    #[test]
    fn explicit_symbol() {
        // Confirm the run command accepts an explicit symbol.
        let cli = TestCli::try_parse_from(["run", "--symbol", "AAPL"]).unwrap();

        assert_eq!(cli.args.symbol, "AAPL");
    }

    #[test]
    fn validate_initial_margin_requirement() {
        // Accept both interior and upper-bound percentages while rejecting invalid values.
        let valid = TestCli::try_parse_from(["run", "--initial-margin-requirement", "100"]);
        let zero = TestCli::try_parse_from(["run", "--initial-margin-requirement", "0"]);
        let excessive = TestCli::try_parse_from(["run", "--initial-margin-requirement", "100.1"]);
        let nonfinite = TestCli::try_parse_from(["run", "--initial-margin-requirement", "NaN"]);

        assert!((valid.unwrap().args.initial_margin_requirement - 100.0_f64).abs() < f64::EPSILON);
        assert!(zero.is_err());
        assert!(excessive.is_err());
        assert!(nonfinite.is_err());
    }

    #[test]
    fn validate_buying_power_buffer() {
        // Accept both buffer boundaries while rejecting out-of-range and nonfinite values.
        let zero = TestCli::try_parse_from(["run", "--buying-power-buffer", "0"]);
        let full = TestCli::try_parse_from(["run", "--buying-power-buffer", "100"]);
        let excessive = TestCli::try_parse_from(["run", "--buying-power-buffer", "100.1"]);
        let negative = TestCli::try_parse_from(["run", "--buying-power-buffer", "-1"]);
        let nonfinite = TestCli::try_parse_from(["run", "--buying-power-buffer", "NaN"]);

        assert!(zero.is_ok());
        assert!(full.is_ok());
        assert!(excessive.is_err());
        assert!(negative.is_err());
        assert!(nonfinite.is_err());
    }

    #[test]
    fn buffer_and_round_down_buying_power() {
        // Apply the equity buffer before the margin formula and floor the result to cents.
        let buying_power = calculate_buying_power(1_000.0, 100.0, 20.0, 75.0);

        assert!((buying_power - 933.33_f64).abs() < f64::EPSILON);
    }

    #[test]
    fn clamp_negative_buying_power() {
        // Report zero once existing margin exceeds the buffered margin budget.
        let buying_power = calculate_buying_power(1_000.0, 900.0, 20.0, 75.0);

        assert!(buying_power.abs() < f64::EPSILON);
    }

    #[test]
    fn clear_nonpositive_bid_and_ask_prices() {
        // Confirm unusable quote updates clear previously valid prices on their side.
        let mut state = VolatileState {
            open_orders: HashMap::new(),
            equity_with_loan_value: None,
            init_margin_req: None,
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
    fn retain_only_valid_account_metrics() {
        // Confirm only finite numeric tracked summaries update volatile state.
        let state = RwLock::new(VolatileState {
            open_orders: HashMap::new(),
            equity_with_loan_value: None,
            init_margin_req: None,
            bid_price: None,
            ask_price: None,
        });
        update_account_metric(&state, AccountSummaryTags::EQUITY_WITH_LOAN_VALUE, "1234.5")
            .unwrap();
        update_account_metric(&state, AccountSummaryTags::INIT_MARGIN_REQ, "234.5").unwrap();
        update_account_metric(&state, AccountSummaryTags::INIT_MARGIN_REQ, "NaN").unwrap();
        update_account_metric(&state, AccountSummaryTags::NET_LIQUIDATION, "9999").unwrap();

        let state = state.read().unwrap();
        assert_eq!(state.equity_with_loan_value, Some(1234.5_f64));
        assert_eq!(state.init_margin_req, Some(234.5_f64));
    }
}
