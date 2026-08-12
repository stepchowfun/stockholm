use crate::{DEFAULT_SYMBOL, state};
use clap::Args as ClapArgs;
use ibapi::{
    Client,
    accounts::{AccountSummaryResult, AccountSummaryTags, PositionUpdate, types::AccountGroup},
    contracts::Contract,
    market_data::IgnoreSize,
    orders::{Action, OrderData, OrderStatus, OrderUpdate, Orders},
    prelude::{StreamExt, Subscription, SubscriptionItemStreamExt},
};
use std::{
    collections::{HashMap, HashSet},
    error::Error,
    io,
};
use time::{Duration, OffsetDateTime, Time};
use time_tz::{OffsetDateTimeExt, timezones::db::america::NEW_YORK};
use tokio::sync::RwLock;
use uuid::Uuid;

// These constants configure trading defaults, routing venues, and failure recovery.
const RUN_DELAY: tokio::time::Duration = tokio::time::Duration::from_secs(1);
const RETRY_DELAY: tokio::time::Duration = tokio::time::Duration::from_secs(10);
const DEFAULT_BUYING_POWER_BUFFER: f64 = 20.0_f64;
const DEFAULT_INITIAL_MARGIN_REQUIREMENT: f64 = 75.0_f64;
const BUY_DISCOUNT_PERCENT: f64 = 0.7_f64;
const SELL_MARKUP_PERCENT: f64 = 0.9_f64;
const BUY_ORDER_TTL: Duration = Duration::seconds(5);
const SELL_ORDER_TTL: Duration = Duration::seconds(30);
const LIQUIDATION_SELL_ORDER_TTL: Duration = Duration::minutes(1);
const QUOTE_MAX_AGE: Duration = Duration::minutes(5);
const CANCEL_RETRY_DELAY: Duration = Duration::seconds(10);
const ORDER_REF_PREFIX: &str = "stockholm:";
const SMART_EXCHANGE: &str = "SMART";
const OVERNIGHT_EXCHANGE: &str = "OVERNIGHT";

// These Eastern times bound the daily liquidation window.
pub(crate) const LIQUIDATION_START_TIME: Time = match Time::from_hms(15, 45, 0) {
    Ok(time) => time,
    Err(_) => panic!("The liquidation start time must be valid."),
};
pub(crate) const LIQUIDATION_END_TIME: Time = match Time::from_hms(20, 5, 0) {
    Ok(time) => time,
    Err(_) => panic!("The liquidation end time must be valid."),
};

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

    // The current position in the configured symbol after its initial snapshot arrives.
    position_shares: Option<f64>,

    // The most recently reported account equity including loan value.
    equity_with_loan_value: Option<f64>,

    // The most recently reported initial margin requirement.
    init_margin_req: Option<f64>,

    // The most recently observed bid price, if one is available.
    bid_price: Option<f64>,

    // The source timestamp of the most recently observed valid bid price.
    bid_price_timestamp: Option<OffsetDateTime>,

    // The most recently observed ask price, if one is available.
    ask_price: Option<f64>,

    // The source timestamp of the most recently observed valid ask price.
    ask_price_timestamp: Option<OffsetDateTime>,
}

// This runtime state keeps durable and connection-local data behind one lock.
struct RuntimeState {
    // The state serialized to disk and retained across connections.
    persistent: state::State,

    // The state rebuilt from Interactive Brokers for each connection.
    volatile: VolatileState,
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
    filled_shares: f64,

    // The number of shares still awaiting execution.
    remaining_shares: f64,
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

// Construct empty connection-local state before broker snapshots arrive.
fn initial_volatile_state() -> VolatileState {
    VolatileState {
        open_orders: HashMap::new(),
        position_shares: None,
        equity_with_loan_value: None,
        init_margin_req: None,
        bid_price: None,
        bid_price_timestamp: None,
        ask_price: None,
        ask_price_timestamp: None,
    }
}

// Run the main trading loop.
pub async fn run(address: &str, client_id: i32, args: &Args) -> Result<(), Box<dyn Error>> {
    // Load persisted state once and pair it with empty connection-local state.
    let runtime_state = RwLock::new(RuntimeState {
        persistent: state::load().unwrap_or_else(|error| {
            warn!(
                "Unable to load state from disk. Proceeding with initial state. Details: {error}",
            );
            state::initial()
        }),
        volatile: initial_volatile_state(),
    });

    // Restart the application after a delay whenever a top-level operation completes.
    loop {
        // Connect to the configured TWS or IB Gateway instance for this attempt.
        match Client::connect(address, client_id).await {
            Ok(client) => {
                if let Err(error) = run_with_connection(&client, args, &runtime_state).await {
                    error!("{error}");
                }
            }
            Err(error) => error!("Connection to Interactive Brokers Gateway failed: {error}"),
        }

        tokio::time::sleep(RETRY_DELAY).await;
    }
}

// Run the control and streaming futures together on one connection.
async fn run_with_connection(
    client: &Client,
    args: &Args,
    runtime_state: &RwLock<RuntimeState>,
) -> Result<(), Box<dyn Error>> {
    // Discard details from the previous connection before rebuilding broker snapshots.
    runtime_state.write().await.volatile = initial_volatile_state();

    // Subscribe before reconciling the initial snapshot so intervening order updates are buffered.
    let order_updates = client.order_update_stream().await?;
    fetch_orders(client, runtime_state).await?;

    // Keep every operating loop alive until any one of them requires a reconnect.
    tokio::try_join!(
        control_loop(
            client,
            runtime_state,
            &args.symbol,
            args.buying_power_buffer,
            args.initial_margin_requirement,
        ),
        stream_account_summary(client, runtime_state),
        stream_order_updates(order_updates, runtime_state),
        stream_positions(client, &args.symbol, runtime_state),
        stream_tick_by_tick(client, &args.symbol, SMART_EXCHANGE, runtime_state),
        stream_tick_by_tick(client, &args.symbol, OVERNIGHT_EXCHANGE, runtime_state),
        clear_stale_quotes(runtime_state),
    )?;

    Ok(())
}

// Repeat control steps until one fails.
async fn control_loop(
    client: &Client,
    runtime_state: &RwLock<RuntimeState>,
    symbol: &str,
    buying_power_buffer: f64,
    initial_margin_requirement: f64,
) -> Result<(), Box<dyn Error>> {
    loop {
        run_control_step(
            client,
            runtime_state,
            symbol,
            buying_power_buffer,
            initial_margin_requirement,
        )
        .await?;
        tokio::time::sleep(RUN_DELAY).await;
    }
}

// Stream account summary updates across all accessible accounts.
async fn stream_account_summary(
    client: &Client,
    runtime_state: &RwLock<RuntimeState>,
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
                update_account_metric(runtime_state, &summary.tag, &summary.value).await?;

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

// Stream order updates for the life of the connection.
async fn stream_order_updates(
    subscription: Subscription<OrderUpdate>,
    runtime_state: &RwLock<RuntimeState>,
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
                update_open_order(runtime_state, data).await?;
            }
            OrderUpdate::OrderStatus(status) => {
                update_order_status(runtime_state, status).await?;
            }
            OrderUpdate::OpenOrder(_)
            | OrderUpdate::ExecutionData(_)
            | OrderUpdate::CommissionReport(_) => {}
        }

        debug!("Order update: {update:?}");
    }

    Err(ibapi::Error::UnexpectedEndOfStream.into())
}

// Stream positions for the configured symbol for the life of the connection.
async fn stream_positions(
    client: &Client,
    symbol: &str,
    runtime_state: &RwLock<RuntimeState>,
) -> Result<(), Box<dyn Error>> {
    // Subscribe to an initial snapshot followed by incremental position updates.
    let subscription = client.positions().await?;
    let mut updates = subscription.filter_data();
    debug!("Streaming positions for {symbol}…");

    // Retain the matching position and use zero when the initial snapshot omits the symbol.
    while let Some(update) = updates.next().await {
        let mut state = runtime_state.write().await;
        match update? {
            PositionUpdate::Position(position) if position.contract.symbol == symbol => {
                state.volatile.position_shares = Some(position.position);
            }
            PositionUpdate::Position(_) => {}
            PositionUpdate::PositionEnd => {
                state.volatile.position_shares.get_or_insert(0.0_f64);
            }
        }
    }

    Err(ibapi::Error::UnexpectedEndOfStream.into())
}

// Periodically clear bid and ask prices whose source timestamps are too old.
async fn clear_stale_quotes(runtime_state: &RwLock<RuntimeState>) -> Result<(), Box<dyn Error>> {
    loop {
        // Reuse the control cadence so stale prices disappear promptly after crossing the limit.
        tokio::time::sleep(RUN_DELAY).await;
        let mut state = runtime_state.write().await;
        clear_stale_quote_prices(&mut state.volatile, OffsetDateTime::now_utc());
    }
}

// Stream tick-by-tick bid and ask prices for the configured symbol and exchange.
async fn stream_tick_by_tick(
    client: &Client,
    symbol: &str,
    exchange: &str,
    runtime_state: &RwLock<RuntimeState>,
) -> Result<(), Box<dyn Error>> {
    // Subscribe to an unlimited quote stream from the requested routing venue.
    let contract = Contract::stock(symbol).on_exchange(exchange).build();
    let subscription = client
        .tick_by_tick(&contract, 0)
        .bid_ask(IgnoreSize::Yes)
        .await?;
    let mut quotes = subscription.filter_data();
    debug!("Streaming tick-by-tick quotes for {symbol} from {exchange}…");

    // Update both sides atomically and propagate stream failures to the connection loop.
    while let Some(quote) = quotes.next().await {
        let quote = quote?;
        {
            let mut state = runtime_state.write().await;
            state.volatile.bid_price = (quote.bid_price > 0.0_f64).then_some(quote.bid_price);
            state.volatile.bid_price_timestamp = state.volatile.bid_price.map(|_| quote.time);
            state.volatile.ask_price = (quote.ask_price > 0.0_f64).then_some(quote.ask_price);
            state.volatile.ask_price_timestamp = state.volatile.ask_price.map(|_| quote.time);
        }
        debug!("Tick-by-tick quote for {symbol} ({exchange}): {quote:?}");
    }

    Err(ibapi::Error::UnexpectedEndOfStream.into())
}

// Run one control-loop step and submit orders for currently available resources.
async fn run_control_step(
    client: &Client,
    runtime_state: &RwLock<RuntimeState>,
    symbol: &str,
    buying_power_buffer: f64,
    initial_margin_requirement: f64,
) -> Result<(), Box<dyn Error>> {
    // Select order lifetimes and trading behavior from the current Eastern time.
    let now = OffsetDateTime::now_utc();
    let liquidating = is_liquidating(now);
    cancel_expired_orders(client, runtime_state, symbol, liquidating, now).await?;

    // Snapshot resources and quotes without retaining the state lock across submissions.
    let (buying_power, bid_price, ask_price, sellable_shares) = {
        let state = runtime_state.read().await;
        let state = &state.volatile;
        let open_buy_order_value = calculate_open_buy_order_value(&state.open_orders);
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
                        open_buy_order_value,
                    )
                });
        let reserved_sell_shares = calculate_open_sell_shares(&state.open_orders, symbol);
        let sellable_shares = state
            .position_shares
            .map(|shares| (shares - reserved_sell_shares).max(0.0_f64).floor());
        info!(
            concat!(
                "Equity: {}; buying power: {}; open orders: {}; ",
                "bid: {}; ask: {}; liquidating: {}",
            ),
            state
                .equity_with_loan_value
                .map_or_else(|| "unavailable".to_string(), |equity| equity.to_string()),
            buying_power.map_or_else(|| "unavailable".to_string(), |power| power.to_string()),
            state.open_orders.len(),
            state
                .bid_price
                .map_or_else(|| "unavailable".to_string(), |price| price.to_string()),
            state
                .ask_price
                .map_or_else(|| "unavailable".to_string(), |price| price.to_string()),
            liquidating,
        );
        (
            buying_power,
            state.bid_price,
            state.ask_price,
            sellable_shares,
        )
    };

    // Choose ordinary market-making limits or a liquidation limit at the ask.
    let (buy_limit, sell_limit) = calculate_order_limits(bid_price, ask_price, liquidating);

    // Use all available buying power for one whole-share discounted limit order.
    if let (Some(buying_power), Some(limit)) = (buying_power, buy_limit)
        && limit > 0.0_f64
    {
        let shares = (buying_power / limit).floor();
        if shares >= 1.0_f64 {
            place_limit_buy(client, symbol, shares, limit, runtime_state).await?;
        }
    }

    // Offer shares only after the initial position snapshot establishes inventory.
    if let (Some(limit), Some(sellable_shares)) = (sell_limit, sellable_shares)
        && limit > 0.0_f64
        && sellable_shares > 0.0_f64
    {
        place_limit_sell(client, symbol, sellable_shares, limit, runtime_state).await?;
    }

    Ok(())
}

// Cancel expired orders whose most recent cancellation attempt is old enough to retry.
async fn cancel_expired_orders(
    client: &Client,
    runtime_state: &RwLock<RuntimeState>,
    symbol: &str,
    liquidating: bool,
    now: OffsetDateTime,
) -> Result<(), Box<dyn Error>> {
    // Join order representations and persist cancellation attempts under one state guard.
    let cancellation_attempts = {
        let mut state = runtime_state.write().await;
        let volatile_orders = state
            .volatile
            .open_orders
            .iter()
            .map(|(&order_id, order)| {
                (
                    order.order_ref.clone(),
                    (order_id, order.side, order.symbol.clone()),
                )
            })
            .collect::<HashMap<_, _>>();
        let mut attempts = Vec::new();
        for order in &mut state.persistent.open_orders {
            let Some((order_id, side, order_symbol)) = volatile_orders.get(&order.order_ref) else {
                continue;
            };
            let liquidation_order = liquidating && order_symbol == symbol;
            if cancellation_due(order, *side, liquidation_order, now) {
                order.last_cancelled_at = Some(now);
                attempts.push((*order_id, order.order_ref.clone()));
            }
        }
        if !attempts.is_empty() {
            state::save(&state.persistent)?;
        }
        attempts
    };

    // Send every due cancellation and leave resources reserved until terminal updates arrive.
    for (order_id, order_ref) in cancellation_attempts {
        info!("Cancelling expired order {order_id} ({order_ref})…");
        let _subscription = client.cancel_order(order_id, "").await?;
    }

    Ok(())
}

// Determine whether the Eastern clock is inside the daily liquidation window.
fn is_liquidating(now: OffsetDateTime) -> bool {
    let eastern_time = now.to_timezone(NEW_YORK).time();

    eastern_time >= LIQUIDATION_START_TIME && eastern_time < LIQUIDATION_END_TIME
}

// Place a limit order to buy the requested number of shares.
async fn place_limit_buy(
    client: &Client,
    symbol: &str,
    shares: f64,
    limit: f64,
    runtime_state: &RwLock<RuntimeState>,
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

    // Record both representations and persist the pending order before submitting it.
    let order_id = client.next_order_id();
    {
        let mut state = runtime_state.write().await;
        state.persistent.open_orders.push(state::OpenOrder {
            order_ref: order_ref.clone(),
            perm_id: None,
            created_at: OffsetDateTime::now_utc(),
            last_cancelled_at: None,
        });
        state.volatile.open_orders.insert(
            order_id,
            VolatileOrder {
                order_ref: order_ref.clone(),
                symbol: symbol.to_string(),
                price: limit,
                side: Side::Buy,
                filled_shares: 0.0_f64,
                remaining_shares: shares,
            },
        );
        state::save(&state.persistent)?;
    }

    // Submit the order only after its state has been safely persisted.
    client.submit_order(order_id, &contract, &order).await?;
    info!("Submitted limit buy {order_id} ({order_ref}): {shares} {symbol} @ ${limit:.2}");

    Ok(())
}

// Place a limit order to sell the requested number of shares.
async fn place_limit_sell(
    client: &Client,
    symbol: &str,
    shares: f64,
    limit: f64,
    runtime_state: &RwLock<RuntimeState>,
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

    // Record both representations and persist the pending order before submitting it.
    let order_id = client.next_order_id();
    {
        let mut state = runtime_state.write().await;
        state.persistent.open_orders.push(state::OpenOrder {
            order_ref: order_ref.clone(),
            perm_id: None,
            created_at: OffsetDateTime::now_utc(),
            last_cancelled_at: None,
        });
        state.volatile.open_orders.insert(
            order_id,
            VolatileOrder {
                order_ref: order_ref.clone(),
                symbol: symbol.to_string(),
                price: limit,
                side: Side::Sell,
                filled_shares: 0.0_f64,
                remaining_shares: shares,
            },
        );
        state::save(&state.persistent)?;
    }

    // Submit the order only after its state has been safely persisted.
    client.submit_order(order_id, &contract, &order).await?;
    info!("Submitted limit sell {order_id} ({order_ref}): {shares} {symbol} @ ${limit:.2}");

    Ok(())
}

// Reconcile current open orders placed by Stockholm.
async fn fetch_orders(
    client: &Client,
    runtime_state: &RwLock<RuntimeState>,
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
                update_open_order(runtime_state, &data).await?;
                debug!("Stockholm open order: {data:?}");
            }
            Orders::OrderStatus(status) => {
                update_order_status(runtime_state, &status).await?;
            }
            Orders::OrderData(_) => {}
        }
    }

    // Prune both representations together using IB's complete open-order snapshot.
    {
        let mut state = runtime_state.write().await;
        let previous_len = state.persistent.open_orders.len();
        state
            .persistent
            .open_orders
            .retain(|order| open_order_refs.contains(&order.order_ref));
        state
            .volatile
            .open_orders
            .retain(|_, order| open_order_refs.contains(&order.order_ref));
        if state.persistent.open_orders.len() != previous_len {
            state::save(&state.persistent)?;
        }
    }

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

// Update tracked metrics from matching account-summary fields.
async fn update_account_metric(
    runtime_state: &RwLock<RuntimeState>,
    tag: &str,
    value: &str,
) -> io::Result<()> {
    // Ignore nonnumeric and nonfinite account-summary values.
    let Ok(value) = value.parse::<f64>() else {
        return Ok(());
    };
    if !value.is_finite() {
        return Ok(());
    }

    // Ignore unrelated fields and retain recognized metrics.
    let mut state = runtime_state.write().await;
    match tag {
        AccountSummaryTags::EQUITY_WITH_LOAN_VALUE => {
            state.volatile.equity_with_loan_value = Some(value);
        }
        AccountSummaryTags::INIT_MARGIN_REQ => state.volatile.init_margin_req = Some(value),
        _ => {}
    }

    Ok(())
}

// Record the latest details for an open Stockholm order.
async fn update_open_order(
    runtime_state: &RwLock<RuntimeState>,
    data: &OrderData,
) -> io::Result<()> {
    // Validate connection-specific details before changing either representation.
    let volatile_order = VolatileOrder {
        order_ref: data.order.order_ref.clone(),
        symbol: data.contract.symbol.to_string(),
        price: data
            .order
            .limit_price
            .ok_or_else(|| io::Error::other("A Stockholm order is missing its limit price."))?,
        side: match data.order.action {
            Action::Buy => Side::Buy,
            Action::Sell => Side::Sell,
            Action::SellShort | Action::SellLong => {
                return Err(io::Error::other(
                    "A Stockholm order has an unsupported institutional side.",
                ));
            }
        },
        filled_shares: data.order.filled_quantity,
        remaining_shares: data.order.total_quantity - data.order.filled_quantity,
    };

    // Refresh volatile details and persistent identifiers under one state guard.
    let perm_id = (data.order.perm_id != 0).then_some(data.order.perm_id);
    let mut state = runtime_state.write().await;
    state
        .volatile
        .open_orders
        .insert(data.order_id, volatile_order);
    let changed = if let Some(order) = state
        .persistent
        .open_orders
        .iter_mut()
        .find(|order| order.order_ref == data.order.order_ref)
    {
        let changed = order.perm_id != perm_id;
        order.perm_id = perm_id;
        changed
    } else {
        state.persistent.open_orders.push(state::OpenOrder {
            order_ref: data.order.order_ref.clone(),
            perm_id,
            created_at: OffsetDateTime::now_utc(),
            last_cancelled_at: None,
        });
        true
    };
    if changed {
        state::save(&state.persistent)?;
    }

    Ok(())
}

// Apply a status and persist any terminal removal under one state guard.
async fn update_order_status(
    runtime_state: &RwLock<RuntimeState>,
    status: &OrderStatus,
) -> io::Result<()> {
    // Keep the state guard until both representations and durable state are updated.
    let mut state = runtime_state.write().await;
    let persistent_changed = if status.status.is_terminal() {
        let order_ref = state
            .volatile
            .open_orders
            .remove(&status.order_id)
            .map(|order| order.order_ref);
        let previous_len = state.persistent.open_orders.len();
        if let Some(order_ref) = order_ref {
            state
                .persistent
                .open_orders
                .retain(|order| order.order_ref != order_ref);
        } else if status.perm_id != 0 {
            state
                .persistent
                .open_orders
                .retain(|order| order.perm_id != Some(status.perm_id));
        }
        state.persistent.open_orders.len() != previous_len
    } else {
        if let Some(order) = state.volatile.open_orders.get_mut(&status.order_id) {
            order.filled_shares = status.filled;
            order.remaining_shares = status.remaining;
        }
        false
    };
    if persistent_changed {
        state::save(&state.persistent)?;
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

// Total the notional still reserved by active buy orders.
fn calculate_open_buy_order_value(open_orders: &HashMap<i32, VolatileOrder>) -> f64 {
    // Reserve only the unfilled notional of orders that increase the long position.
    open_orders
        .values()
        .filter(|order| order.side == Side::Buy)
        .map(|order| order.price * order.remaining_shares)
        .sum()
}

// Total the shares still reserved by active sell orders.
fn calculate_open_sell_shares(open_orders: &HashMap<i32, VolatileOrder>, symbol: &str) -> f64 {
    // Reserve only matching unfilled sells because positions are tracked per symbol.
    open_orders
        .values()
        .filter(|order| order.symbol == symbol && order.side == Side::Sell)
        .map(|order| order.remaining_shares)
        .sum()
}

// Calculate the limits used by the current normal or liquidation behavior.
fn calculate_order_limits(
    bid_price: Option<f64>,
    ask_price: Option<f64>,
    liquidating: bool,
) -> (Option<f64>, Option<f64>) {
    if liquidating {
        (None, ask_price.map(round_up_to_cent))
    } else {
        (
            bid_price.map(|price| {
                round_down_to_cent(price * (1.0_f64 - BUY_DISCOUNT_PERCENT / 100.0_f64))
            }),
            ask_price
                .map(|price| round_up_to_cent(price * (1.0_f64 + SELL_MARKUP_PERCENT / 100.0_f64))),
        )
    }
}

// Clear each quote side independently once its source timestamp exceeds the age limit.
fn clear_stale_quote_prices(state: &mut VolatileState, now: OffsetDateTime) {
    let cutoff = now - QUOTE_MAX_AGE;

    // Clear a stale bid and its timestamp together.
    if state
        .bid_price_timestamp
        .is_some_and(|timestamp| timestamp < cutoff)
    {
        state.bid_price = None;
        state.bid_price_timestamp = None;
    }

    // Clear a stale ask and its timestamp together.
    if state
        .ask_price_timestamp
        .is_some_and(|timestamp| timestamp < cutoff)
    {
        state.ask_price = None;
        state.ask_price_timestamp = None;
    }
}

// Determine whether an open order has expired and is eligible for another cancellation attempt.
fn cancellation_due(
    order: &state::OpenOrder,
    side: Side,
    liquidating: bool,
    now: OffsetDateTime,
) -> bool {
    // Apply the liquidation lifetime or the normal side-specific lifetime.
    let time_to_live = if liquidating {
        match side {
            Side::Buy => Duration::ZERO,
            Side::Sell => LIQUIDATION_SELL_ORDER_TTL,
        }
    } else {
        match side {
            Side::Buy => BUY_ORDER_TTL,
            Side::Sell => SELL_ORDER_TTL,
        }
    };

    // Space repeated cancellation requests even after the order has expired.
    now >= order.created_at + time_to_live
        && order
            .last_cancelled_at
            .is_none_or(|last_cancelled_at| now >= last_cancelled_at + CANCEL_RETRY_DELAY)
}

// Calculate buffered buying power and round it down to the nearest cent.
fn calculate_buying_power(
    equity_with_loan_value: f64,
    init_margin_req: f64,
    buying_power_buffer: f64,
    initial_margin_requirement: f64,
    open_buy_order_value: f64,
) -> f64 {
    // Reserve buffered equity and the margin needed by unfilled buy orders.
    let margin_ratio = initial_margin_requirement / 100.0_f64;
    let effective_equity = equity_with_loan_value * (1.0_f64 - buying_power_buffer / 100.0_f64);
    let open_buy_order_margin = open_buy_order_value * margin_ratio;
    let margin_capacity = (effective_equity - init_margin_req - open_buy_order_margin).max(0.0_f64);
    round_down_to_cent(margin_capacity / margin_ratio)
}

// Round a positive limit price down to an accepted cent boundary.
fn round_down_to_cent(price: f64) -> f64 {
    (price * 100.0_f64).floor() / 100.0_f64
}

// Round a positive limit price up to an accepted cent boundary.
fn round_up_to_cent(price: f64) -> f64 {
    (price * 100.0_f64).ceil() / 100.0_f64
}

#[cfg(test)]
mod tests {
    use super::{
        Args, RuntimeState, Side, VolatileOrder, calculate_buying_power,
        calculate_open_buy_order_value, calculate_open_sell_shares, calculate_order_limits,
        cancellation_due, clear_stale_quote_prices, initial_volatile_state, is_liquidating,
        round_down_to_cent, round_up_to_cent, update_account_metric,
    };
    use crate::state;
    use clap::Parser;
    use ibapi::accounts::AccountSummaryTags;
    use std::collections::HashMap;
    use time::{Date, Duration, Month, OffsetDateTime};
    use tokio::sync::RwLock;

    // This parser exposes the run arguments for focused tests.
    #[derive(Parser)]
    struct TestCli {
        #[command(flatten)]
        args: Args,
    }

    // Construct UTC test instants without relying on formatter-sensitive macros.
    fn utc_datetime(month: Month, day: u8, hour: u8, minute: u8, second: u8) -> OffsetDateTime {
        Date::from_calendar_date(2026, month, day)
            .unwrap()
            .with_hms(hour, minute, second)
            .unwrap()
            .assume_utc()
    }

    #[test]
    fn default_symbol() {
        // Confirm the run command falls back to the shared default symbol.
        let cli = TestCli::try_parse_from(["run"]).unwrap();

        assert_eq!(cli.args.symbol, "SOXL");
        assert!((cli.args.buying_power_buffer - 20.0_f64).abs() < f64::EPSILON);
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
        let buying_power = calculate_buying_power(1_000.0, 100.0, 20.0, 75.0, 0.0);

        assert!((buying_power - 933.33_f64).abs() < f64::EPSILON);
    }

    #[test]
    fn clamp_negative_buying_power() {
        // Report zero once existing margin exceeds the buffered margin budget.
        let buying_power = calculate_buying_power(1_000.0, 900.0, 20.0, 75.0, 0.0);

        assert!(buying_power.abs() < f64::EPSILON);
    }

    #[test]
    fn reserve_open_buy_order_margin() {
        // Deduct the unfilled buy notional after converting it to required margin.
        let buying_power = calculate_buying_power(1_000.0, 100.0, 20.0, 75.0, 200.0);

        assert!((buying_power - 733.33_f64).abs() < f64::EPSILON);
    }

    #[test]
    fn total_only_remaining_buy_orders() {
        // Ignore filled quantities and sell orders when totaling reserved notional.
        let open_orders = HashMap::from([
            (
                1_i32,
                VolatileOrder {
                    order_ref: "stockholm:buy".to_string(),
                    symbol: "SOXL".to_string(),
                    price: 10.0,
                    side: Side::Buy,
                    filled_shares: 3.0,
                    remaining_shares: 2.0,
                },
            ),
            (
                2_i32,
                VolatileOrder {
                    order_ref: "stockholm:sell".to_string(),
                    symbol: "SOXL".to_string(),
                    price: 20.0,
                    side: Side::Sell,
                    filled_shares: 0.0,
                    remaining_shares: 4.0,
                },
            ),
        ]);

        assert!((calculate_open_buy_order_value(&open_orders) - 20.0_f64).abs() < f64::EPSILON);
        assert!((calculate_open_sell_shares(&open_orders, "SOXL") - 4.0_f64).abs() < f64::EPSILON);
        assert!(calculate_open_sell_shares(&open_orders, "AAPL").abs() < f64::EPSILON);
    }

    #[test]
    fn select_limits_for_current_trading_behavior() {
        // Suppress buys and quote directly at the ask only while liquidating.
        assert_eq!(
            calculate_order_limits(Some(10.129_f64), Some(10.231_f64), false),
            (Some(10.05_f64), Some(10.33_f64)),
        );
        assert_eq!(
            calculate_order_limits(Some(10.129_f64), Some(10.231_f64), true),
            (None, Some(10.24_f64)),
        );
        assert_eq!(
            calculate_order_limits(Some(10.129_f64), None, true),
            (None, None),
        );
    }

    #[test]
    fn clear_only_quotes_older_than_five_minutes() {
        // Expire each side independently while retaining a quote exactly at the age boundary.
        let now = OffsetDateTime::UNIX_EPOCH + Duration::minutes(10);
        let mut state = initial_volatile_state();
        state.bid_price = Some(10.0_f64);
        state.bid_price_timestamp = Some(now - Duration::minutes(5) - Duration::seconds(1));
        state.ask_price = Some(11.0_f64);
        state.ask_price_timestamp = Some(now - Duration::minutes(5));

        clear_stale_quote_prices(&mut state, now);

        assert_eq!(state.bid_price, None);
        assert_eq!(state.bid_price_timestamp, None);
        assert_eq!(state.ask_price, Some(11.0_f64));
        assert_eq!(state.ask_price_timestamp, Some(now - Duration::minutes(5)));
    }

    #[test]
    fn round_limit_prices_conservatively() {
        // Keep buys below and sells above fractional-cent strategy prices.
        assert!((round_down_to_cent(10.129_f64) - 10.12_f64).abs() < f64::EPSILON);
        assert!((round_up_to_cent(10.121_f64) - 10.13_f64).abs() < f64::EPSILON);
    }

    #[test]
    fn retry_expired_order_cancellations() {
        // Cancel after each order lifetime and then at ten-second retry intervals.
        let now = OffsetDateTime::UNIX_EPOCH + Duration::seconds(5);
        let mut order = state::OpenOrder {
            order_ref: "stockholm:test".to_string(),
            perm_id: None,
            created_at: OffsetDateTime::UNIX_EPOCH,
            last_cancelled_at: None,
        };

        assert!(!cancellation_due(
            &order,
            Side::Sell,
            false,
            OffsetDateTime::UNIX_EPOCH + Duration::seconds(29),
        ));
        assert!(cancellation_due(
            &order,
            Side::Sell,
            false,
            OffsetDateTime::UNIX_EPOCH + Duration::seconds(30),
        ));
        assert!(cancellation_due(&order, Side::Buy, false, now));
        order.last_cancelled_at = Some(now - Duration::seconds(9));
        assert!(!cancellation_due(&order, Side::Buy, false, now));
        order.last_cancelled_at = Some(now - Duration::seconds(10));
        assert!(cancellation_due(&order, Side::Buy, false, now));
    }

    #[test]
    fn expire_orders_quickly_during_liquidation() {
        // Expire buys immediately and sells after one minute during liquidation.
        let order = state::OpenOrder {
            order_ref: "stockholm:test".to_string(),
            perm_id: None,
            created_at: OffsetDateTime::UNIX_EPOCH,
            last_cancelled_at: None,
        };

        assert!(cancellation_due(
            &order,
            Side::Buy,
            true,
            OffsetDateTime::UNIX_EPOCH,
        ));
        assert!(!cancellation_due(
            &order,
            Side::Sell,
            true,
            OffsetDateTime::UNIX_EPOCH + Duration::seconds(59),
        ));
        assert!(cancellation_due(
            &order,
            Side::Sell,
            true,
            OffsetDateTime::UNIX_EPOCH + Duration::seconds(60),
        ));
    }

    #[test]
    fn liquidate_between_start_and_end_times() {
        // Compare Eastern wall-clock boundaries in both daylight and standard time.
        assert!(!is_liquidating(utc_datetime(Month::August, 10, 19, 44, 59)));
        assert!(is_liquidating(utc_datetime(Month::August, 10, 19, 45, 0)));
        assert!(is_liquidating(utc_datetime(Month::August, 11, 0, 4, 59)));
        assert!(!is_liquidating(utc_datetime(Month::August, 11, 0, 5, 0)));
        assert!(!is_liquidating(utc_datetime(Month::January, 9, 20, 44, 59)));
        assert!(is_liquidating(utc_datetime(Month::January, 9, 20, 45, 0)));
        assert!(is_liquidating(utc_datetime(Month::January, 10, 1, 4, 59)));
        assert!(!is_liquidating(utc_datetime(Month::January, 10, 1, 5, 0)));
    }

    #[tokio::test]
    async fn retain_only_valid_account_metrics() {
        // Confirm only finite numeric tracked summaries update volatile state.
        let state = RwLock::new(RuntimeState {
            persistent: state::initial(),
            volatile: initial_volatile_state(),
        });
        update_account_metric(&state, AccountSummaryTags::EQUITY_WITH_LOAN_VALUE, "1234.5")
            .await
            .unwrap();
        update_account_metric(&state, AccountSummaryTags::INIT_MARGIN_REQ, "234.5")
            .await
            .unwrap();
        update_account_metric(&state, AccountSummaryTags::INIT_MARGIN_REQ, "NaN")
            .await
            .unwrap();
        update_account_metric(&state, AccountSummaryTags::NET_LIQUIDATION, "9999")
            .await
            .unwrap();

        let state = state.read().await;
        assert_eq!(state.volatile.equity_with_loan_value, Some(1234.5_f64));
        assert_eq!(state.volatile.init_margin_req, Some(234.5_f64));
    }
}
