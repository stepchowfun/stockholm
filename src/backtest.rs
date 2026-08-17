use crate::run::{LIQUIDATION_END_TIME, LIQUIDATION_START_TIME};
use clap::{Args as ClapArgs, ValueEnum};
use rayon::prelude::*;
use std::{error::Error, fs, path::PathBuf, sync::Mutex};
use time::{OffsetDateTime, Time};
use time_tz::{OffsetDateTimeExt, timezones::db::america::NEW_YORK};

// These Eastern times bound the unreliable early-morning trade-reporting window.
pub const UNRELIABLE_DATA_START_TIME: Time = match Time::from_hms(4, 0, 0) {
    Ok(time) => time,
    Err(_) => panic!("The unreliable data start time must be valid."),
};
pub const UNRELIABLE_DATA_END_TIME: Time = match Time::from_hms(4, 15, 0) {
    Ok(time) => time,
    Err(_) => panic!("The unreliable data end time must be valid."),
};

// This convention annualizes statistics calculated from one return per trading day.
const TRADING_DAYS_PER_YEAR: f64 = 252.0_f64;

// These strategies can be evaluated by a backtest.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum Strategy {
    BuyAndHold,
    MarketMaker,
    MarketMakerGrid,
}

// These arguments configure a backtest run.
#[derive(ClapArgs)]
pub struct Args {
    /// Trading strategy to evaluate. Required for every backtest.
    #[arg(long, value_enum)]
    strategy: Strategy,

    /// CSV files containing historical market data. Required by every strategy.
    #[arg(long, required = true, num_args = 1..)]
    data_paths: Vec<PathBuf>,

    /// Starting cash used by market-maker and market-maker-grid.
    #[arg(long, default_value_t = 1_000_000.0, value_parser = parse_positive_f64)]
    initial_cash: f64,

    /// Maximum total shares filled per bar. Used by both market-maker strategies.
    #[arg(long, default_value_t = 1_000.0, value_parser = parse_positive_f64)]
    bar_volume_limit: f64,

    /// Buy-order lifetime in elapsed timestamp seconds. Ignored by other strategies.
    #[arg(long, default_value_t = 3_600)]
    buy_ttl: u64,

    /// Sell-order lifetime in elapsed timestamp seconds. Ignored by other strategies.
    #[arg(long, default_value_t = 14_400)]
    sell_ttl: u64,

    /// Buy-limit discount used by market-maker. Ignored by other strategies.
    #[arg(long, default_value_t = 0.25, value_parser = parse_discount_percent)]
    discount_percent: f64,

    /// Sell-limit markup used by market-maker. Ignored by other strategies.
    #[arg(long, default_value_t = 0.25, value_parser = parse_nonnegative_f64)]
    markup_percent: f64,

    /// Share of liquidation value available for buying. Used by market-maker.
    #[arg(long, default_value_t = 80.0, value_parser = parse_percent)]
    bet_size: f64,

    /// Buy-order lifetimes in elapsed timestamp seconds searched by market-maker-grid.
    #[arg(
        long,
        value_delimiter = ',',
        num_args = 1..,
        default_value = "5,15,30,60,120,300,900,3600,7200,14400,43200,86400"
    )]
    buy_ttls: Vec<u64>,

    /// Sell-order lifetimes in elapsed timestamp seconds searched by market-maker-grid.
    #[arg(
        long,
        value_delimiter = ',',
        num_args = 1..,
        default_value = "5,15,30,60,120,300,900,3600,7200,14400,43200,86400"
    )]
    sell_ttls: Vec<u64>,

    /// Buy-limit discounts searched by market-maker-grid. Ignored by other strategies.
    #[arg(
        long,
        value_delimiter = ',',
        num_args = 1..,
        default_value = "0.01,0.03,0.1,0.3,1,3,10",
        value_parser = parse_discount_percent
    )]
    discount_percentages: Vec<f64>,

    /// Sell-limit markups searched by market-maker-grid. Ignored by other strategies.
    #[arg(
        long,
        value_delimiter = ',',
        num_args = 1..,
        default_value = "0.01,0.03,0.1,0.3,1,3,10",
        value_parser = parse_nonnegative_f64
    )]
    markup_percentages: Vec<f64>,

    /// Bet sizes searched by market-maker-grid. Ignored by other strategies.
    #[arg(
        long,
        value_delimiter = ',',
        num_args = 1..,
        default_value = "80,90,100",
        value_parser = parse_percent
    )]
    bet_sizes: Vec<f64>,
}

// This bar contains the timestamp and prices needed to simulate order timing and fills.
struct Bar {
    timestamp: i64,
    low: f64,
    high: f64,
    close: f64,
    liquidate: bool,
}

// This trading day retains its source filename for market-maker event logs.
struct Day {
    filename: String,
    bars: Vec<Bar>,
}

// This order reserves either cash or shares until it fills or expires.
struct LimitOrder {
    placed_timestamp: i64,
    price: f64,
    remaining_shares: f64,
}

// This logger adds source context to events from one market-maker trading day.
struct MarketMakerLogger<'a> {
    enabled: bool,
    filename: &'a str,
}

// This configuration defines one market-maker simulation candidate.
#[derive(Clone, Copy)]
struct MarketMakerConfig {
    initial_cash: f64,
    bar_volume_limit: f64,
    buy_ttl: u64,
    sell_ttl: u64,
    discount_percent: f64,
    markup_percent: f64,
    bet_size: f64,
}

// These statistics summarize one market-maker simulation across all trading days.
#[derive(Clone)]
struct MarketMakerResult {
    final_value: f64,
    annualized_sharpe: Option<f64>,
    daily_returns: Vec<f64>,
}

// This result pairs one grid configuration with its simulation statistics.
#[derive(Clone)]
struct GridCandidate {
    config: MarketMakerConfig,
    result: MarketMakerResult,
}

// These results identify the configurations favored by return and risk-adjusted return.
struct GridResult {
    highest_return: GridCandidate,
    highest_annualized_sharpe: Option<GridCandidate>,
}

// Backtest a trading strategy.
pub fn run(args: &Args) -> Result<(), Box<dyn Error>> {
    // Sort by filename so every strategy receives the data in chronological order.
    let mut data_paths = args.data_paths.iter().collect::<Vec<_>>();
    data_paths.sort();

    // Load every sorted file before dispatching to the selected strategy.
    let files = data_paths
        .into_iter()
        .map(|path| {
            let contents = fs::read_to_string(path)
                .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
            Ok((path.clone(), contents))
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;

    // Evaluate the selected strategy and print its result to standard output.
    match args.strategy {
        Strategy::BuyAndHold => {
            let final_value = buy_and_hold(&files, args.initial_cash)?;
            println!("{final_value}");
        }
        Strategy::MarketMaker => {
            let result = market_maker(&files, args)?;
            print_market_maker_result(None, &result, market_maker_config(args));
        }
        Strategy::MarketMakerGrid => {
            let result = market_maker_grid(&files, args)?;
            print_grid_result(&result);
        }
    }

    Ok(())
}

// Simulate repeatedly buying below and selling above the current market price.
fn market_maker(
    files: &[(PathBuf, String)],
    args: &Args,
) -> Result<MarketMakerResult, Box<dyn Error>> {
    // Parse each chronological trading day before changing the simulated portfolio.
    let days = parse_days(files)?;
    simulate_market_maker(&days, market_maker_config(args), true)
}

// Evaluate nearby parameter combinations and return the winners by return and Sharpe ratio.
fn market_maker_grid(
    files: &[(PathBuf, String)],
    args: &Args,
) -> Result<GridResult, Box<dyn Error>> {
    // Parse the historical days once because every candidate uses identical market data.
    let days = parse_days(files)?;

    // Count the Cartesian product before allocating its configurations.
    let total_candidates = args.buy_ttls.len()
        * args.sell_ttls.len()
        * args.discount_percentages.len()
        * args.markup_percentages.len()
        * args.bet_sizes.len();
    eprintln!("Searching {total_candidates} market-maker configurations...");

    // Build configurations in search order so winner selection can resolve ties deterministically.
    let mut configs = Vec::with_capacity(total_candidates);
    for &buy_ttl in &args.buy_ttls {
        for &sell_ttl in &args.sell_ttls {
            for &discount_percent in &args.discount_percentages {
                for &markup_percent in &args.markup_percentages {
                    for &bet_size in &args.bet_sizes {
                        configs.push(MarketMakerConfig {
                            initial_cash: args.initial_cash,
                            bar_volume_limit: args.bar_volume_limit,
                            buy_ttl,
                            sell_ttl,
                            discount_percent,
                            markup_percent,
                            bet_size,
                        });
                    }
                }
            }
        }
    }

    // Evaluate independent candidates in parallel and serialize their ordered progress updates.
    let completed_candidates = Mutex::new(0_usize);
    let candidates = configs
        .par_iter()
        .map(|&config| {
            let result =
                simulate_market_maker(&days, config, false).map_err(|error| error.to_string())?;
            let mut completed_candidates = completed_candidates
                .lock()
                .map_err(|error| format!("failed to lock grid-search progress: {error}"))?;
            *completed_candidates += 1;
            if *completed_candidates
                % (args.discount_percentages.len()
                    * args.markup_percentages.len()
                    * args.bet_sizes.len())
                == 0
            {
                let progress_tenths = 1_000 * *completed_candidates / total_candidates;
                eprintln!(
                    "Searched {completed_candidates}/{total_candidates} configurations ({}.{:01}%)",
                    progress_tenths / 10,
                    progress_tenths % 10,
                );
            }
            Ok(GridCandidate { config, result })
        })
        .collect::<Result<Vec<_>, String>>()?;

    // Separate the completed progress display from the winning configurations.
    eprintln!();

    // Search results in configuration order while retaining the first candidate in a tie.
    let mut highest_return = None::<GridCandidate>;
    let mut highest_annualized_sharpe = None::<GridCandidate>;
    for candidate in candidates {
        if highest_return
            .as_ref()
            .is_none_or(|highest| candidate.result.final_value > highest.result.final_value)
        {
            highest_return = Some(candidate.clone());
        }
        if let Some(candidate_sharpe) = candidate.result.annualized_sharpe {
            let exceeds_highest = highest_annualized_sharpe
                .as_ref()
                .and_then(|highest| highest.result.annualized_sharpe)
                .is_none_or(|highest_sharpe| candidate_sharpe > highest_sharpe);
            if exceeds_highest {
                highest_annualized_sharpe = Some(candidate);
            }
        }
    }

    // A return winner always exists, while Sharpe requires at least two daily observations.
    Ok(GridResult {
        highest_return: highest_return.ok_or("the parameter grid contains no valid candidates")?,
        highest_annualized_sharpe,
    })
}

// Simulate one market-maker configuration over each parsed trading day.
fn simulate_market_maker(
    days: &[Day],
    config: MarketMakerConfig,
    log_orders: bool,
) -> Result<MarketMakerResult, Box<dyn Error>> {
    // Compound account value across days while retaining each daily return for risk measurement.
    let mut final_value = config.initial_cash;
    let mut daily_returns = Vec::with_capacity(days.len());
    for day in days {
        let initial_value = final_value;
        final_value =
            simulate_market_maker_day(&day.bars, initial_value, config, &day.filename, log_orders)?;
        daily_returns.push(final_value / initial_value - 1.0_f64);
    }

    Ok(MarketMakerResult {
        final_value,
        annualized_sharpe: annualized_sharpe_ratio(&daily_returns)?,
        daily_returns,
    })
}

// Simulate one market-maker configuration over one trading day.
fn simulate_market_maker_day(
    bars: &[Bar],
    initial_value: f64,
    config: MarketMakerConfig,
    filename: &str,
    log_orders: bool,
) -> Result<f64, Box<dyn Error>> {
    // Start the day with the entire incoming account value available as cash.
    let logger = MarketMakerLogger {
        enabled: log_orders,
        filename,
    };
    let mut available_cash = initial_value;
    let mut available_shares = 0.0_f64;
    let mut buy_orders = Vec::<LimitOrder>::new();
    let mut sell_orders = Vec::<LimitOrder>::new();
    let buy_ttl = i64::try_from(config.buy_ttl)?;
    let sell_ttl = i64::try_from(config.sell_ttl)?;

    // Cancel, fill, and replace orders once for every chronological bar.
    for bar in bars {
        // Cancel pending orders and sell up to one bar's volume at the current close.
        if bar.liquidate {
            available_cash += buy_orders
                .drain(..)
                .map(|order| order.remaining_shares * order.price)
                .sum::<f64>();
            available_shares += sell_orders
                .drain(..)
                .map(|order| order.remaining_shares)
                .sum::<f64>();
            let filled_shares = available_shares.min(config.bar_volume_limit);
            available_cash += filled_shares * bar.close;
            available_shares -= filled_shares;
            logger.liquidation(bar.timestamp, filled_shares, bar.close);
            continue;
        }

        // Return resources reserved by orders older than their configured lifetimes.
        expire_market_maker_orders(
            bar.timestamp,
            buy_ttl,
            sell_ttl,
            &mut available_cash,
            &mut available_shares,
            &mut buy_orders,
            &mut sell_orders,
        );

        // Share one fill budget across every order eligible during this bar.
        let mut remaining_bar_volume = config.bar_volume_limit;

        // Partially fill eligible buy orders while preserving volume for later orders.
        for order in &mut buy_orders {
            if remaining_bar_volume <= 0.0_f64 {
                break;
            }
            if bar.low <= order.price {
                let filled_shares = order.remaining_shares.min(remaining_bar_volume);
                available_shares += filled_shares;
                order.remaining_shares -= filled_shares;
                remaining_bar_volume -= filled_shares;
                logger.execution(bar.timestamp, "buy", filled_shares, order);
            }
        }
        buy_orders.retain(|order| order.remaining_shares > 0.0_f64);

        // Partially fill eligible sell orders with the volume left after buy executions.
        for order in &mut sell_orders {
            if remaining_bar_volume <= 0.0_f64 {
                break;
            }
            if bar.high >= order.price {
                let filled_shares = order.remaining_shares.min(remaining_bar_volume);
                available_cash += filled_shares * order.price;
                order.remaining_shares -= filled_shares;
                remaining_bar_volume -= filled_shares;
                logger.execution(bar.timestamp, "sell", filled_shares, order);
            }
        }
        sell_orders.retain(|order| order.remaining_shares > 0.0_f64);

        // Keep the configured share of liquidation value available for buying.
        let reserved_cash = buy_orders
            .iter()
            .map(|order| order.remaining_shares * order.price)
            .sum::<f64>();
        let reserved_shares = sum_remaining_shares(&sell_orders);
        let liquidation_value =
            available_cash + reserved_cash + (available_shares + reserved_shares) * bar.close;
        let cash_floor = liquidation_value * (1.0_f64 - config.bet_size / 100.0_f64);
        let cash_available_to_bet = (available_cash - cash_floor).max(0.0_f64);

        // Reserve usable cash for the largest whole-share discounted buy order.
        let buy_limit = bar.close * (1.0_f64 - config.discount_percent / 100.0_f64);
        let buy_shares = (cash_available_to_bet / buy_limit).floor();
        if buy_shares >= 1.0_f64 {
            available_cash -= buy_shares * buy_limit;
            buy_orders.push(LimitOrder {
                placed_timestamp: bar.timestamp,
                price: buy_limit,
                remaining_shares: buy_shares,
            });
        }

        // Reserve every available share for one marked-up sell order.
        if available_shares > 0.0_f64 {
            let sell_limit = bar.close * (1.0_f64 + config.markup_percent / 100.0_f64);
            sell_orders.push(LimitOrder {
                placed_timestamp: bar.timestamp,
                price: sell_limit,
                remaining_shares: available_shares,
            });
            available_shares = 0.0_f64;
        }
    }

    // Mark reserved cash and all held shares to the final close.
    let final_price = bars.last().unwrap().close;
    let reserved_cash = buy_orders
        .iter()
        .map(|order| order.remaining_shares * order.price)
        .sum::<f64>();
    let reserved_shares = sum_remaining_shares(&sell_orders);
    let final_value =
        available_cash + reserved_cash + (available_shares + reserved_shares) * final_price;

    Ok(final_value)
}

// Cancel expired orders and return their reserved resources to the portfolio.
fn expire_market_maker_orders(
    timestamp: i64,
    buy_ttl: i64,
    sell_ttl: i64,
    available_cash: &mut f64,
    available_shares: &mut f64,
    buy_orders: &mut Vec<LimitOrder>,
    sell_orders: &mut Vec<LimitOrder>,
) {
    // Refund cash reserved by buy orders that have exceeded their lifetime.
    buy_orders.retain(|order| {
        if timestamp.saturating_sub(order.placed_timestamp) > buy_ttl {
            *available_cash += order.remaining_shares * order.price;
            false
        } else {
            true
        }
    });

    // Release shares reserved by sell orders that have exceeded their lifetime.
    sell_orders.retain(|order| {
        if timestamp.saturating_sub(order.placed_timestamp) > sell_ttl {
            *available_shares += order.remaining_shares;
            false
        } else {
            true
        }
    });
}

// Total the unfilled shares across a collection of limit orders.
fn sum_remaining_shares(orders: &[LimitOrder]) -> f64 {
    // Count only the quantity that remains reserved by each order.
    orders.iter().map(|order| order.remaining_shares).sum()
}

// Format market-maker events only when detailed output is enabled.
impl MarketMakerLogger<'_> {
    // Log a full or partial execution with enough information to correlate the order.
    fn execution(&self, timestamp: i64, side: &str, filled_shares: f64, order: &LimitOrder) {
        if self.enabled {
            let filename = self.filename;
            info!(
                concat!(
                    "{} @ {}: Executed {} shares of {} order from timestamp {} ",
                    "@ ${:.2} ({} remaining)",
                ),
                filename,
                timestamp,
                filled_shares,
                side,
                order.placed_timestamp,
                order.price,
                order.remaining_shares,
            );
        }
    }

    // Log shares sold directly during the end-of-day liquidation window.
    fn liquidation(&self, timestamp: i64, shares: f64, price: f64) {
        if self.enabled && shares > 0.0_f64 {
            let filename = self.filename;
            info!(
                concat!(
                    "{} @ {}: Executed liquidation sell for {} shares ",
                    "@ ${:.2}",
                ),
                filename,
                timestamp,
                shares,
                price,
            );
        }
    }
}

// Calculate annualized excess return per unit of risk from daily return rates.
fn annualized_sharpe_ratio(returns: &[f64]) -> Result<Option<f64>, Box<dyn Error>> {
    // At least two observations are needed to estimate historical return variability.
    if returns.len() < 2 {
        return Ok(None);
    }

    // Use a zero risk-free rate so the metric ranks return per unit of variability.
    let count = f64::from(u32::try_from(returns.len())?);
    let mean_return = returns.iter().sum::<f64>() / count;

    // Estimate historical daily volatility with the sample standard deviation.
    let variance = returns
        .iter()
        .map(|value| (value - mean_return).powi(2))
        .sum::<f64>()
        / (count - 1.0_f64);
    let standard_deviation = variance.sqrt();

    // Give constant-return strategies an ordered result without producing `NaN`.
    let daily_sharpe = if standard_deviation == 0.0_f64 {
        match mean_return.total_cmp(&0.0_f64) {
            std::cmp::Ordering::Greater => f64::INFINITY,
            std::cmp::Ordering::Less => f64::NEG_INFINITY,
            std::cmp::Ordering::Equal => 0.0_f64,
        }
    } else {
        mean_return / standard_deviation
    };

    // Scale the daily ratio under the conventional zero-serial-correlation assumption.
    Ok(Some(daily_sharpe * TRADING_DAYS_PER_YEAR.sqrt()))
}

// Copy market-maker command-line values into one simulation configuration.
fn market_maker_config(args: &Args) -> MarketMakerConfig {
    // Keep simulation code independent from unrelated backtest arguments.
    MarketMakerConfig {
        initial_cash: args.initial_cash,
        bar_volume_limit: args.bar_volume_limit,
        buy_ttl: args.buy_ttl,
        sell_ttl: args.sell_ttl,
        discount_percent: args.discount_percent,
        markup_percent: args.markup_percent,
        bet_size: args.bet_size,
    }
}

// Print one market-maker result with an optional grid-ranking label.
fn print_market_maker_result(
    label: Option<&str>,
    result: &MarketMakerResult,
    config: MarketMakerConfig,
) {
    // Identify a grid winner while leaving single-strategy reports unlabelled.
    if let Some(label) = label {
        println!("{label}");
    }

    // Present summary statistics before the detailed daily returns.
    println!("Final account value: {:.2}", result.final_value);
    print_sharpe_ratio(result.annualized_sharpe);

    // Number days in chronological input order for easy comparison between reports.
    println!("Daily returns:");
    for (index, daily_return) in result.daily_returns.iter().enumerate() {
        println!("  Day {}: {:.2}%", index + 1, 100.0_f64 * daily_return);
    }

    // Finish with the effective simulation parameters.
    print_config(config);
}

// Print both winning grid candidates as a human-readable report.
fn print_grid_result(result: &GridResult) {
    // Give each optimization criterion its own complete section.
    print_market_maker_result(
        Some("Highest return"),
        &result.highest_return.result,
        result.highest_return.config,
    );
    println!();
    if let Some(candidate) = &result.highest_annualized_sharpe {
        print_market_maker_result(
            Some("Highest Sharpe ratio"),
            &candidate.result,
            candidate.config,
        );
    } else {
        println!("Highest Sharpe ratio");
        println!("Annualized Sharpe ratio: unavailable (requires at least two daily returns)");
    }
}

// Print one available annualized Sharpe ratio or explain why it is unavailable.
fn print_sharpe_ratio(sharpe: Option<f64>) {
    match sharpe {
        Some(sharpe) => println!("Annualized Sharpe ratio: {sharpe:.4}"),
        None => {
            println!("Annualized Sharpe ratio: unavailable (requires at least two daily returns)");
        }
    }
}

// Print a labeled market-maker configuration.
fn print_config(config: MarketMakerConfig) {
    // Keep the label separate so the fields can also be reused by grid reports.
    println!("Configuration:");
    print_config_fields(config);
}

// Print the fields of one market-maker configuration.
fn print_config_fields(config: MarketMakerConfig) {
    // Present the shared liquidation schedule alongside the tunable simulation fields.
    println!("  Initial cash: {:.2}", config.initial_cash);
    println!("  Liquidation start time (ET): {LIQUIDATION_START_TIME}");
    println!("  Liquidation end time (ET): {LIQUIDATION_END_TIME}");
    println!("  Bar volume limit: {}", config.bar_volume_limit);
    println!("  Buy TTL: {}", config.buy_ttl);
    println!("  Sell TTL: {}", config.sell_ttl);
    println!("  Discount: {}%", config.discount_percent);
    println!("  Markup: {}%", config.markup_percent);
    println!("  Bet size: {}%", config.bet_size);
}

// Parse the low, high, and closing prices from every input file as one trading day.
fn parse_days(files: &[(PathBuf, String)]) -> Result<Vec<Day>, Box<dyn Error>> {
    // Preserve sorted day and record order while validating each price.
    let mut days = Vec::with_capacity(files.len());
    for (path, contents) in files {
        let mut reader = csv::Reader::from_reader(contents.as_bytes());
        let headers = reader.headers()?;
        let timestamp_index = column_index(headers, path, "date")?;
        let low_index = column_index(headers, path, "low")?;
        let high_index = column_index(headers, path, "high")?;
        let close_index = column_index(headers, path, "close")?;
        let records = reader.records().collect::<Result<Vec<_>, _>>()?;
        if records.is_empty() {
            return Err(format!("{} must contain at least one data row", path.display()).into());
        }
        let mut bars = Vec::with_capacity(records.len());
        for (index, record) in records.iter().enumerate() {
            let line = index + 2;
            let timestamp = parse_timestamp(
                record.get(timestamp_index),
                path,
                &format!("date on line {line}"),
            )?;
            if is_unreliable_early_data(timestamp) {
                continue;
            }
            bars.push(Bar {
                timestamp: timestamp.unix_timestamp(),
                low: parse_price(record.get(low_index), path, &format!("low on line {line}"))?,
                high: parse_price(
                    record.get(high_index),
                    path,
                    &format!("high on line {line}"),
                )?,
                close: parse_price(
                    record.get(close_index),
                    path,
                    &format!("close on line {line}"),
                )?,
                liquidate: is_liquidating(timestamp),
            });
        }

        // Require at least one usable bar after applying the early-data filter.
        if bars.is_empty() {
            return Err(format!(
                "{} must contain at least one row outside the unreliable early-data window",
                path.display(),
            )
            .into());
        }

        days.push(Day {
            filename: path
                .file_name()
                .unwrap_or(path.as_os_str())
                .to_string_lossy()
                .into_owned(),
            bars,
        });
    }
    if days.is_empty() {
        return Err("at least one data file is required".into());
    }

    Ok(days)
}

// Parse one required Unix timestamp with a contextual error.
fn parse_timestamp(
    value: Option<&str>,
    path: &std::path::Path,
    description: &str,
) -> Result<OffsetDateTime, Box<dyn Error>> {
    // Reject missing, noninteger, and out-of-range timestamps before simulation.
    let value = value.ok_or_else(|| format!("{} is missing its {description}", path.display()))?;
    let timestamp = value
        .parse::<i64>()
        .map_err(|error| format!("invalid {description} in {}: {error}", path.display()))?;
    OffsetDateTime::from_unix_timestamp(timestamp)
        .map_err(|error| format!("invalid {description} in {}: {error}", path.display()).into())
}

// Identify trade reports that cannot be treated as contemporaneous fill opportunities.
fn is_unreliable_early_data(timestamp: OffsetDateTime) -> bool {
    let eastern_time = timestamp.to_timezone(NEW_YORK).time();

    eastern_time >= UNRELIABLE_DATA_START_TIME && eastern_time < UNRELIABLE_DATA_END_TIME
}

// Identify bars inside the shared Eastern-time liquidation window.
fn is_liquidating(timestamp: OffsetDateTime) -> bool {
    let eastern_time = timestamp.to_timezone(NEW_YORK).time();

    eastern_time >= LIQUIDATION_START_TIME && eastern_time < LIQUIDATION_END_TIME
}

// Locate one required CSV price column.
fn column_index(
    headers: &csv::StringRecord,
    path: &std::path::Path,
    name: &str,
) -> Result<usize, Box<dyn Error>> {
    // Report the source file when a required market-data field is absent.
    headers
        .iter()
        .position(|header| header == name)
        .ok_or_else(|| format!("{} must contain a {name} column", path.display()).into())
}

// Calculate the final value produced by buying first and marking to the final close.
fn buy_and_hold(files: &[(PathBuf, String)], initial_cash: f64) -> Result<f64, Box<dyn Error>> {
    // Read the first open and final close while requiring data in every input file.
    let mut first_open = None;
    let mut last_close = None;
    for (path, contents) in files {
        let mut reader = csv::Reader::from_reader(contents.as_bytes());
        let headers = reader.headers()?;
        let timestamp_index = column_index(headers, path, "date")?;
        let open_index = headers
            .iter()
            .position(|header| header == "open")
            .ok_or_else(|| format!("{} must contain an open column", path.display()))?;
        let close_index = headers
            .iter()
            .position(|header| header == "close")
            .ok_or_else(|| format!("{} must contain a close column", path.display()))?;
        let records = reader.records().collect::<Result<Vec<_>, _>>()?;
        let mut first_record = None;
        let mut last_record = None;
        for (index, record) in records.iter().enumerate() {
            let line = index + 2;
            let timestamp = parse_timestamp(
                record.get(timestamp_index),
                path,
                &format!("date on line {line}"),
            )?;
            if is_unreliable_early_data(timestamp) {
                continue;
            }
            first_record.get_or_insert(record);
            last_record = Some(record);
        }
        let first_record = first_record.ok_or_else(|| {
            format!(
                "{} must contain at least one row outside the unreliable early-data window",
                path.display(),
            )
        })?;
        let last_record = last_record.unwrap();

        // Parse finite positive prices before using the boundary records.
        let open = parse_price(first_record.get(open_index), path, "opening")?;
        let close = parse_price(last_record.get(close_index), path, "closing")?;
        first_open.get_or_insert(open);
        last_close = Some(close);
    }

    let first_open = first_open.ok_or("at least one data file is required")?;
    let shares = initial_cash / first_open;
    Ok(shares * last_close.unwrap())
}

// Parse one required boundary price with a contextual error.
fn parse_price(
    value: Option<&str>,
    path: &std::path::Path,
    description: &str,
) -> Result<f64, Box<dyn Error>> {
    // Reject missing, nonnumeric, nonfinite, and nonpositive prices consistently.
    let value =
        value.ok_or_else(|| format!("{} is missing its {description} price", path.display()))?;
    let price = value
        .parse::<f64>()
        .map_err(|error| format!("invalid {description} price in {}: {error}", path.display()))?;
    if !price.is_finite() || price <= 0.0_f64 {
        return Err(format!(
            "{description} price in {} must be finite and positive",
            path.display(),
        )
        .into());
    }

    Ok(price)
}

// Parse a finite positive floating-point command-line argument.
fn parse_positive_f64(value: &str) -> Result<f64, String> {
    // Reject values that cannot represent usable starting capital.
    let value = value.parse::<f64>().map_err(|error| error.to_string())?;
    if !value.is_finite() || value <= 0.0_f64 {
        return Err("value must be finite and greater than zero".to_string());
    }

    Ok(value)
}

// Parse a finite nonnegative floating-point command-line argument.
fn parse_nonnegative_f64(value: &str) -> Result<f64, String> {
    // Permit a zero markup while rejecting negative and nonfinite percentages.
    let value = value.parse::<f64>().map_err(|error| error.to_string())?;
    if !value.is_finite() || value < 0.0_f64 {
        return Err("value must be finite and nonnegative".to_string());
    }

    Ok(value)
}

// Parse a discount percentage that always produces a positive limit price.
fn parse_discount_percent(value: &str) -> Result<f64, String> {
    // Reject discounts at or above one hundred percent to keep buy limits valid.
    let value = parse_nonnegative_f64(value)?;
    if value >= 100.0_f64 {
        return Err("value must be less than 100".to_string());
    }

    Ok(value)
}

// Parse a percentage that may span the entire inclusive range from zero to one hundred.
fn parse_percent(value: &str) -> Result<f64, String> {
    // Reject percentages outside the range needed to represent a portfolio fraction.
    let value = parse_nonnegative_f64(value)?;
    if value > 100.0_f64 {
        return Err("value must be at most 100".to_string());
    }

    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::{
        Args, Bar, Day, MarketMakerConfig, Strategy, TRADING_DAYS_PER_YEAR,
        annualized_sharpe_ratio, buy_and_hold, is_unreliable_early_data, market_maker,
        market_maker_grid, parse_days, simulate_market_maker,
    };
    use crate::{Cli, Subcommand};
    use clap::Parser;
    use std::path::PathBuf;
    use time::OffsetDateTime;

    #[test]
    fn parse_backtest_subcommand() {
        // Confirm the backtest mode accepts the buy-and-hold strategy.
        let cli = Cli::try_parse_from([
            "stockholm",
            "backtest",
            "--strategy",
            "buy-and-hold",
            "--data-paths",
            "monday.csv",
            "tuesday.csv",
        ])
        .unwrap();

        let Some(Subcommand::Backtest(args)) = cli.command else {
            panic!("expected backtest subcommand");
        };
        assert_eq!(args.strategy, Strategy::BuyAndHold);
        assert_eq!(
            args.data_paths,
            vec![PathBuf::from("monday.csv"), PathBuf::from("tuesday.csv")],
        );
        assert!((args.initial_cash - 1_000_000.0).abs() < f64::EPSILON);
        assert_eq!(args.buy_ttl, 3_600);
        assert_eq!(args.sell_ttl, 14_400);
        assert!((args.discount_percent - 0.25).abs() < f64::EPSILON);
        assert!((args.markup_percent - 0.25).abs() < f64::EPSILON);
        assert!((args.bet_size - 80.0).abs() < f64::EPSILON);
        assert_eq!(args.buy_ttls.len(), 12);
        assert_eq!(args.buy_ttls.first(), Some(&5));
        assert_eq!(args.buy_ttls.last(), Some(&86_400));
        assert_eq!(args.sell_ttls.len(), 12);
        assert_eq!(args.sell_ttls.first(), Some(&5));
        assert_eq!(args.sell_ttls.last(), Some(&86_400));
        assert_eq!(args.discount_percentages.len(), 7);
        assert_eq!(args.discount_percentages.first(), Some(&0.01_f64));
        assert_eq!(args.discount_percentages.last(), Some(&10.0_f64));
        assert_eq!(args.markup_percentages.len(), 7);
        assert_eq!(args.markup_percentages.first(), Some(&0.01_f64));
        assert_eq!(args.markup_percentages.last(), Some(&10.0_f64));
        assert_eq!(args.bet_sizes, vec![80.0_f64, 90.0_f64, 100.0_f64]);
        assert!((args.bar_volume_limit - 1_000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn mark_liquidation_by_eastern_time() {
        // Apply the shared inclusive start and exclusive end to historical bars.
        let files = vec![(
            PathBuf::from("prices.csv"),
            concat!(
                "date,low,high,close\n",
                "1786391099,100,100,100\n",
                "1786391100,100,100,100\n",
                "1786406699,100,100,100\n",
                "1786406700,100,100,100\n",
            )
            .to_string(),
        )];

        let days = parse_days(&files).unwrap();
        let liquidation_flags = days[0]
            .bars
            .iter()
            .map(|bar| bar.liquidate)
            .collect::<Vec<_>>();

        assert_eq!(liquidation_flags, vec![false, true, true, false]);
    }

    #[test]
    fn exclude_early_data_across_eastern_time_offsets() {
        // Apply the same inclusive start and exclusive end in daylight and standard time.
        for start in [1_784_016_000_i64, 1_767_949_200_i64] {
            assert!(!is_unreliable_early_data(
                OffsetDateTime::from_unix_timestamp(start - 1).unwrap(),
            ));
            assert!(is_unreliable_early_data(
                OffsetDateTime::from_unix_timestamp(start).unwrap(),
            ));
            assert!(is_unreliable_early_data(
                OffsetDateTime::from_unix_timestamp(start + 899).unwrap(),
            ));
            assert!(!is_unreliable_early_data(
                OffsetDateTime::from_unix_timestamp(start + 900).unwrap(),
            ));
        }
    }

    #[test]
    fn discard_unreliable_rows_before_parsing_market_maker_prices() {
        // Ignore malformed prices in the excluded window and retain the 4:15 a.m. boundary.
        let files = vec![(
            PathBuf::from("prices.csv"),
            concat!(
                "date,low,high,close\n",
                "1784016000,invalid,invalid,invalid\n",
                "1784016899,invalid,invalid,invalid\n",
                "1784016900,100,101,100\n",
            )
            .to_string(),
        )];

        let days = parse_days(&files).unwrap();

        assert_eq!(days[0].bars.len(), 1);
        assert_eq!(days[0].bars[0].timestamp, 1_784_016_900);
    }

    #[test]
    fn calculate_buy_and_hold_from_chronological_files() {
        // Confirm the strategy uses the first open and final close it receives.
        let files = vec![
            (
                PathBuf::from("monday.csv"),
                "date,open,close\n1000,100,110\n1001,110,120\n".to_string(),
            ),
            (
                PathBuf::from("tuesday.csv"),
                "date,open,close\n2000,200,210\n2001,210,230\n".to_string(),
            ),
        ];

        assert!((buy_and_hold(&files, 1_000.0).unwrap() - 2_300.0).abs() < f64::EPSILON);
    }

    #[test]
    fn discard_unreliable_rows_before_parsing_buy_and_hold_prices() {
        // Invest at the first reliable open without validating excluded trade-report prices.
        let files = vec![(
            PathBuf::from("prices.csv"),
            concat!(
                "date,open,close\n",
                "1784016000,invalid,invalid\n",
                "1784016900,100,120\n",
            )
            .to_string(),
        )];

        assert!((buy_and_hold(&files, 1_000.0).unwrap() - 1_200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn fill_market_maker_orders() {
        // Confirm discounted buys and marked-up sells reserve and return resources.
        let files = vec![(
            PathBuf::from("prices.csv"),
            concat!(
                "date,low,high,close\n",
                "36000,100,100,100\n",
                "36001,99,100,100\n",
                "36002,100,101,100\n",
            )
            .to_string(),
        )];
        let args = market_maker_args(1_000.0, 3_600, 14_400);

        let result = market_maker(&files, &args).unwrap();
        assert!((result.final_value - 1_020.0).abs() < f64::EPSILON);
        assert_eq!(result.annualized_sharpe, None);
    }

    #[test]
    fn limit_market_maker_bet_size() {
        // Confirm the configured cash floor limits the size of new buy orders.
        let files = vec![(
            PathBuf::from("prices.csv"),
            concat!(
                "date,low,high,close\n",
                "36000,100,100,100\n",
                "36001,99,100,100\n",
                "36002,100,101,100\n",
            )
            .to_string(),
        )];
        let mut args = market_maker_args(1_000.0, 3_600, 14_400);
        args.bet_size = 80.0_f64;

        let result = market_maker(&files, &args).unwrap();
        assert!((result.final_value - 1_016.0).abs() < f64::EPSILON);
    }

    #[test]
    fn compound_market_maker_days() {
        // Confirm each file produces one daily return and passes its final value to the next day.
        let files = vec![
            (
                PathBuf::from("monday.csv"),
                "date,low,high,close\n36000,100,100,100\n36001,99,100,100\n36002,100,101,100\n"
                    .to_string(),
            ),
            (
                PathBuf::from("tuesday.csv"),
                "date,low,high,close\n122400,100,100,100\n122401,99,100,100\n122402,100,101,100\n"
                    .to_string(),
            ),
        ];
        let args = market_maker_args(1_000.0, 3_600, 14_400);

        let result = market_maker(&files, &args).unwrap();
        assert!((result.final_value - 1_040.0).abs() < f64::EPSILON);
        assert!(
            result
                .annualized_sharpe
                .is_some_and(|sharpe| sharpe.is_finite() && sharpe > 0.0_f64),
        );
    }

    #[test]
    fn refund_expired_market_maker_orders() {
        // Confirm canceling an expired buy restores its reserved cash.
        let files = vec![(
            PathBuf::from("prices.csv"),
            "date,low,high,close\n36000,100,100,100\n36001,200,200,200\n".to_string(),
        )];
        let args = market_maker_args(1_000.0, 0, 14_400);

        let result = market_maker(&files, &args).unwrap();
        assert!((result.final_value - 1_000.0).abs() < f64::EPSILON);
        assert_eq!(result.annualized_sharpe, None);
    }

    #[test]
    fn liquidate_market_maker_inventory() {
        // Confirm the liquidation window cancels orders and sells every held share at the close.
        let bars = vec![
            Bar {
                timestamp: 1_000,
                low: 100.0,
                high: 100.0,
                close: 100.0,
                liquidate: false,
            },
            Bar {
                timestamp: 1_001,
                low: 99.0,
                high: 100.0,
                close: 100.0,
                liquidate: false,
            },
            Bar {
                timestamp: 1_002,
                low: 90.0,
                high: 90.0,
                close: 90.0,
                liquidate: true,
            },
        ];
        let config = MarketMakerConfig {
            initial_cash: 1_000.0,
            bar_volume_limit: 1_000.0,
            buy_ttl: 3_600,
            sell_ttl: 14_400,
            discount_percent: 1.0,
            markup_percent: 1.0,
            bet_size: 100.0,
        };

        let result = simulate_market_maker(
            &[Day {
                filename: "prices.csv".to_string(),
                bars,
            }],
            config,
            false,
        )
        .unwrap();
        assert!((result.final_value - 910.0).abs() < f64::EPSILON);
    }

    #[test]
    fn select_best_market_maker_grid_candidate() {
        // Confirm the grid chooses the reproducible parameters with the highest final value.
        let files = vec![(
            PathBuf::from("prices.csv"),
            concat!(
                "date,low,high,close\n",
                "36000,100,100,100\n",
                "36001,99,100,100\n",
                "36002,100,102,100\n",
            )
            .to_string(),
        )];
        let args = market_maker_args(1_000.0, 10, 10);
        let result = market_maker_grid(&files, &args).unwrap();

        assert!((result.highest_return.result.final_value - 1_020.0).abs() < f64::EPSILON);
        assert!((result.highest_return.config.discount_percent - 1.0).abs() < f64::EPSILON);
        assert!((result.highest_return.config.markup_percent - 1.0).abs() < f64::EPSILON);
        assert_eq!(result.highest_return.config.buy_ttl, 5);
        assert_eq!(result.highest_return.config.sell_ttl, 5);
        assert!(result.highest_annualized_sharpe.is_none());
    }

    #[test]
    fn partially_fill_market_maker_orders() {
        // Confirm an eligible order needs multiple bars when it exceeds the volume limit.
        let bars = vec![
            Bar {
                timestamp: 1_000,
                low: 100.0,
                high: 100.0,
                close: 100.0,
                liquidate: false,
            },
            Bar {
                timestamp: 1_001,
                low: 99.0,
                high: 100.0,
                close: 100.0,
                liquidate: false,
            },
            Bar {
                timestamp: 1_002,
                low: 100.0,
                high: 101.0,
                close: 100.0,
                liquidate: false,
            },
        ];
        let config = MarketMakerConfig {
            initial_cash: 1_000.0,
            bar_volume_limit: 5.0,
            buy_ttl: 3_600,
            sell_ttl: 14_400,
            discount_percent: 1.0,
            markup_percent: 1.0,
            bet_size: 100.0,
        };

        let result = simulate_market_maker(
            &[Day {
                filename: "prices.csv".to_string(),
                bars,
            }],
            config,
            false,
        )
        .unwrap();
        assert!((result.final_value - 1_010.0).abs() < f64::EPSILON);
    }

    #[test]
    fn share_volume_limit_across_market_maker_orders() {
        // Create two pending buy orders before making both eligible on the same bar.
        let bars = vec![
            Bar {
                timestamp: 1_000,
                low: 100.0,
                high: 100.0,
                close: 100.0,
                liquidate: false,
            },
            Bar {
                timestamp: 1_001,
                low: 99.0,
                high: 100.0,
                close: 100.0,
                liquidate: false,
            },
            Bar {
                timestamp: 1_002,
                low: 99.0,
                high: 100.0,
                close: 100.0,
                liquidate: false,
            },
            Bar {
                timestamp: 1_003,
                low: 100.0,
                high: 101.0,
                close: 100.0,
                liquidate: false,
            },
            Bar {
                timestamp: 1_004,
                low: 99.0,
                high: 100.0,
                close: 100.0,
                liquidate: false,
            },
            Bar {
                timestamp: 1_005,
                low: 90.0,
                high: 90.0,
                close: 90.0,
                liquidate: true,
            },
        ];
        let config = MarketMakerConfig {
            initial_cash: 1_000.0,
            bar_volume_limit: 2.0,
            buy_ttl: 3_600,
            sell_ttl: 14_400,
            discount_percent: 1.0,
            markup_percent: 1.0,
            bet_size: 100.0,
        };

        // Confirm the two eligible buys consume at most two shares of volume in total.
        let result = simulate_market_maker(
            &[Day {
                filename: "prices.csv".to_string(),
                bars,
            }],
            config,
            false,
        )
        .unwrap();
        assert!((result.final_value - 968.0).abs() < f64::EPSILON);
    }

    #[test]
    fn calculate_annualized_sharpe_from_daily_returns() {
        // Annualize mean excess return divided by the sample standard deviation.
        let sharpe = annualized_sharpe_ratio(&[0.1_f64, 0.2_f64])
            .unwrap()
            .unwrap();
        let expected = (0.15_f64 / 0.005_f64.sqrt()) * TRADING_DAYS_PER_YEAR.sqrt();

        assert!((sharpe - expected).abs() < 1e-12_f64);
        assert_eq!(
            annualized_sharpe_ratio(&[0.0_f64, 0.0_f64]).unwrap(),
            Some(0.0_f64),
        );
        assert_eq!(annualized_sharpe_ratio(&[0.1_f64]).unwrap(), None);
    }

    // Construct focused market-maker settings without invoking command-line parsing.
    fn market_maker_args(initial_cash: f64, buy_ttl: u64, sell_ttl: u64) -> Args {
        Args {
            strategy: Strategy::MarketMaker,
            data_paths: Vec::new(),
            initial_cash,
            buy_ttl,
            sell_ttl,
            discount_percent: 1.0,
            markup_percent: 1.0,
            bet_size: 100.0,
            buy_ttls: vec![
                5, 15, 30, 60, 120, 300, 900, 3_600, 7_200, 14_400, 43_200, 86_400,
            ],
            sell_ttls: vec![
                5, 15, 30, 60, 120, 300, 900, 3_600, 7_200, 14_400, 43_200, 86_400,
            ],
            discount_percentages: vec![0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            markup_percentages: vec![0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            bet_sizes: vec![100.0],
            bar_volume_limit: 1_000.0,
        }
    }
}
