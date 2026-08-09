use clap::{Args as ClapArgs, ValueEnum};
use std::{error::Error, fs, path::PathBuf};

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

    /// Seconds before each file's final timestamp when liquidation begins.
    #[arg(long, default_value_t = 900)]
    liquidation_seconds: u64,

    /// Maximum shares of each eligible order filled per bar. Used by both market-maker strategies.
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
    remaining: f64,
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
    liquidation_seconds: u64,
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
    sharpe: f64,
    daily_returns: Vec<f64>,
}

// This result pairs one grid configuration with its simulation statistics.
struct GridCandidate {
    config: MarketMakerConfig,
    result: MarketMakerResult,
}

// These results identify the configurations favored by return and risk-adjusted return.
struct GridResult {
    highest_return: GridCandidate,
    highest_sharpe: GridCandidate,
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
            print_market_maker_result(&result, market_maker_config(args));
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
    let days = parse_days(files, args.liquidation_seconds)?;
    simulate_market_maker(&days, market_maker_config(args), true)
}

// Evaluate nearby parameter combinations and return the winners by return and Sharpe ratio.
fn market_maker_grid(
    files: &[(PathBuf, String)],
    args: &Args,
) -> Result<GridResult, Box<dyn Error>> {
    // Parse the historical days once because every candidate uses identical market data.
    let days = parse_days(files, args.liquidation_seconds)?;

    // Track completed candidates while keeping progress messages separate from CSV output.
    let candidates_per_ttl_pair =
        args.discount_percentages.len() * args.markup_percentages.len() * args.bet_sizes.len();
    let total_candidates = args.buy_ttls.len() * args.sell_ttls.len() * candidates_per_ttl_pair;
    let mut completed_candidates = 0_usize;
    eprintln!("Searching {total_candidates} market-maker configurations...");

    // Search the Cartesian product while retaining the first candidate in a tie.
    let mut highest_return = None::<GridCandidate>;
    let mut highest_sharpe = None::<GridCandidate>;
    for &buy_ttl in &args.buy_ttls {
        for &sell_ttl in &args.sell_ttls {
            for &discount_percent in &args.discount_percentages {
                for &markup_percent in &args.markup_percentages {
                    for &bet_size in &args.bet_sizes {
                        let config = MarketMakerConfig {
                            initial_cash: args.initial_cash,
                            liquidation_seconds: args.liquidation_seconds,
                            bar_volume_limit: args.bar_volume_limit,
                            buy_ttl,
                            sell_ttl,
                            discount_percent,
                            markup_percent,
                            bet_size,
                        };
                        let result = simulate_market_maker(&days, config, false)?;
                        if highest_return.as_ref().is_none_or(|candidate| {
                            result.final_value > candidate.result.final_value
                        }) {
                            highest_return = Some(GridCandidate {
                                config,
                                result: result.clone(),
                            });
                        }
                        if highest_sharpe
                            .as_ref()
                            .is_none_or(|candidate| result.sharpe > candidate.result.sharpe)
                        {
                            highest_sharpe = Some(GridCandidate { config, result });
                        }
                    }
                }
            }
            completed_candidates += candidates_per_ttl_pair;
            let progress_tenths = 1_000 * completed_candidates / total_candidates;
            eprintln!(
                "Searched {completed_candidates}/{total_candidates} configurations ({}.{:01}%)",
                progress_tenths / 10,
                progress_tenths % 10,
            );
        }
    }

    // Separate the completed progress display from the winning configurations.
    eprintln!();

    // Both winners exist whenever the parameter grid contains at least one candidate.
    Ok(GridResult {
        highest_return: highest_return.ok_or("the parameter grid contains no valid candidates")?,
        highest_sharpe: highest_sharpe.ok_or("the parameter grid contains no valid candidates")?,
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
        sharpe: sharpe_ratio(&daily_returns)?,
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
                .map(|order| order.remaining * order.price)
                .sum::<f64>();
            available_shares += sell_orders
                .drain(..)
                .map(|order| order.remaining)
                .sum::<f64>();
            let filled_shares = available_shares.min(config.bar_volume_limit);
            available_cash += filled_shares * bar.close;
            available_shares -= filled_shares;
            logger.liquidation(bar.timestamp, filled_shares, bar.close);
            continue;
        }

        // Return resources reserved by orders older than their configured lifetimes.
        buy_orders.retain(|order| {
            if bar.timestamp.saturating_sub(order.placed_timestamp) > buy_ttl {
                available_cash += order.remaining * order.price;
                false
            } else {
                true
            }
        });
        sell_orders.retain(|order| {
            if bar.timestamp.saturating_sub(order.placed_timestamp) > sell_ttl {
                available_shares += order.remaining;
                false
            } else {
                true
            }
        });

        // Partially fill each eligible buy order by at most one bar's configured volume.
        for order in &mut buy_orders {
            if bar.low <= order.price {
                let filled_shares = order.remaining.min(config.bar_volume_limit);
                available_shares += filled_shares;
                order.remaining -= filled_shares;
                logger.execution(bar.timestamp, "buy", filled_shares, order);
            }
        }
        buy_orders.retain(|order| order.remaining > 0.0_f64);

        // Partially fill each eligible sell order by at most one bar's configured volume.
        for order in &mut sell_orders {
            if bar.high >= order.price {
                let filled_shares = order.remaining.min(config.bar_volume_limit);
                available_cash += filled_shares * order.price;
                order.remaining -= filled_shares;
                logger.execution(bar.timestamp, "sell", filled_shares, order);
            }
        }
        sell_orders.retain(|order| order.remaining > 0.0_f64);

        // Keep the configured share of liquidation value available for buying.
        let reserved_cash = buy_orders
            .iter()
            .map(|order| order.remaining * order.price)
            .sum::<f64>();
        let reserved_shares = sell_orders.iter().map(|order| order.remaining).sum::<f64>();
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
                remaining: buy_shares,
            });
        }

        // Reserve every available share for one marked-up sell order.
        if available_shares > 0.0_f64 {
            let sell_limit = bar.close * (1.0_f64 + config.markup_percent / 100.0_f64);
            sell_orders.push(LimitOrder {
                placed_timestamp: bar.timestamp,
                price: sell_limit,
                remaining: available_shares,
            });
            available_shares = 0.0_f64;
        }
    }

    // Mark reserved cash and all held shares to the final close.
    let final_price = bars.last().unwrap().close;
    let reserved_cash = buy_orders
        .iter()
        .map(|order| order.remaining * order.price)
        .sum::<f64>();
    let reserved_shares = sell_orders.iter().map(|order| order.remaining).sum::<f64>();
    let final_value =
        available_cash + reserved_cash + (available_shares + reserved_shares) * final_price;

    Ok(final_value)
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
                order.remaining,
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

// Calculate unannualized risk-adjusted return from daily return rates.
fn sharpe_ratio(returns: &[f64]) -> Result<f64, Box<dyn Error>> {
    // Use the population moments because the supplied days are the full backtest period.
    let count = f64::from(u32::try_from(returns.len())?);
    let mean = returns.iter().sum::<f64>() / count;
    let variance = returns
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / count;
    let standard_deviation = variance.sqrt();

    // Give constant-return strategies an ordered result without producing `NaN` for zero returns.
    let sharpe = if standard_deviation == 0.0_f64 {
        match mean.total_cmp(&0.0_f64) {
            std::cmp::Ordering::Greater => f64::INFINITY,
            std::cmp::Ordering::Less => f64::NEG_INFINITY,
            std::cmp::Ordering::Equal => 0.0_f64,
        }
    } else {
        mean / standard_deviation
    };

    Ok(sharpe)
}

// Copy market-maker command-line values into one simulation configuration.
fn market_maker_config(args: &Args) -> MarketMakerConfig {
    // Keep simulation code independent from unrelated backtest arguments.
    MarketMakerConfig {
        initial_cash: args.initial_cash,
        liquidation_seconds: args.liquidation_seconds,
        bar_volume_limit: args.bar_volume_limit,
        buy_ttl: args.buy_ttl,
        sell_ttl: args.sell_ttl,
        discount_percent: args.discount_percent,
        markup_percent: args.markup_percent,
        bet_size: args.bet_size,
    }
}

// Print one market-maker result as a human-readable report.
fn print_market_maker_result(result: &MarketMakerResult, config: MarketMakerConfig) {
    // Present the summary before the effective simulation parameters.
    println!("Final account value: {:.2}", result.final_value);
    println!("Sharpe ratio: {:.4}", result.sharpe);
    println!();
    print_config(config);
}

// Print both winning grid candidates as a human-readable report.
fn print_grid_result(result: &GridResult) {
    // Give each optimization criterion its own complete section.
    print_grid_candidate("Highest return", &result.highest_return);
    println!();
    print_grid_candidate("Highest Sharpe ratio", &result.highest_sharpe);
}

// Print one winning grid candidate and its daily return series.
fn print_grid_candidate(label: &str, candidate: &GridCandidate) {
    // Present summary statistics before the detailed daily returns and configuration.
    println!("{label}");
    println!("Final account value: {:.2}", candidate.result.final_value);
    println!("Sharpe ratio: {:.4}", candidate.result.sharpe);
    println!("Daily returns:");
    for (index, daily_return) in candidate.result.daily_returns.iter().enumerate() {
        println!("  Day {}: {:.2}%", index + 1, 100.0_f64 * daily_return);
    }
    println!("Configuration:");
    print_config_fields(candidate.config);
}

// Print a labeled market-maker configuration.
fn print_config(config: MarketMakerConfig) {
    // Keep the label separate so the fields can also be reused by grid reports.
    println!("Configuration:");
    print_config_fields(config);
}

// Print the fields of one market-maker configuration.
fn print_config_fields(config: MarketMakerConfig) {
    // Follow command-line argument order to make reproducing the run straightforward.
    println!("  Initial cash: {:.2}", config.initial_cash);
    println!("  Liquidation seconds: {}", config.liquidation_seconds);
    println!("  Bar volume limit: {}", config.bar_volume_limit);
    println!("  Buy TTL: {}", config.buy_ttl);
    println!("  Sell TTL: {}", config.sell_ttl);
    println!("  Discount: {}%", config.discount_percent);
    println!("  Markup: {}%", config.markup_percent);
    println!("  Bet size: {}%", config.bet_size);
}

// Parse the low, high, and closing prices from every input file as one trading day.
fn parse_days(
    files: &[(PathBuf, String)],
    liquidation_seconds: u64,
) -> Result<Vec<Day>, Box<dyn Error>> {
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
            bars.push(Bar {
                timestamp: parse_timestamp(
                    record.get(timestamp_index),
                    path,
                    &format!("date on line {line}"),
                )?,
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
                liquidate: false,
            });
        }

        // Mark bars within the configured elapsed-time window before the final timestamp.
        let liquidation_seconds = i64::try_from(liquidation_seconds)?;
        let final_timestamp = bars
            .last()
            .ok_or_else(|| format!("{} must contain at least one data row", path.display()))?
            .timestamp;
        for bar in &mut bars {
            bar.liquidate = liquidation_seconds > 0
                && final_timestamp.saturating_sub(bar.timestamp) < liquidation_seconds;
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
) -> Result<i64, Box<dyn Error>> {
    // Reject missing and noninteger timestamps before they reach simulation timing and logs.
    let value = value.ok_or_else(|| format!("{} is missing its {description}", path.display()))?;
    value
        .parse::<i64>()
        .map_err(|error| format!("invalid {description} in {}: {error}", path.display()).into())
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
        let open_index = headers
            .iter()
            .position(|header| header == "open")
            .ok_or_else(|| format!("{} must contain an open column", path.display()))?;
        let close_index = headers
            .iter()
            .position(|header| header == "close")
            .ok_or_else(|| format!("{} must contain a close column", path.display()))?;
        let records = reader.records().collect::<Result<Vec<_>, _>>()?;
        let first_record = records
            .first()
            .ok_or_else(|| format!("{} must contain at least one data row", path.display()))?;
        let last_record = records.last().unwrap();

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
        Args, Bar, Day, MarketMakerConfig, Strategy, buy_and_hold, market_maker, market_maker_grid,
        parse_days, sharpe_ratio, simulate_market_maker,
    };
    use crate::{Cli, Subcommand};
    use clap::Parser;
    use std::path::PathBuf;

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
        assert_eq!(args.liquidation_seconds, 900);
        assert!((args.bar_volume_limit - 1_000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn mark_liquidation_by_elapsed_time() {
        // Exclude an adjacent row separated from the final bar by more than the time window.
        let files = vec![(
            PathBuf::from("prices.csv"),
            concat!(
                "date,low,high,close\n",
                "1000,100,100,100\n",
                "1001,100,100,100\n",
                "2000,100,100,100\n",
            )
            .to_string(),
        )];

        let days = parse_days(&files, 10).unwrap();
        let liquidation_flags = days[0]
            .bars
            .iter()
            .map(|bar| bar.liquidate)
            .collect::<Vec<_>>();

        assert_eq!(liquidation_flags, vec![false, false, true]);
    }

    #[test]
    fn calculate_buy_and_hold_from_chronological_files() {
        // Confirm the strategy uses the first open and final close it receives.
        let files = vec![
            (
                PathBuf::from("monday.csv"),
                "open,close\n100,110\n110,120\n".to_string(),
            ),
            (
                PathBuf::from("tuesday.csv"),
                "open,close\n200,210\n210,230\n".to_string(),
            ),
        ];

        assert!((buy_and_hold(&files, 1_000.0).unwrap() - 2_300.0).abs() < f64::EPSILON);
    }

    #[test]
    fn fill_market_maker_orders() {
        // Confirm discounted buys and marked-up sells reserve and return resources.
        let files = vec![(
            PathBuf::from("prices.csv"),
            concat!(
                "date,low,high,close\n",
                "1000,100,100,100\n",
                "1001,99,100,100\n",
                "1002,100,101,100\n",
            )
            .to_string(),
        )];
        let args = market_maker_args(1_000.0, 3_600, 14_400);

        let result = market_maker(&files, &args).unwrap();
        assert!((result.final_value - 1_020.0).abs() < f64::EPSILON);
        assert!(result.sharpe.is_infinite() && result.sharpe.is_sign_positive());
    }

    #[test]
    fn limit_market_maker_bet_size() {
        // Confirm the configured cash floor limits the size of new buy orders.
        let files = vec![(
            PathBuf::from("prices.csv"),
            concat!(
                "date,low,high,close\n",
                "1000,100,100,100\n",
                "1001,99,100,100\n",
                "1002,100,101,100\n",
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
                "date,low,high,close\n1000,100,100,100\n1001,99,100,100\n1002,100,101,100\n"
                    .to_string(),
            ),
            (
                PathBuf::from("tuesday.csv"),
                "date,low,high,close\n2000,100,100,100\n2001,99,100,100\n2002,100,101,100\n"
                    .to_string(),
            ),
        ];
        let args = market_maker_args(1_000.0, 3_600, 14_400);

        let result = market_maker(&files, &args).unwrap();
        assert!((result.final_value - 1_040.0).abs() < f64::EPSILON);
        assert!(result.sharpe.is_finite() && result.sharpe > 0.0_f64);
    }

    #[test]
    fn refund_expired_market_maker_orders() {
        // Confirm canceling an expired buy restores its reserved cash.
        let files = vec![(
            PathBuf::from("prices.csv"),
            "date,low,high,close\n1000,100,100,100\n1001,200,200,200\n".to_string(),
        )];
        let args = market_maker_args(1_000.0, 0, 14_400);

        let result = market_maker(&files, &args).unwrap();
        assert!((result.final_value - 1_000.0).abs() < f64::EPSILON);
        assert!(result.sharpe.abs() < f64::EPSILON);
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
            liquidation_seconds: 900,
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
                "1000,100,100,100\n",
                "1001,99,100,100\n",
                "1002,100,102,100\n",
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
        assert!(result.highest_sharpe.result.sharpe.is_infinite());
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
            liquidation_seconds: 900,
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
    fn calculate_sharpe_from_daily_returns() {
        // Confirm Sharpe uses the mean and population standard deviation of daily returns.
        let sharpe = sharpe_ratio(&[0.1_f64, 0.2_f64]).unwrap();

        assert!((sharpe - 3.0_f64).abs() < 1e-12_f64);
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
            liquidation_seconds: 0,
            bar_volume_limit: 1_000.0,
        }
    }
}
