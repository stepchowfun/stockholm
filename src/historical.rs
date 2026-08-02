use clap::Args as ClapArgs;
use ibapi::{
    Client,
    contracts::Contract,
    market_data::historical::{self, BarSize},
};
use std::error::Error;
use time::{self, Date, OffsetDateTime, Time, format_description::well_known::Iso8601};

// These constants configure historical data requests.
const HISTORICAL_CHUNK_SECONDS: i64 = 1_800;
const SYMBOL: &str = "SOXL";

// These arguments configure a historical data request.
#[derive(ClapArgs)]
pub(crate) struct Args {
    /// Amount of data preceding the ending date.
    #[arg(long, default_value = "1d", value_parser = parse_historical_duration)]
    duration: HistoricalDuration,

    /// Ending date in YYYY-MM-DD format.
    #[arg(long, default_value = "today", value_parser = parse_date)]
    date: Date,

    /// Symbol whose historical data should be fetched.
    #[arg(long, default_value = SYMBOL)]
    symbol: String,
}

// This type represents a historical duration as an exact number of seconds.
#[derive(Clone, Copy, Debug, PartialEq)]
struct HistoricalDuration {
    seconds: i64,
    ib_duration: historical::Duration,
}

// Connect to Interactive Brokers and fetch the requested historical data.
pub(crate) async fn run(address: &str, client_id: i32, args: &Args) -> Result<(), Box<dyn Error>> {
    // Connect once because historical requests are not retried indefinitely.
    let client = Client::connect(address, client_id).await?;
    fetch_historical_data(&client, args).await
}

// Fetch historical one-second bars and print them as CSV rows.
async fn fetch_historical_data(client: &Client, args: &Args) -> Result<(), Box<dyn Error>> {
    // Find the regular trading sessions within the requested calendar range.
    let end = args.date.with_time(Time::MAX).assume_utc();
    let start = end - time::Duration::seconds(args.duration.seconds);
    let contract = Contract::stock(&args.symbol).build();
    let schedule = client
        .historical_schedules(&contract, args.duration.ib_duration)
        .ending(end)
        .fetch()
        .await?;

    // Print each completed session window while preserving chronological order.
    println!("date,open,high,low,close,volume,wap,count");
    for session in schedule.sessions {
        // Intersect each session with the requested calendar range.
        let session_start = session.start.max(start);
        let session_end = session.end.min(end);
        let session_seconds = (session_end - session_start).whole_seconds();
        if session_seconds > 0 {
            let chunk_count = divide_rounding_up(session_seconds, HISTORICAL_CHUNK_SECONDS);
            let chunk_seconds = divide_rounding_up(session_seconds, chunk_count);
            let mut chunk_start = session_start;

            // Divide the session into windows supported for one-second bars.
            while chunk_start < session_end {
                let chunk_end =
                    (chunk_start + time::Duration::seconds(chunk_seconds)).min(session_end);
                let request_seconds = i32::try_from((chunk_end - chunk_start).whole_seconds())?;
                let historical_data = client
                    .historical_data(&contract, BarSize::Sec)
                    .duration(historical::Duration::seconds(request_seconds))
                    .ending(chunk_end)
                    .fetch()
                    .await?;

                // Print only bars inside this window because IB may clamp requests.
                for bar in historical_data
                    .bars
                    .into_iter()
                    .filter(|bar| bar.date > chunk_start.into() && bar.date <= chunk_end.into())
                {
                    println!(
                        "{},{},{},{},{},{},{},{}",
                        bar.date,
                        bar.open,
                        bar.high,
                        bar.low,
                        bar.close,
                        bar.volume,
                        bar.wap,
                        bar.count,
                    );
                }
                chunk_start = chunk_end;
            }
        }
    }

    Ok(())
}

// Divide positive integers while rounding any partial quotient upward.
fn divide_rounding_up(dividend: i64, divisor: i64) -> i64 {
    dividend / divisor + i64::from(dividend % divisor != 0)
}

// Parse compact historical durations such as `1d` into IB's duration type.
fn parse_historical_duration(value: &str) -> Result<HistoricalDuration, String> {
    // Separate the positive numeric quantity from its one-letter unit.
    if !value.is_ascii() {
        return Err("duration must contain only ASCII characters".to_owned());
    }
    let split_at = value
        .len()
        .checked_sub(1)
        .ok_or("duration cannot be empty")?;
    let (quantity, unit) = value.split_at(split_at);
    let quantity = quantity.parse::<i64>().map_err(|error| error.to_string())?;
    if quantity <= 0 {
        return Err("duration must be positive".to_owned());
    }

    // Convert every supported unit into an exact duration for chunking.
    let (multiplier, ib_duration): (i64, fn(i32) -> historical::Duration) =
        match unit.to_ascii_lowercase().as_str() {
            "s" => (1, historical::Duration::seconds),
            "d" => (24 * 60 * 60, historical::Duration::days),
            "w" => (7 * 24 * 60 * 60, historical::Duration::weeks),
            "m" => (30 * 24 * 60 * 60, historical::Duration::months),
            "y" => (365 * 24 * 60 * 60, historical::Duration::years),
            _ => return Err("duration unit must be s, d, w, m, or y".to_owned()),
        };
    let seconds = quantity
        .checked_mul(multiplier)
        .ok_or("duration is too large")?;

    let quantity = i32::try_from(quantity).map_err(|_| "duration quantity is too large")?;

    Ok(HistoricalDuration {
        seconds,
        ib_duration: ib_duration(quantity),
    })
}

// Parse an ISO date or resolve the convenient `today` default locally.
fn parse_date(value: &str) -> Result<Date, String> {
    // Resolve today's date using the local offset when it is available.
    if value.eq_ignore_ascii_case("today") {
        return Ok(OffsetDateTime::now_local()
            .unwrap_or_else(|_| OffsetDateTime::now_utc())
            .date());
    }

    Date::parse(value, &Iso8601::DATE).map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::Args;
    use clap::Parser;

    // This parser exposes the historical arguments for focused tests.
    #[derive(Parser)]
    struct TestCli {
        #[command(flatten)]
        args: Args,
    }

    #[test]
    fn parse_defaults() {
        // Confirm the historical subcommand uses the documented defaults.
        let cli = TestCli::try_parse_from(["historical"]).unwrap();

        assert_eq!(cli.args.duration.seconds, 86_400);
        assert_eq!(cli.args.symbol, "SOXL");
    }
}
