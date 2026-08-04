use crate::DEFAULT_SYMBOL;
use clap::Args as ClapArgs;
use ibapi::{
    Client,
    contracts::Contract,
    market_data::{TradingHours, historical::BarSize},
};
use std::error::Error;
use time::{self, OffsetDateTime, format_description::well_known::Iso8601};

// These constants configure historical data requests.
const HISTORICAL_MINIMUM_CHUNK_SECONDS: i64 = 1_800;
const HISTORICAL_MINIMUM_SECONDS: i64 = 60;

// These arguments configure a historical data request.
#[derive(ClapArgs)]
pub struct Args {
    /// Beginning of the requested range as an ISO 8601 datetime.
    #[arg(long, value_parser = parse_datetime)]
    start: OffsetDateTime,

    /// End of the requested range as an ISO 8601 datetime.
    #[arg(long, value_parser = parse_datetime)]
    end: OffsetDateTime,

    /// Symbol whose historical data should be fetched.
    #[arg(long, default_value = DEFAULT_SYMBOL)]
    symbol: String,

    /// Interval represented by each historical bar.
    ///
    /// Accepted values are SEC, SEC5, SEC10, SEC15, SEC30, MIN, MIN2, MIN3, MIN4,
    /// MIN5, MIN10, MIN15, MIN20, MIN30, HOUR, HOUR2, HOUR3, HOUR4, HOUR8, DAY,
    /// WEEK, and MONTH.
    #[arg(long, default_value = "SEC")]
    interval: BarSize,
}

// Connect to Interactive Brokers and fetch the requested historical data.
pub async fn run(address: &str, client_id: i32, args: &Args) -> Result<(), Box<dyn Error>> {
    // Connect once because historical requests are not retried indefinitely.
    let client = Client::connect(address, client_id).await?;
    fetch_historical_data(&client, args).await
}

// Fetch historical bars and print them as CSV rows.
async fn fetch_historical_data(client: &Client, args: &Args) -> Result<(), Box<dyn Error>> {
    // Reject empty and reversed ranges before submitting requests.
    if args.end <= args.start {
        return Err("end datetime must be after start datetime".into());
    }

    // Request every chunk in the range so extended-hours data is not skipped.
    let contract = Contract::stock(&args.symbol).build();
    println!("date,open,high,low,close,volume,wap,count");
    let mut chunk_start = args.start;
    let chunk_seconds = historical_chunk_seconds(args.interval);

    // Divide the range into windows supported for the selected bar size.
    while chunk_start < args.end {
        let chunk_end = (chunk_start + time::Duration::seconds(chunk_seconds)).min(args.end);
        let request_start = historical_request_start(chunk_start, chunk_end, args.interval);
        let historical_data = client
            .historical_data(&contract, args.interval)
            .trading_hours(TradingHours::Extended)
            .between(request_start, chunk_end)
            .fetch()
            .await?;

        // Print bar-start timestamps inside this window.
        for bar in historical_data
            .bars
            .into_iter()
            .filter(|bar| bar.date >= chunk_start.into() && bar.date < chunk_end.into())
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

    Ok(())
}

// Include the preceding second while requesting enough time for the selected bar size.
fn historical_request_start(
    chunk_start: OffsetDateTime,
    chunk_end: OffsetDateTime,
    interval: BarSize,
) -> OffsetDateTime {
    // Pad short final chunks to IBKR's minimum duration or one complete bar.
    let minimum_seconds = HISTORICAL_MINIMUM_SECONDS.max(bar_size_seconds(interval));
    (chunk_start - time::Duration::SECOND).min(chunk_end - time::Duration::seconds(minimum_seconds))
}

// Use chunks that can contain at least one bar without shrinking one-second requests.
fn historical_chunk_seconds(interval: BarSize) -> i64 {
    HISTORICAL_MINIMUM_CHUNK_SECONDS.max(bar_size_seconds(interval))
}

// Convert each IBKR bar size to a conservative duration in seconds.
fn bar_size_seconds(interval: BarSize) -> i64 {
    match interval {
        BarSize::Sec => 1,
        BarSize::Sec5 => 5,
        BarSize::Sec10 => 10,
        BarSize::Sec15 => 15,
        BarSize::Sec30 => 30,
        BarSize::Min => 60,
        BarSize::Min2 => 120,
        BarSize::Min3 => 180,
        BarSize::Min4 => 240,
        BarSize::Min5 => 300,
        BarSize::Min10 => 600,
        BarSize::Min15 => 900,
        BarSize::Min20 => 1_200,
        BarSize::Min30 => 1_800,
        BarSize::Hour => 3_600,
        BarSize::Hour2 => 7_200,
        BarSize::Hour3 => 10_800,
        BarSize::Hour4 => 14_400,
        BarSize::Hour8 => 28_800,
        BarSize::Day => 86_400,
        BarSize::Week => 604_800,
        BarSize::Month => 2_678_400,
    }
}

// Parse an ISO 8601 datetime with an explicit UTC offset.
fn parse_datetime(value: &str) -> Result<OffsetDateTime, String> {
    OffsetDateTime::parse(value, &Iso8601::DEFAULT).map_err(|error| error.to_string())
}

#[cfg(test)]
mod tests {
    use super::{Args, historical_chunk_seconds, historical_request_start};
    use clap::Parser;
    use ibapi::market_data::historical::BarSize;
    use time::OffsetDateTime;

    // This parser exposes the historical arguments for focused tests.
    #[derive(Parser)]
    struct TestCli {
        #[command(flatten)]
        args: Args,
    }

    #[test]
    fn parse_range() {
        // Confirm the historical subcommand accepts explicit datetime bounds.
        let cli = TestCli::try_parse_from([
            "historical",
            "--start",
            "2026-07-31T09:30:00-04:00",
            "--end",
            "2026-07-31T16:00:00-04:00",
        ])
        .unwrap();

        assert_eq!((cli.args.end - cli.args.start).whole_seconds(), 23_400);
        assert_eq!(cli.args.symbol, "SOXL");
        assert_eq!(cli.args.interval, BarSize::Sec);
    }

    #[test]
    fn parse_interval() {
        // Confirm the historical subcommand accepts an explicit bar interval.
        let cli = TestCli::try_parse_from([
            "historical",
            "--start",
            "2026-07-31T09:30:00-04:00",
            "--end",
            "2026-07-31T16:00:00-04:00",
            "--interval",
            "MIN5",
        ])
        .unwrap();

        assert_eq!(cli.args.interval, BarSize::Min5);
    }

    #[test]
    fn pad_short_request() {
        // Confirm a short output window is requested using IBKR's minimum duration.
        let chunk_start = OffsetDateTime::from_unix_timestamp(1_785_527_995).unwrap();
        let chunk_end = OffsetDateTime::from_unix_timestamp(1_785_528_005).unwrap();
        let request_start = historical_request_start(chunk_start, chunk_end, BarSize::Sec);

        assert_eq!((chunk_end - request_start).whole_seconds(), 60);
    }

    #[test]
    fn overlap_full_request() {
        // Confirm a full output window retains the preceding-second overlap.
        let chunk_start = OffsetDateTime::from_unix_timestamp(1_785_504_600).unwrap();
        let chunk_end = OffsetDateTime::from_unix_timestamp(1_785_506_400).unwrap();
        let request_start = historical_request_start(chunk_start, chunk_end, BarSize::Sec);

        assert_eq!((chunk_start - request_start).whole_seconds(), 1);
    }

    #[test]
    fn expand_chunks_for_large_intervals() {
        // Confirm each request window can contain at least one selected bar.
        assert_eq!(historical_chunk_seconds(BarSize::Hour), 3_600);
        assert_eq!(historical_chunk_seconds(BarSize::Day), 86_400);
    }
}
