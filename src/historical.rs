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
const HISTORICAL_CHUNK_SECONDS: i64 = 1_800;

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
}

// Connect to Interactive Brokers and fetch the requested historical data.
pub async fn run(address: &str, client_id: i32, args: &Args) -> Result<(), Box<dyn Error>> {
    // Connect once because historical requests are not retried indefinitely.
    let client = Client::connect(address, client_id).await?;
    fetch_historical_data(&client, args).await
}

// Fetch historical one-second bars and print them as CSV rows.
async fn fetch_historical_data(client: &Client, args: &Args) -> Result<(), Box<dyn Error>> {
    // Reject empty and reversed ranges before submitting requests.
    if args.end <= args.start {
        return Err("end datetime must be after start datetime".into());
    }

    // Request every chunk in the range so extended-hours data is not skipped.
    let contract = Contract::stock(&args.symbol).build();
    println!("date,open,high,low,close,volume,wap,count");
    let mut chunk_start = args.start;

    // Divide the range into windows supported for one-second bars.
    while chunk_start < args.end {
        let chunk_end =
            (chunk_start + time::Duration::seconds(HISTORICAL_CHUNK_SECONDS)).min(args.end);
        let historical_data = client
            .historical_data(&contract, BarSize::Sec)
            .trading_hours(TradingHours::Extended)
            .between(chunk_start - time::Duration::SECOND, chunk_end)
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

// Parse an ISO 8601 datetime with an explicit UTC offset.
fn parse_datetime(value: &str) -> Result<OffsetDateTime, String> {
    OffsetDateTime::parse(value, &Iso8601::DEFAULT).map_err(|error| error.to_string())
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
    }
}
