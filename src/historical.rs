use clap::Args as ClapArgs;
use ibapi::{
    Client,
    contracts::Contract,
    market_data::historical::{self, BarSize},
};
use std::error::Error;
use time::{self, OffsetDateTime, format_description::well_known::Iso8601};

// These constants configure historical data requests.
const HISTORICAL_CHUNK_SECONDS: i64 = 1_800;
const SYMBOL: &str = "SOXL";

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
    #[arg(long, default_value = SYMBOL)]
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
    // Validate and convert the requested range for the Interactive Brokers API.
    let duration_seconds = (args.end - args.start).whole_seconds();
    if duration_seconds <= 0 {
        return Err("end datetime must be after start datetime".into());
    }
    let duration_seconds =
        i32::try_from(duration_seconds).map_err(|_| "historical range is too large")?;

    // Find the regular trading sessions within the requested datetime range.
    let contract = Contract::stock(&args.symbol).build();
    let schedule = client
        .historical_schedules(&contract, historical::Duration::seconds(duration_seconds))
        .ending(args.end)
        .fetch()
        .await?;

    // Print each completed session window while preserving chronological order.
    println!("date,open,high,low,close,volume,wap,count");
    for session in schedule.sessions {
        // Intersect each session with the requested calendar range.
        let session_start = session.start.max(args.start);
        let session_end = session.end.min(args.end);
        let session_seconds = (session_end - session_start).whole_seconds();
        if session_seconds > 0 {
            let chunk_count = divide_rounding_up(session_seconds, HISTORICAL_CHUNK_SECONDS);
            let chunk_seconds = divide_rounding_up(session_seconds, chunk_count);
            let mut chunk_start = session_start;

            // Divide the session into windows supported for one-second bars.
            while chunk_start < session_end {
                let chunk_end =
                    (chunk_start + time::Duration::seconds(chunk_seconds)).min(session_end);
                let historical_data = client
                    .historical_data(&contract, BarSize::Sec)
                    .between(chunk_start, chunk_end)
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
            "2026-07-31T13:30:00Z",
            "--end",
            "2026-07-31T20:00:00Z",
        ])
        .unwrap();

        assert_eq!((cli.args.end - cli.args.start).whole_seconds(), 23_400);
        assert_eq!(cli.args.symbol, "SOXL");
    }
}
