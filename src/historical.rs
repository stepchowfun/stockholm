use crate::DEFAULT_SYMBOL;
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
    // Validate and convert the requested range for the Interactive Brokers API.
    let duration = args.end - args.start;
    if !duration.is_positive() {
        return Err("end datetime must be after start datetime".into());
    }
    let duration_seconds =
        i32::try_from(duration.whole_seconds() + i64::from(duration.subsec_nanoseconds() != 0_i32))
            .map_err(|_| "historical range is too large")?;

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
        if session_end > session_start {
            let mut chunk_start = session_start;

            // Divide the session into windows supported for one-second bars.
            while chunk_start < session_end {
                let chunk_end = (chunk_start + time::Duration::seconds(HISTORICAL_CHUNK_SECONDS))
                    .min(session_end);
                let historical_data = client
                    .historical_data(&contract, BarSize::Sec)
                    .between(chunk_start, chunk_end)
                    .fetch()
                    .await?;

                // Print bar-start timestamps inside this window because IB may clamp requests.
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
        }
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
