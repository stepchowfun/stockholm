use clap::{Args as ClapArgs, ValueEnum};
use std::{error::Error, fs, path::PathBuf};

// These strategies can be evaluated by a backtest.
#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub enum Strategy {
    BuyAndHold,
}

// These arguments configure a backtest run.
#[derive(ClapArgs)]
pub struct Args {
    /// Trading strategy to evaluate.
    #[arg(long, value_enum)]
    strategy: Strategy,

    /// CSV files containing historical market data.
    #[arg(long, required = true, num_args = 1..)]
    data_paths: Vec<PathBuf>,
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
            let change = buy_and_hold(&files)?;
            println!("{change}");
        }
    }

    Ok(())
}

// Calculate the absolute price change produced by buying first and selling last.
fn buy_and_hold(files: &[(PathBuf, String)]) -> Result<f64, Box<dyn Error>> {
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
    Ok(last_close.unwrap() - first_open)
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

#[cfg(test)]
mod tests {
    use super::{Strategy, buy_and_hold};
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

        assert!((buy_and_hold(&files).unwrap() - 130.0).abs() < f64::EPSILON);
    }
}
