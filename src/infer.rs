use crate::train::{ModelConfig, log_returns, parse_prices};
use burn::{
    backend::{NdArray, ndarray::NdArrayDevice},
    module::Module,
    prelude::*,
    record::CompactRecorder,
};
use clap::Args as ClapArgs;
use std::{error::Error, fs, io, path::PathBuf};

// These arguments configure a single model inference.
#[derive(ClapArgs)]
pub struct Args {
    /// Directory containing the trained model and its configuration.
    #[arg(long, default_value = "model")]
    model_directory: PathBuf,

    /// CSV file whose latest contiguous series contains exactly one input window.
    #[arg(long, default_value = "data/inference-sample.csv")]
    input_path: PathBuf,
}

// Load a trained model and print one forecast as CSV.
pub fn run(args: &Args) -> Result<(), Box<dyn Error>> {
    // Load the architecture and normalization values saved by the train subcommand.
    let config = ModelConfig::load(args.model_directory.join("model.json"))?;
    if !config.return_mean.is_finite()
        || !config.return_deviation.is_finite()
        || config.return_deviation <= f32::EPSILON
    {
        return Err("model configuration contains invalid normalization values".into());
    }

    // Parse the latest raw price series and transform it exactly like a training window.
    let contents = fs::read_to_string(&args.input_path)
        .map_err(|error| format!("failed to read {}: {error}", args.input_path.display()))?;
    let prices = parse_prices(&contents)
        .map_err(|error| format!("failed to parse {}: {error}", args.input_path.display()))?;
    let normalized = log_returns(&prices)
        .into_iter()
        .map(|value| (value - config.return_mean) / config.return_deviation)
        .collect::<Vec<_>>();

    // Restore the trained parameters and evaluate the single normalized input window.
    let device = NdArrayDevice::Cpu;
    let model = config.init::<NdArray>(&device).load_file(
        args.model_directory.join("model"),
        &CompactRecorder::new(),
        &device,
    )?;
    let inputs = Tensor::<NdArray, 1>::from_floats(normalized.as_slice(), &device).unsqueeze();
    let normalized_predictions = model.forward(inputs).into_data().to_vec::<f32>()?;
    let predicted_returns = normalized_predictions
        .into_iter()
        .map(|value| value * config.return_deviation + config.return_mean);
    let predicted_prices = forecast_prices(*prices.last().unwrap(), predicted_returns);

    // Emit the predicted opening prices without mixing diagnostics into standard output.
    write_predictions(io::stdout().lock(), &predicted_prices)?;

    Ok(())
}

// Convert predicted log returns into successive opening-price forecasts.
fn forecast_prices(
    initial_price: f32,
    predicted_returns: impl IntoIterator<Item = f32>,
) -> Vec<f32> {
    // Apply each return to the preceding observed or predicted price.
    let mut price = initial_price;
    predicted_returns
        .into_iter()
        .map(|predicted_return| {
            price *= predicted_return.exp();
            price
        })
        .collect()
}

// Serialize predicted prices as a one-column CSV stream.
fn write_predictions(writer: impl io::Write, prices: &[f32]) -> Result<(), Box<dyn Error>> {
    // Write a header and one chronological forecast per record.
    let mut writer = csv::Writer::from_writer(writer);
    writer.write_record(["open"])?;
    for price in prices {
        writer.write_record([price.to_string()])?;
    }
    writer.flush()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{forecast_prices, write_predictions};
    use crate::{Cli, Subcommand};
    use clap::Parser;
    use std::path::PathBuf;

    #[test]
    fn parse_infer_subcommand() {
        // Confirm inference supplies the conventional artifact and input paths by default.
        let cli = Cli::try_parse_from(["stockholm", "infer"]).unwrap();

        let Some(Subcommand::Infer(args)) = cli.command else {
            panic!("expected infer subcommand");
        };
        assert_eq!(args.model_directory, PathBuf::from("model"));
        assert_eq!(args.input_path, PathBuf::from("data/inference-sample.csv"));
    }

    #[test]
    fn reconstruct_and_write_predictions() {
        // Confirm predicted log returns become chronological price records.
        let prices = forecast_prices(100.0, [0.0, 2.0_f32.ln()]);
        let mut output = Vec::new();
        write_predictions(&mut output, &prices).unwrap();

        assert_eq!(String::from_utf8(output).unwrap(), "open\n100\n200\n");
    }
}
