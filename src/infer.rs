use crate::train::{
    BATCH_SIZE, INPUTS, ModelConfig, OUTCOME_COUNT, log_returns, parse_price_series,
};
use burn::{
    backend::{Flex, flex::FlexDevice},
    module::Module,
    prelude::*,
    record::CompactRecorder,
    tensor::activation::softmax,
};
use clap::Args as ClapArgs;
use std::{error::Error, fs, path::PathBuf};

// These arguments configure inference over a historical price series.
#[derive(ClapArgs)]
pub struct Args {
    /// Directory containing the trained model and its configuration.
    #[arg(long, default_value = "model")]
    model_directory: PathBuf,

    /// CSV file whose latest contiguous series contains at least one input window.
    #[arg(long, default_value = "data/validation/SOXL-2026-07-22.csv")]
    input_path: PathBuf,

    /// CSV file where timestamped prediction probabilities will be saved.
    #[arg(long, default_value = "data/inference/inference-output.csv")]
    output_path: PathBuf,
}

// Load a trained model and save a prediction for every complete input window.
pub fn run(args: &Args) -> Result<(), Box<dyn Error>> {
    // Load the architecture and normalization values saved by the train subcommand.
    let config = ModelConfig::load(args.model_directory.join("model.json"))?;
    if !config.return_mean.is_finite()
        || !config.return_deviation.is_finite()
        || config.return_deviation <= f32::EPSILON
    {
        return Err("model configuration contains invalid normalization values".into());
    }

    // Parse the latest reliable series with each price attached to its source timestamp.
    let contents = fs::read_to_string(&args.input_path)
        .map_err(|error| format!("failed to read {}: {error}", args.input_path.display()))?;
    let series = parse_price_series(&contents)
        .map_err(|error| format!("failed to parse {}: {error}", args.input_path.display()))?;
    let series = series
        .last()
        .ok_or("the CSV file contains no reliable market-hours price data")?;
    let prices = series.iter().map(|price| price.price).collect::<Vec<_>>();
    if prices.len() <= INPUTS {
        return Err(format!("inference requires at least {} prices", INPUTS + 1).into());
    }

    // Restore the trained parameters and prepare the output destination.
    let device = FlexDevice;
    let model = config.init::<Flex>(&device).load_file(
        args.model_directory.join("model"),
        &CompactRecorder::new(),
        &device,
    )?;
    if let Some(parent) = args.output_path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let mut writer = csv::Writer::from_path(&args.output_path)?;
    writer.write_record([
        "timestamp",
        "lower_probability",
        "upper_probability",
        "neither_probability",
    ])?;

    // Normalize and evaluate overlapping windows in bounded batches.
    let window_count = prices.len() - INPUTS;
    for batch_start in (0..window_count).step_by(BATCH_SIZE) {
        let batch_end = (batch_start + BATCH_SIZE).min(window_count);
        let mut normalized = Vec::with_capacity((batch_end - batch_start) * INPUTS);
        for index in batch_start..batch_end {
            normalized.extend(
                log_returns(&prices[index..=index + INPUTS])
                    .into_iter()
                    .map(|value| (value - config.return_mean) / config.return_deviation),
            );
        }

        // Pair each outcome distribution with the timestamp ending its input window.
        let inputs = Tensor::<Flex, 1>::from_floats(normalized.as_slice(), &device)
            .reshape([batch_end - batch_start, INPUTS]);
        let probabilities = softmax(model.forward(inputs), 1)
            .into_data()
            .to_vec::<f32>()?;
        for (offset, outcomes) in probabilities
            .as_chunks::<OUTCOME_COUNT>()
            .0
            .iter()
            .enumerate()
        {
            let timestamp = series[batch_start + offset + INPUTS].timestamp;
            writer.write_record([
                timestamp.to_string(),
                outcomes[0].to_string(),
                outcomes[1].to_string(),
                outcomes[2].to_string(),
            ])?;
        }
    }
    writer.flush()?;

    // Report the generated artifact without mixing it into the CSV itself.
    println!("Saved predictions to {}.", args.output_path.display());

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{Cli, Subcommand};
    use clap::Parser;
    use std::path::PathBuf;

    #[test]
    fn parse_infer_subcommand() {
        // Confirm inference supplies conventional model, input, and output paths by default.
        let cli = Cli::try_parse_from(["stockholm", "infer"]).unwrap();

        let Some(Subcommand::Infer(args)) = cli.command else {
            panic!("expected infer subcommand");
        };
        assert_eq!(args.model_directory, PathBuf::from("model"));
        assert_eq!(
            args.input_path,
            PathBuf::from("data/validation/SOXL-2026-07-22.csv"),
        );
        assert_eq!(
            args.output_path,
            PathBuf::from("data/inference/inference-output.csv"),
        );
    }
}
