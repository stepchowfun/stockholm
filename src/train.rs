use crate::backtest::{UNRELIABLE_DATA_END_TIME, UNRELIABLE_DATA_START_TIME};
use burn::{
    backend::{Autodiff, NdArray, ndarray::NdArrayDevice},
    data::{
        dataloader::{DataLoader, DataLoaderBuilder, batcher::Batcher},
        dataset::InMemDataset,
    },
    module::AutodiffModule,
    nn::{
        Dropout, DropoutConfig, Linear, LinearConfig, Relu,
        loss::{MseLoss, Reduction::Mean},
    },
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
};
use clap::Args as ClapArgs;
use std::{error::Error, fs, marker::PhantomData, path::PathBuf, sync::Arc};
use time::OffsetDateTime;
use time_tz::{OffsetDateTimeExt, timezones::db::america::NEW_YORK};

// Fix the forecasting horizon and the amount of history supplied to the model.
const INPUTS: usize = 56;
const OUTPUTS: usize = 64;

// These arguments configure a time-series training run.
#[derive(ClapArgs)]
pub struct Args {
    /// CSV files used to train the model.
    #[arg(long, required = true, num_args = 1..)]
    training_paths: Vec<PathBuf>,

    /// CSV files used only to validate the model.
    #[arg(long, required = true, num_args = 1..)]
    validation_paths: Vec<PathBuf>,

    /// Number of examples processed in each optimization step.
    #[arg(long, default_value_t = 64, value_parser = parse_positive_usize)]
    batch_size: usize,

    /// Number of complete passes through the training dataset.
    #[arg(long, default_value_t = 5, value_parser = parse_positive_usize)]
    epochs: usize,

    /// Step size used by the Adam optimizer.
    #[arg(long, default_value_t = 1e-3_f64, value_parser = parse_positive_f64)]
    learning_rate: f64,

    /// Probability of dropping each hidden activation during training.
    #[arg(long, default_value_t = 0.0_f64, value_parser = parse_dropout)]
    dropout: f64,

    /// Seed used for model initialization and training-data shuffling.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Directory where the trained model and its configuration will be saved.
    #[arg(long, default_value = "model")]
    model_directory: PathBuf,
}

// This item stores one normalized input and target window.
#[derive(Clone, Debug)]
struct SeriesItem {
    inputs: Vec<f32>,
    targets: Vec<f32>,
}

// This batch stores the tensors consumed and predicted by the model.
#[derive(Clone, Debug)]
struct SeriesBatch<B: Backend> {
    inputs: Tensor<B, 2>,
    targets: Tensor<B, 2>,
}

// This batcher converts in-memory windows to backend tensors.
#[derive(Clone, Debug)]
struct SeriesBatcher<B: Backend> {
    backend: PhantomData<B>,
}

impl<B: Backend> SeriesBatcher<B> {
    // Construct a batcher for the selected backend.
    fn new() -> Self {
        Self {
            backend: PhantomData,
        }
    }
}

impl<B: Backend> Batcher<B, SeriesItem, SeriesBatch<B>> for SeriesBatcher<B> {
    // Stack each window into a two-dimensional batch.
    fn batch(&self, items: Vec<SeriesItem>, device: &B::Device) -> SeriesBatch<B> {
        // Convert every input window before concatenating along the batch dimension.
        let inputs = items
            .iter()
            .map(|item| Tensor::<B, 1>::from_floats(item.inputs.as_slice(), device).unsqueeze())
            .collect();
        let inputs = Tensor::cat(inputs, 0);

        // Convert every target window using the same layout as the inputs.
        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1>::from_floats(item.targets.as_slice(), device).unsqueeze())
            .collect();
        let targets = Tensor::cat(targets, 0);

        SeriesBatch { inputs, targets }
    }
}

// This model predicts several future returns directly from a fixed input window.
#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    input: Linear<B>,
    output: Linear<B>,
    activation: Relu,
    dropout: Dropout,
}

// This configuration records the model, training, and preprocessing settings.
#[derive(Config, Debug)]
pub struct ModelConfig {
    batch_size: usize,
    epochs: usize,
    learning_rate: f64,
    dropout: f64,
    seed: u64,
    pub return_mean: f32,
    pub return_deviation: f32,
}

impl ModelConfig {
    // Initialize every model layer on the selected device.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            input: LinearConfig::new(INPUTS, 128).init(device),
            output: LinearConfig::new(128, OUTPUTS).init(device),
            activation: Relu::new(),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> Model<B> {
    // Apply the hidden transformation and output projection.
    pub fn forward(&self, inputs: Tensor<B, 2>) -> Tensor<B, 2> {
        let values = self.activation.forward(self.input.forward(inputs));
        let values = self.dropout.forward(values);
        self.output.forward(values)
    }
}

// This structure contains prepared datasets and their normalization parameters.
struct PreparedData {
    training: Vec<SeriesItem>,
    validation: Vec<SeriesItem>,
    mean: f32,
    deviation: f32,
}

// Read the price history and train a forecasting model.
pub fn run(args: &Args) -> Result<(), Box<dyn Error>> {
    // Load the two file groups independently so validation data never enters training.
    let training_series = load_series(&args.training_paths)?;
    let validation_series = load_series(&args.validation_paths)?;
    let data = prepare_data(&training_series, &validation_series, INPUTS, OUTPUTS)?;

    // Use Burn's portable CPU backend for deterministic local training.
    let device = NdArrayDevice::Cpu;
    train::<Autodiff<NdArray>>(&device, args, &data)?;

    Ok(())
}

// Load opening prices while preserving every contiguous segment as an independent series.
fn load_series(paths: &[PathBuf]) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    paths.iter().try_fold(Vec::new(), |mut series, path| {
        // Parse and append every independent segment produced by this file.
        let contents = fs::read_to_string(path)
            .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
        let parsed_series = parse_training_prices(&contents)
            .map_err(|error| format!("failed to parse {}: {error}", path.display()))?;
        series.extend(parsed_series);

        Ok(series)
    })
}

// Train the model, evaluate it chronologically, and save its artifacts.
fn train<B: AutodiffBackend>(
    device: &B::Device,
    args: &Args,
    data: &PreparedData,
) -> Result<(), Box<dyn Error>> {
    // Collect every reproducibility setting into the model's saved configuration.
    let config = ModelConfig::new(
        args.batch_size,
        args.epochs,
        args.learning_rate,
        args.dropout,
        args.seed,
        data.mean,
        data.deviation,
    );

    // Seed Burn and construct the fixed architecture.
    B::seed(device, config.seed);
    let mut model = config.init::<B>(device);
    let mut optimizer = AdamConfig::new().init::<B, Model<B>>();

    // Scan all windows while shuffling only the training order each epoch.
    let training_loader = DataLoaderBuilder::new(SeriesBatcher::<B>::new())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(1)
        .build(InMemDataset::new(data.training.clone()));
    let training_evaluation_loader =
        DataLoaderBuilder::new(SeriesBatcher::<B::InnerBackend>::new())
            .batch_size(config.batch_size)
            .num_workers(1)
            .build(InMemDataset::new(data.training.clone()));
    let validation_loader = DataLoaderBuilder::new(SeriesBatcher::<B::InnerBackend>::new())
        .batch_size(config.batch_size)
        .num_workers(1)
        .build(InMemDataset::new(data.validation.clone()));

    // Establish the validation target by predicting no future price movement.
    let baseline_loss = baseline_loss(&data.validation, data.mean, data.deviation);
    println!(
        "No-change baseline RMSE: {:.2} bps",
        baseline_loss.sqrt() * f64::from(data.deviation) * 10_000.0_f64,
    );

    // Report the initialized model's error before applying any optimizer steps.
    let initial_model = model.valid();
    let initial_training_loss = validation_loss(&initial_model, &training_evaluation_loader);
    let initial_validation_loss = validation_loss(&initial_model, &validation_loader);
    println!(
        "Initial RMSE: train {:.2} bps, validation {:.2} bps",
        initial_training_loss.sqrt() * data.deviation * 10_000.0_f32,
        initial_validation_loss.sqrt() * data.deviation * 10_000.0_f32,
    );

    // Optimize mean-squared error and display validation progress each epoch.
    println!(
        "Training on {} windows and validating on {} windows…",
        data.training.len(),
        data.validation.len(),
    );
    for epoch in 1..=config.epochs {
        train_epoch(
            &mut model,
            &mut optimizer,
            &training_loader,
            config.learning_rate,
        );

        // Evaluate the fully updated model on both splits for comparable epoch metrics.
        let valid_model = model.valid();
        let training_loss = validation_loss(&valid_model, &training_evaluation_loader);
        let validation_loss = validation_loss(&valid_model, &validation_loader);
        println!(
            "Epoch {:>2}/{}: train RMSE {:.2} bps, validation RMSE {:.2} bps",
            epoch,
            config.epochs,
            training_loss.sqrt() * data.deviation * 10_000.0_f32,
            validation_loss.sqrt() * data.deviation * 10_000.0_f32,
        );
    }

    // Report the trained model's final validation error without repeating the baseline.
    let final_loss = validation_loss(&model.valid(), &validation_loader);
    println!(
        "Final validation RMSE: {:.2} bps",
        final_loss.sqrt() * data.deviation * 10_000.0_f32,
    );

    // Save the Burn parameters and their complete reconstruction configuration.
    fs::create_dir_all(&args.model_directory)?;
    model.save_file(args.model_directory.join("model"), &CompactRecorder::new())?;
    config.save(args.model_directory.join("model.json"))?;
    println!(
        "Saved training artifacts to {}.",
        args.model_directory.display(),
    );

    Ok(())
}

// Optimize the model over one complete shuffled pass through the training data.
fn train_epoch<B: AutodiffBackend, O>(
    model: &mut Model<B>,
    optimizer: &mut O,
    loader: &Arc<dyn DataLoader<B, SeriesBatch<B>>>,
    learning_rate: f64,
) where
    O: Optimizer<Model<B>, B>,
{
    // Apply one gradient update for every shuffled batch.
    for batch in loader.iter() {
        let predictions = model.forward(batch.inputs);
        let loss = MseLoss::new().forward(predictions, batch.targets, Mean);

        // Associate gradients with model parameters before applying Adam.
        let gradients = GradientsParams::from_grads(loss.backward(), model);
        *model = optimizer.step(learning_rate, model.clone(), gradients);
    }
}

// Transform and normalize raw prices before windowing every independent series.
fn prepare_data(
    training_prices: &[Vec<f32>],
    validation_prices: &[Vec<f32>],
    inputs: usize,
    outputs: usize,
) -> Result<PreparedData, Box<dyn Error>> {
    // Convert each price series independently so returns never cross a timestamp gap.
    let training_series = training_prices
        .iter()
        .map(|prices| log_returns(prices))
        .collect::<Vec<_>>();
    let validation_series = validation_prices
        .iter()
        .map(|prices| log_returns(prices))
        .collect::<Vec<_>>();

    // Require every series to contain at least one complete model window.
    for (kind, series) in [
        ("training", &training_series),
        ("validation", &validation_series),
    ] {
        if series.is_empty() {
            return Err(format!("at least one {kind} series is required").into());
        }
        for (index, returns) in series.iter().enumerate() {
            if returns.len() < inputs + outputs {
                return Err(format!(
                    "{kind} series {} is too short for {inputs} inputs and {outputs} outputs",
                    index + 1,
                )
                .into());
            }
        }
    }

    // Fit normalization exclusively to returns from the training series.
    let training_count = training_series.iter().map(Vec::len).sum::<usize>();
    let training_sum = training_series.iter().flatten().sum::<f32>();
    let mean = training_sum / usize_to_f32(training_count);
    let variance = training_series
        .iter()
        .flatten()
        .map(|value| (value - mean).powi(2))
        .sum::<f32>()
        / usize_to_f32(training_count);
    let deviation = variance.sqrt();
    if !deviation.is_finite() || deviation <= f32::EPSILON {
        return Err("training returns must have nonzero finite variance".into());
    }

    // Normalize and window each series separately so no example crosses a timestamp gap.
    let prepare = |series: &[Vec<f32>]| {
        let mut items = Vec::new();
        for returns in series {
            let normalized = returns
                .iter()
                .map(|value| (value - mean) / deviation)
                .collect::<Vec<_>>();
            items.extend(windows(&normalized, inputs, outputs));
        }
        items
    };
    let training = prepare(&training_series);
    let validation = prepare(&validation_series);

    Ok(PreparedData {
        training,
        validation,
        mean,
        deviation,
    })
}

// Create every overlapping direct multi-horizon example in chronological order.
fn windows(values: &[f32], inputs: usize, outputs: usize) -> Vec<SeriesItem> {
    values
        .windows(inputs + outputs)
        .map(|window| SeriesItem {
            inputs: window[..inputs].to_vec(),
            targets: window[inputs..].to_vec(),
        })
        .collect()
}

// Measure mean-squared error without constructing an autodiff graph.
fn validation_loss<B: Backend>(
    model: &Model<B>,
    loader: &Arc<dyn DataLoader<B, SeriesBatch<B>>>,
) -> f32 {
    // Accumulate sample-weighted loss across the ordered validation windows.
    let mut total_loss = 0.0_f32;
    let mut total_items = 0_usize;
    for batch in loader.iter() {
        let item_count = batch.targets.dims()[0];
        let predictions = model.forward(batch.inputs);
        let loss = MseLoss::new().forward(predictions, batch.targets, Mean);
        total_loss += loss.into_scalar().elem::<f32>() * usize_to_f32(item_count);
        total_items += item_count;
    }

    total_loss / usize_to_f32(total_items)
}

// Measure the normalized error from predicting zero return at every horizon.
fn baseline_loss(items: &[SeriesItem], mean: f32, deviation: f32) -> f64 {
    // Accumulate normalized errors in double precision across the large target set.
    let prediction = -f64::from(mean) / f64::from(deviation);
    let (squared_error, target_count) = items.iter().flat_map(|item| item.targets.iter()).fold(
        (0.0_f64, 0.0_f64),
        |(squared_error, count), target| {
            let error = f64::from(*target) - prediction;
            (squared_error + error.powi(2), count + 1.0_f64)
        },
    );
    squared_error / target_count
}

// Convert raw prices into successive logarithmic returns.
pub fn log_returns(prices: &[f32]) -> Vec<f32> {
    prices
        .windows(2)
        .map(|pair| (pair[1] / pair[0]).ln())
        .collect()
}

// Convert collection sizes for floating-point averages where minor rounding is acceptable.
#[allow(clippy::cast_precision_loss)]
fn usize_to_f32(value: usize) -> f32 {
    value as f32
}

// Parse the latest contiguous opening-price series after excluding unreliable reports.
pub fn parse_prices(contents: &str) -> Result<Vec<f32>, Box<dyn Error>> {
    let prices = parse_training_prices(contents)?;
    Ok(prices
        .last()
        .cloned()
        .ok_or("the CSV file contains no reliable price data")?)
}

// Parse contiguous training-price series after excluding delayed early-morning trade reports.
fn parse_training_prices(contents: &str) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    // Locate required columns by name so their order remains explicit.
    let mut reader = csv::Reader::from_reader(contents.as_bytes());
    let headers = reader.headers()?;
    let open_index = headers
        .iter()
        .position(|header| header == "open")
        .ok_or("the CSV file must contain an open column")?;
    let timestamp_index = headers
        .iter()
        .position(|header| header == "date")
        .ok_or("the CSV file must contain a date column")?;

    // Split retained prices whenever consecutive source timestamps are not one second apart.
    let mut price_series = Vec::<Vec<f32>>::new();
    let mut previous_timestamp: Option<i64> = None;
    for (index, result) in reader.records().enumerate() {
        let record = result?;
        let line = index + 2;
        let value = record
            .get(timestamp_index)
            .ok_or_else(|| format!("missing timestamp on line {line}"))?;
        let timestamp = value
            .parse::<i64>()
            .map_err(|error| format!("invalid timestamp on line {line}: {error}"))?;
        let datetime = OffsetDateTime::from_unix_timestamp(timestamp)
            .map_err(|error| format!("invalid timestamp on line {line}: {error}"))?;
        let eastern_time = datetime.to_timezone(NEW_YORK).time();
        if eastern_time >= UNRELIABLE_DATA_START_TIME && eastern_time < UNRELIABLE_DATA_END_TIME {
            continue;
        }

        // Attach line numbers to malformed values so input problems are actionable.
        let value = record
            .get(open_index)
            .ok_or_else(|| format!("missing opening price on line {line}"))?;
        let price: f32 = value
            .parse()
            .map_err(|error| format!("invalid opening price on line {line}: {error}"))?;
        if !price.is_finite() || price <= 0.0_f32 {
            return Err(format!("opening price on line {line} must be finite and positive").into());
        }
        if previous_timestamp.is_none_or(|previous| previous.checked_add(1) != Some(timestamp)) {
            price_series.push(Vec::new());
        }
        price_series
            .last_mut()
            .ok_or("the price series must contain a current chunk")?
            .push(price);
        previous_timestamp = Some(timestamp);
    }

    Ok(price_series)
}

// Parse a strictly positive model dimension from the command line.
fn parse_positive_usize(value: &str) -> Result<usize, String> {
    // Reject zero explicitly because empty tensor dimensions are not meaningful here.
    let parsed = value.parse::<usize>().map_err(|error| error.to_string())?;
    if parsed == 0 {
        return Err("value must be greater than zero".to_string());
    }

    Ok(parsed)
}

// Parse a finite positive floating-point parameter from the command line.
fn parse_positive_f64(value: &str) -> Result<f64, String> {
    // Reject values that would make optimizer updates invalid or ineffective.
    let parsed = value.parse::<f64>().map_err(|error| error.to_string())?;
    if !parsed.is_finite() || parsed <= 0.0_f64 {
        return Err("value must be finite and greater than zero".to_string());
    }

    Ok(parsed)
}

// Parse a finite dropout probability that leaves some activations enabled.
fn parse_dropout(value: &str) -> Result<f64, String> {
    // Permit disabled dropout while preventing Burn's scaling from dividing by zero.
    let parsed = value.parse::<f64>().map_err(|error| error.to_string())?;
    if !parsed.is_finite() || !(0.0_f64..1.0_f64).contains(&parsed) {
        return Err("value must be finite, nonnegative, and less than one".to_string());
    }

    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::{
        SeriesItem, baseline_loss, parse_dropout, parse_prices, parse_training_prices, prepare_data,
    };
    use crate::{Cli, Subcommand};
    use clap::Parser;
    use std::path::PathBuf;

    #[test]
    fn parse_train_subcommand() {
        // Confirm the training mode supplies useful defaults for optional settings.
        let cli = Cli::try_parse_from([
            "stockholm",
            "train",
            "--training-paths",
            "monday.csv",
            "tuesday.csv",
            "--validation-paths",
            "wednesday.csv",
        ])
        .unwrap();

        let Some(Subcommand::Train(args)) = cli.command else {
            panic!("expected train subcommand");
        };
        assert_eq!(
            args.training_paths,
            vec![PathBuf::from("monday.csv"), PathBuf::from("tuesday.csv")],
        );
        assert_eq!(args.validation_paths, vec![PathBuf::from("wednesday.csv")]);
        assert_eq!(args.batch_size, 64);
        assert_eq!(args.epochs, 5);
        assert!((args.learning_rate - 1e-3_f64).abs() < f64::EPSILON);
        assert!(args.dropout.abs() < f64::EPSILON);
        assert_eq!(args.seed, 42);
        assert_eq!(args.model_directory, PathBuf::from("model"));
    }

    #[test]
    fn validate_dropout_probability() {
        // Accept disabled and partial dropout while rejecting invalid scaling probabilities.
        assert!(parse_dropout("0").unwrap().abs() < f64::EPSILON);
        assert!((parse_dropout("0.5").unwrap() - 0.5_f64).abs() < f64::EPSILON);
        for value in ["-0.1", "1", "NaN", "infinity"] {
            assert!(parse_dropout(value).is_err());
        }
    }

    #[test]
    fn parse_price_lines() {
        // Confirm the header and unused historical columns do not alter opening prices.
        let prices = parse_prices(concat!(
            "date,open,high,low,close,volume,wap,count\n",
            "1,100,900,1,2,3,4,5\n",
            "2,101.5,800,1,2,3,4,5\n",
            "3,99,700,1,2,3,4,5\n",
        ))
        .unwrap();

        assert_eq!(prices, vec![100.0_f32, 101.5_f32, 99.0_f32]);
    }

    #[test]
    fn discard_unreliable_training_prices_across_eastern_time_offsets() {
        // Filter the shared window, split surrounding prices, and select the latest for inference.
        for start in [1_784_016_000_i64, 1_767_949_200_i64] {
            let contents = format!(
                concat!(
                    "date,open\n",
                    "{},99\n",
                    "{},invalid\n",
                    "{},invalid\n",
                    "{},100\n",
                ),
                start - 1,
                start,
                start + 899,
                start + 900,
            );

            assert_eq!(
                parse_training_prices(&contents).unwrap(),
                vec![vec![99.0_f32], vec![100.0_f32]],
            );
            assert_eq!(parse_prices(&contents).unwrap(), vec![100.0_f32]);
        }
    }

    #[test]
    fn split_prices_at_timestamp_gaps() {
        // Keep one-second observations together while separating gaps and reversed timestamps.
        let prices = parse_training_prices(concat!(
            "date,open\n",
            "1784030400,100\n",
            "1784030401,101\n",
            "1784030403,102\n",
            "1784030404,103\n",
            "1784030402,104\n",
        ))
        .unwrap();

        assert_eq!(
            prices,
            vec![
                vec![100.0_f32, 101.0_f32],
                vec![102.0_f32, 103.0_f32],
                vec![104.0_f32],
            ],
        );
    }

    #[test]
    fn reject_nonpositive_prices() {
        // Confirm price parsing rejects values outside the logarithmic return domain.
        let error = parse_prices("date,open\n1,100\n2,0\n").unwrap_err();

        assert!(error.to_string().contains("finite and positive"));
    }

    #[test]
    fn create_separate_training_and_validation_windows() {
        // Confirm raw price series contribute only to their selected dataset.
        let prices = (1_u16..=101).map(f32::from).collect::<Vec<_>>();
        let series = std::slice::from_ref(&prices);
        let data = prepare_data(series, series, 4, 2).unwrap();

        assert_eq!(data.training.len(), 95);
        assert_eq!(data.validation.len(), 95);
        assert!(baseline_loss(&data.validation, data.mean, data.deviation).is_finite());
    }

    #[test]
    fn calculate_baseline_loss_in_double_precision() {
        // Preserve errors too small to affect a much larger term in single precision.
        let items = [SeriesItem {
            inputs: Vec::new(),
            targets: vec![10_000.0_f32, 1.0_f32],
        }];

        assert!((baseline_loss(&items, 0.0_f32, 1.0_f32) - 50_000_000.5_f64).abs() < f64::EPSILON);
    }

    #[test]
    fn keep_series_windows_separate() {
        // Confirm combining series does not add windows across their boundary.
        let prices = (1_u16..=101).map(f32::from).collect::<Vec<_>>();
        let series = [prices.clone(), prices];
        let data = prepare_data(&series, &series, 4, 2).unwrap();

        assert_eq!(data.training.len(), 190);
        assert_eq!(data.validation.len(), 190);
    }
}
