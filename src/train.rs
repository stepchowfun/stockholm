use crate::backtest::{UNRELIABLE_DATA_END_TIME, UNRELIABLE_DATA_START_TIME};
use burn::{
    backend::{Autodiff, Flex, flex::FlexDevice},
    data::{
        dataloader::{DataLoader, DataLoaderBuilder, batcher::Batcher},
        dataset::InMemDataset,
    },
    module::AutodiffModule,
    nn::{
        Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig1d, Relu,
        conv::{Conv1d, Conv1dConfig},
        loss::BinaryCrossEntropyLossConfig,
        pool::{AvgPool1d, AvgPool1dConfig},
    },
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
};
use clap::Args as ClapArgs;
use std::{error::Error, fs, marker::PhantomData, path::PathBuf, sync::Arc};
use time::{OffsetDateTime, Time};
use time_tz::{OffsetDateTimeExt, timezones::db::america::NEW_YORK};

// Fix the price history supplied to the model and the future crossing horizon.
pub const INPUTS: usize = 128;
const OUTPUTS: usize = 128;

// Bound memory use and keep optimizer updates frequent enough for this overlapping dataset.
pub const BATCH_SIZE: usize = 64;

// Require a future price to exceed the last observed price by this relative amount.
const TARGET_INCREASE: f32 = 0.002_f32;

// Reject windows where a future price falls this far below the last observed price.
const MAXIMUM_DECREASE: f32 = 0.002_f32;

// Highlight precision among predictions confident enough to support selective action.
const HIGH_CONFIDENCE_PROBABILITY: f32 = 0.8_f32;

// Reduce adjacent returns to a compact representation before the learned layers.
const POOL_SIZE: usize = 4;

// Configure the temporal convolutions and flattened pooled representation.
const KERNEL_SIZE: usize = 5;
const FIRST_CHANNELS: usize = 8;
const SECOND_CHANNELS: usize = 16;
const POOLED_LENGTH: usize = INPUTS / POOL_SIZE;
const LINEAR_INPUTS: usize = SECOND_CHANNELS * POOLED_LENGTH;

// These Eastern times bound the regular market session used by the model.
const MARKET_OPEN_TIME: Time = match Time::from_hms(9, 30, 0) {
    Ok(time) => time,
    Err(_) => panic!("The market open time must be valid."),
};
const MARKET_CLOSE_TIME: Time = match Time::from_hms(16, 0, 0) {
    Ok(time) => time,
    Err(_) => panic!("The market close time must be valid."),
};

// These arguments configure a time-series training run.
#[derive(ClapArgs)]
pub struct Args {
    /// CSV files used to train the model.
    #[arg(long, required = true, num_args = 1..)]
    training_paths: Vec<PathBuf>,

    /// CSV files used only to validate the model.
    #[arg(long, required = true, num_args = 1..)]
    validation_paths: Vec<PathBuf>,

    /// Number of complete passes through the training dataset.
    #[arg(long, default_value_t = 5, value_parser = parse_positive_usize)]
    epochs: usize,

    /// Step size used by the Adam optimizer.
    #[arg(long, default_value_t = 1e-4_f64, value_parser = parse_positive_f64)]
    learning_rate: f64,

    /// Probability of dropping each hidden activation during training.
    #[arg(long, default_value_t = 0.5_f64, value_parser = parse_dropout)]
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
    target: bool,
}

// This batch stores the tensors consumed and predicted by the model.
#[derive(Clone, Debug)]
struct SeriesBatch<B: Backend> {
    inputs: Tensor<B, 2>,
    targets: Tensor<B, 2, Int>,
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

        // Convert every crossing label into a single-column target tensor.
        let targets = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_ints([i32::from(item.target)], device).unsqueeze()
            })
            .collect();
        let targets = Tensor::cat(targets, 0);

        SeriesBatch { inputs, targets }
    }
}

// This model predicts whether a future price will cross the target increase.
#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    first_convolution: Conv1d<B>,
    first_dropout: Dropout,
    second_convolution: Conv1d<B>,
    second_dropout: Dropout,
    first_activation: Relu,
    pooling: AvgPool1d,
    first_linear: Linear<B>,
    third_dropout: Dropout,
    second_activation: Relu,
    second_linear: Linear<B>,
}

// This configuration records the model, training, and preprocessing settings.
#[derive(Config, Debug)]
pub struct ModelConfig {
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
            first_convolution: Conv1dConfig::new(1, FIRST_CHANNELS, KERNEL_SIZE)
                .with_padding(PaddingConfig1d::Same)
                .init(device),
            first_dropout: DropoutConfig::new(self.dropout).init(),
            second_convolution: Conv1dConfig::new(FIRST_CHANNELS, SECOND_CHANNELS, KERNEL_SIZE)
                .with_padding(PaddingConfig1d::Same)
                .init(device),
            second_dropout: DropoutConfig::new(self.dropout).init(),
            first_activation: Relu::new(),
            pooling: AvgPool1dConfig::new(POOL_SIZE).init(),
            first_linear: LinearConfig::new(LINEAR_INPUTS, 128).init(device),
            third_dropout: DropoutConfig::new(self.dropout).init(),
            second_activation: Relu::new(),
            second_linear: LinearConfig::new(128, 1).init(device),
        }
    }
}

impl<B: Backend> Model<B> {
    // Extract and downsample temporal features before producing the output logit.
    pub fn forward(&self, inputs: Tensor<B, 2>) -> Tensor<B, 2> {
        let values = self
            .first_dropout
            .forward(self.first_convolution.forward(inputs.unsqueeze_dim::<3>(1)));
        let values = self
            .second_dropout
            .forward(self.second_convolution.forward(values));
        let values = self.first_activation.forward(values);
        let values = self.pooling.forward(values).flatten(1_usize, 2_usize);
        let values = self.first_linear.forward(values);
        let values = self.third_dropout.forward(values);
        let values = self.second_activation.forward(values);
        self.second_linear.forward(values)
    }
}

// This structure contains prepared datasets and their normalization parameters.
struct PreparedData {
    training: Vec<SeriesItem>,
    validation: Vec<SeriesItem>,
    mean: f32,
    deviation: f32,
}

// Keep each parsed price attached to its source timestamp for aligned inference output.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimestampedPrice {
    pub timestamp: i64,
    pub price: f32,
}

// Read the price history and train a forecasting model.
pub fn run(args: &Args) -> Result<(), Box<dyn Error>> {
    // Load the two file groups independently so validation data never enters training.
    let training_series = load_series(&args.training_paths)?;
    let validation_series = load_series(&args.validation_paths)?;
    let data = prepare_data(&training_series, &validation_series, INPUTS, OUTPUTS)?;

    // Use Burn's CPU backend for consistent training and inference calculations.
    let device = FlexDevice;
    train::<Autodiff<Flex>>(&device, args, &data)?;

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
        .batch_size(BATCH_SIZE)
        .shuffle(config.seed)
        .num_workers(1)
        .build(InMemDataset::new(data.training.clone()));
    let training_evaluation_loader =
        DataLoaderBuilder::new(SeriesBatcher::<B::InnerBackend>::new())
            .batch_size(BATCH_SIZE)
            .num_workers(1)
            .build(InMemDataset::new(data.training.clone()));
    let validation_loader = DataLoaderBuilder::new(SeriesBatcher::<B::InnerBackend>::new())
        .batch_size(BATCH_SIZE)
        .num_workers(1)
        .build(InMemDataset::new(data.validation.clone()));

    // Report the dataset sizes before calculating initial reference metrics.
    println!(
        "Training on {} windows and validating on {} windows…",
        data.training.len(),
        data.validation.len(),
    );

    // Establish the classification baseline from the validation label distribution.
    let positive_count = data.validation.iter().filter(|item| item.target).count();
    let positive_rate = usize_to_f32(positive_count) / usize_to_f32(data.validation.len());
    let baseline_accuracy = positive_rate.max(1.0_f32 - positive_rate);
    println!(
        "Validation positive rate: {:.2}%, majority baseline accuracy: {:.2}%",
        positive_rate * 100.0_f32,
        baseline_accuracy * 100.0_f32,
    );

    // Report the initialized model's classification quality before optimizer steps.
    let initial_model = model.valid();
    let initial_training = validation_metrics(&initial_model, &training_evaluation_loader);
    let initial_validation = validation_metrics(&initial_model, &validation_loader);
    println!(
        "Initial: train loss {:.4}, accuracy {:.2}%; validation loss {:.4}, accuracy {:.2}%",
        initial_training.loss,
        initial_training.accuracy() * 100.0_f32,
        initial_validation.loss,
        initial_validation.accuracy() * 100.0_f32,
    );

    // Save the static configuration before producing replaceable epoch checkpoints.
    fs::create_dir_all(&args.model_directory)?;
    config.save(args.model_directory.join("model.json"))?;

    // Optimize binary cross-entropy and display validation progress each epoch.
    for epoch in 1..=config.epochs {
        train_epoch(
            &mut model,
            &mut optimizer,
            &training_loader,
            config.learning_rate,
        );

        // Evaluate the fully updated model on both splits for comparable epoch metrics.
        let valid_model = model.valid();
        let training = validation_metrics(&valid_model, &training_evaluation_loader);
        let validation = validation_metrics(&valid_model, &validation_loader);
        println!(
            concat!(
                "Epoch {:>2}/{}: train loss {:.4}, accuracy {:.2}%; ",
                "validation loss {:.4}, accuracy {:.2}%, precision {:.2}%, recall {:.2}%, ",
                "precision@0.8 {:.2}%",
            ),
            epoch,
            config.epochs,
            training.loss,
            training.accuracy() * 100.0_f32,
            validation.loss,
            validation.accuracy() * 100.0_f32,
            validation.precision() * 100.0_f32,
            validation.recall() * 100.0_f32,
            validation.high_confidence_precision() * 100.0_f32,
        );

        // Persist the latest completed epoch so interrupted runs retain usable parameters.
        println!(
            "Saving model after epoch {epoch} to {}…",
            args.model_directory.display(),
        );
        model
            .clone()
            .save_file(args.model_directory.join("model"), &CompactRecorder::new())?;
        println!(
            "Saved model after epoch {epoch} to {}.",
            args.model_directory.display(),
        );
    }

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
        let loss = BinaryCrossEntropyLossConfig::new()
            .with_logits(true)
            .init(&model.devices()[0])
            .forward(predictions, batch.targets);

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
    // Convert each price series independently so normalization never crosses a timestamp gap.
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

    // Normalize and label each price window separately so no example crosses a timestamp gap.
    let prepare = |series: &[Vec<f32>]| {
        let mut items = Vec::new();
        for prices in series {
            items.extend(windows(prices, inputs, outputs, mean, deviation));
        }
        items
    };
    let training = prepare(training_prices);
    let validation = prepare(validation_prices);

    Ok(PreparedData {
        training,
        validation,
        mean,
        deviation,
    })
}

// Create chronological examples labeled by future crossings of the target price.
fn windows(
    prices: &[f32],
    inputs: usize,
    outputs: usize,
    mean: f32,
    deviation: f32,
) -> Vec<SeriesItem> {
    prices
        .windows(inputs + outputs + 1)
        .map(|window| {
            // Calculate both barriers from the final price observed by the model.
            let reference_price = window[inputs];
            let upper_target = reference_price * (1.0_f32 + TARGET_INCREASE);
            let lower_limit = reference_price * (1.0_f32 - MAXIMUM_DECREASE);
            let future_prices = &window[inputs + 1..];
            let target = future_prices
                .iter()
                .find_map(|future_price| {
                    if *future_price > upper_target {
                        Some(true)
                    } else if *future_price < lower_limit {
                        Some(false)
                    } else {
                        None
                    }
                })
                .unwrap_or(false);

            // Normalize only the returns presented to the model, leaving the price label exact.
            let inputs = log_returns(&window[..=inputs])
                .into_iter()
                .map(|value| (value - mean) / deviation)
                .collect();

            SeriesItem { inputs, target }
        })
        .collect()
}

// These metrics summarize binary predictions at the conventional 0.5 threshold.
struct ClassificationMetrics {
    loss: f32,
    correct: usize,
    total: usize,
    true_positives: usize,
    predicted_positives: usize,
    actual_positives: usize,
    high_confidence_true_positives: usize,
    high_confidence_predicted_positives: usize,
}

impl ClassificationMetrics {
    // Calculate the share of labels classified correctly.
    fn accuracy(&self) -> f32 {
        usize_to_f32(self.correct) / usize_to_f32(self.total)
    }

    // Calculate positive predictive value, defining an empty prediction set as zero.
    fn precision(&self) -> f32 {
        if self.predicted_positives == 0 {
            return 0.0_f32;
        }
        usize_to_f32(self.true_positives) / usize_to_f32(self.predicted_positives)
    }

    // Calculate the share of actual positives detected, defining an empty class as zero.
    fn recall(&self) -> f32 {
        if self.actual_positives == 0 {
            return 0.0_f32;
        }
        usize_to_f32(self.true_positives) / usize_to_f32(self.actual_positives)
    }

    // Calculate precision among predictions at or above the high-confidence threshold.
    fn high_confidence_precision(&self) -> f32 {
        if self.high_confidence_predicted_positives == 0 {
            return 0.0_f32;
        }
        usize_to_f32(self.high_confidence_true_positives)
            / usize_to_f32(self.high_confidence_predicted_positives)
    }
}

// Measure binary loss and thresholded classification quality without autodiff.
fn validation_metrics<B: Backend>(
    model: &Model<B>,
    loader: &Arc<dyn DataLoader<B, SeriesBatch<B>>>,
) -> ClassificationMetrics {
    // Accumulate sample-weighted loss and confusion counts across ordered windows.
    let mut total_loss = 0.0_f32;
    let mut metrics = ClassificationMetrics {
        loss: 0.0_f32,
        correct: 0,
        total: 0,
        true_positives: 0,
        predicted_positives: 0,
        actual_positives: 0,
        high_confidence_true_positives: 0,
        high_confidence_predicted_positives: 0,
    };
    let high_confidence_logit =
        (HIGH_CONFIDENCE_PROBABILITY / (1.0_f32 - HIGH_CONFIDENCE_PROBABILITY)).ln();
    for batch in loader.iter() {
        let item_count = batch.targets.dims()[0];
        let predictions = model.forward(batch.inputs);
        let loss = BinaryCrossEntropyLossConfig::new()
            .with_logits(true)
            .init(&model.devices()[0])
            .forward(predictions.clone(), batch.targets.clone());
        total_loss += loss.into_scalar().elem::<f32>() * usize_to_f32(item_count);
        let logits = predictions.into_data().to_vec::<f32>().unwrap();
        let targets = batch.targets.float().into_data().to_vec::<f32>().unwrap();

        // Count classifications directly so the reporting logic remains easy to inspect.
        for (logit, target) in logits.into_iter().zip(targets) {
            let prediction = logit >= 0.0_f32;
            let high_confidence_prediction = logit >= high_confidence_logit;
            let target = target >= 0.5_f32;
            metrics.correct += usize::from(prediction == target);
            metrics.true_positives += usize::from(prediction && target);
            metrics.predicted_positives += usize::from(prediction);
            metrics.actual_positives += usize::from(target);
            metrics.high_confidence_true_positives +=
                usize::from(high_confidence_prediction && target);
            metrics.high_confidence_predicted_positives += usize::from(high_confidence_prediction);
            metrics.total += 1;
        }
    }

    metrics.loss = total_loss / usize_to_f32(metrics.total);
    metrics
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

// Parse contiguous market-hours series while retaining timestamps beside their prices.
pub fn parse_price_series(contents: &str) -> Result<Vec<Vec<TimestampedPrice>>, Box<dyn Error>> {
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
    let mut price_series = Vec::<Vec<TimestampedPrice>>::new();
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
        let unreliable =
            eastern_time >= UNRELIABLE_DATA_START_TIME && eastern_time < UNRELIABLE_DATA_END_TIME;
        let outside_market = eastern_time < MARKET_OPEN_TIME || eastern_time >= MARKET_CLOSE_TIME;
        if unreliable || outside_market {
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
            .push(TimestampedPrice { timestamp, price });
        previous_timestamp = Some(timestamp);
    }

    Ok(price_series)
}

// Discard timestamps after parsing because training consumes only price values.
fn parse_training_prices(contents: &str) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    Ok(parse_price_series(contents)?
        .into_iter()
        .map(|series| series.into_iter().map(|price| price.price).collect())
        .collect())
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
    use super::{parse_dropout, parse_price_series, parse_training_prices, prepare_data, windows};
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
        assert_eq!(args.epochs, 5);
        assert!((args.learning_rate - 1e-4_f64).abs() < f64::EPSILON);
        assert!((args.dropout - 0.5_f64).abs() < f64::EPSILON);
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
        let prices = parse_training_prices(concat!(
            "date,open,high,low,close,volume,wap,count\n",
            "1784035800,100,900,1,2,3,4,5\n",
            "1784035801,101.5,800,1,2,3,4,5\n",
            "1784035802,99,700,1,2,3,4,5\n",
        ))
        .unwrap();

        assert_eq!(prices, vec![vec![100.0_f32, 101.5_f32, 99.0_f32]]);
    }

    #[test]
    fn discard_unreliable_training_prices_across_eastern_time_offsets() {
        // Ignore premarket, unreliable, and closing rows while retaining the regular session.
        for start in [1_784_016_000_i64, 1_767_949_200_i64] {
            let contents = format!(
                concat!(
                    "date,open\n",
                    "{},invalid\n",
                    "{},invalid\n",
                    "{},invalid\n",
                    "{},99\n",
                    "{},100\n",
                    "{},invalid\n",
                ),
                start - 1,
                start,
                start + 899,
                start + 19_800,
                start + 19_801,
                start + 43_200,
            );

            assert_eq!(
                parse_training_prices(&contents).unwrap(),
                vec![vec![99.0_f32, 100.0_f32]],
            );
            assert_eq!(
                parse_price_series(&contents).unwrap()[0]
                    .iter()
                    .map(|price| price.timestamp)
                    .collect::<Vec<_>>(),
                vec![start + 19_800, start + 19_801],
            );
        }
    }

    #[test]
    fn split_prices_at_timestamp_gaps() {
        // Keep one-second observations together while separating gaps and reversed timestamps.
        let prices = parse_training_prices(concat!(
            "date,open\n",
            "1784035800,100\n",
            "1784035801,101\n",
            "1784035803,102\n",
            "1784035804,103\n",
            "1784035802,104\n",
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
        let error = parse_training_prices("date,open\n1784035800,100\n1784035801,0\n").unwrap_err();

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
        assert!(data.training.iter().all(|item| item.target));
    }

    #[test]
    fn label_future_price_crossings() {
        // Require the strict 0.5% gain to occur before any strict 0.4% loss.
        let upper_target = 100.0_f32 * (1.0_f32 + super::TARGET_INCREASE);
        let lower_limit = 100.0_f32 * (1.0_f32 - super::MAXIMUM_DECREASE);
        let crossing = windows(
            &[90.0_f32, 95.0, 100.0, lower_limit, upper_target + 0.1_f32],
            2,
            2,
            0.0,
            1.0,
        );
        let no_gain = windows(
            &[90.0_f32, 95.0, 100.0, upper_target, upper_target],
            2,
            2,
            0.0,
            1.0,
        );
        let loss_before_gain = windows(
            &[
                90.0_f32,
                95.0,
                100.0,
                lower_limit - 0.1_f32,
                upper_target + 0.1_f32,
            ],
            2,
            2,
            0.0,
            1.0,
        );
        let loss_after_gain = windows(
            &[
                90.0_f32,
                95.0,
                100.0,
                upper_target + 0.1_f32,
                lower_limit - 0.1_f32,
            ],
            2,
            2,
            0.0,
            1.0,
        );

        assert_eq!(crossing[0].inputs.len(), 2);
        assert!(crossing[0].target);
        assert!(!no_gain[0].target);
        assert!(!loss_before_gain[0].target);
        assert!(loss_after_gain[0].target);
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
