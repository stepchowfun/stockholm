use crate::backtest::{UNRELIABLE_DATA_END_TIME, UNRELIABLE_DATA_START_TIME};
use burn::{
    backend::{Autodiff, Flex, flex::FlexDevice},
    data::{
        dataloader::{DataLoader, DataLoaderBuilder, batcher::Batcher},
        dataset::InMemDataset,
    },
    module::AutodiffModule,
    nn::{
        Dropout, DropoutConfig, Linear, LinearConfig, Relu,
        loss::CrossEntropyLossConfig,
        pool::{AvgPool1d, AvgPool1dConfig},
    },
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
};
use chrono::Local;
use clap::Args as ClapArgs;
use std::{
    error::Error,
    fs,
    io::{self, Write},
    marker::PhantomData,
    path::{Path, PathBuf},
    sync::Arc,
};
use tempfile::{Builder as TempFileBuilder, NamedTempFile};
use time::{OffsetDateTime, Time};
use time_tz::{OffsetDateTimeExt, timezones::db::america::NEW_YORK};

// Fix the price history supplied to the model and the future crossing horizon.
pub const INPUTS: usize = 128;
const OUTPUTS: usize = 128;

// Bound memory use and keep optimizer updates frequent enough for this overlapping dataset.
pub const BATCH_SIZE: usize = 64;

// Require a future price to exceed the last observed price by this relative amount.
pub const TARGET_INCREASE: f32 = 0.005_f32;

// Reject windows where a future price falls this far below the last observed price.
pub const MAXIMUM_DECREASE: f32 = 0.005_f32;

// Aggressively reduce adjacent returns before the learned layers.
const POOL_SIZE: usize = 8;
const POOLED_LENGTH: usize = INPUTS / POOL_SIZE;

// Configure the hidden linear layer.
const LINEAR_OUTPUTS: usize = 128;

// Assign stable indices to the mutually exclusive future price outcomes.
pub const OUTCOME_COUNT: usize = 3;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PriceOutcome {
    LowerFirst = 0,
    UpperFirst = 1,
    Neither = 2,
}

// These Eastern times bound the regular market session used by the model.
pub const MARKET_OPEN_TIME: Time = match Time::from_hms(9, 30, 0) {
    Ok(time) => time,
    Err(_) => panic!("The market open time must be valid."),
};
pub const MARKET_CLOSE_TIME: Time = match Time::from_hms(16, 0, 0) {
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
    target: PriceOutcome,
}

// This batch stores the tensors consumed and predicted by the model.
#[derive(Clone, Debug)]
struct SeriesBatch<B: Backend> {
    inputs: Tensor<B, 2>,
    targets: Tensor<B, 1, Int>,
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

        // Convert every outcome into the class index expected by cross-entropy loss.
        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1, Int>::from_ints([item.target as i32], device))
            .collect();
        let targets = Tensor::cat(targets, 0);

        SeriesBatch { inputs, targets }
    }
}

// This model predicts which price barrier will be reached first, if either.
#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    pooling: AvgPool1d,
    first_linear: Linear<B>,
    dropout: Dropout,
    activation: Relu,
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
            pooling: AvgPool1dConfig::new(POOL_SIZE).init(),
            first_linear: LinearConfig::new(POOLED_LENGTH, LINEAR_OUTPUTS).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            activation: Relu::new(),
            second_linear: LinearConfig::new(LINEAR_OUTPUTS, OUTCOME_COUNT).init(device),
        }
    }
}

impl<B: Backend> Model<B> {
    // Downsample each return window before producing the outcome logits.
    pub fn forward(&self, inputs: Tensor<B, 2>) -> Tensor<B, 2> {
        let values = self
            .pooling
            .forward(inputs.unsqueeze_dim::<3>(1))
            .flatten(1_usize, 2_usize);
        let values = self.first_linear.forward(values);
        let values = self.dropout.forward(values);
        let values = self.activation.forward(values);
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
    let mut config = ModelConfig::new(
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
    let mut class_counts = [0_usize; OUTCOME_COUNT];
    for item in &data.validation {
        class_counts[item.target as usize] += 1;
    }
    let class_rates =
        class_counts.map(|count| usize_to_f32(count) / usize_to_f32(data.validation.len()));
    let baseline_accuracy = class_rates.into_iter().fold(0.0_f32, f32::max);
    println!(
        concat!(
            "Validation rates: lower first {:.2}%, upper first {:.2}%, neither {:.2}%; ",
            "majority baseline accuracy: {:.2}%",
        ),
        class_rates[PriceOutcome::LowerFirst as usize] * 100.0_f32,
        class_rates[PriceOutcome::UpperFirst as usize] * 100.0_f32,
        class_rates[PriceOutcome::Neither as usize] * 100.0_f32,
        baseline_accuracy * 100.0_f32,
    );

    // Report the initialized model's classification quality before optimizer steps.
    let initial_model = model.valid();
    let initial_training = validation_metrics(&initial_model, &training_evaluation_loader);
    let initial_validation = validation_metrics(&initial_model, &validation_loader);
    report_metrics("Initial", &initial_training, &initial_validation);

    // Retain the requested duration while the saved configuration tracks completed work.
    let requested_epochs = config.epochs;
    for epoch in 1..=requested_epochs {
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
        let label = format!("Epoch {epoch:>2}/{requested_epochs}");
        report_metrics(&label, &training, &validation);

        // Persist the latest completed epoch so interrupted runs retain usable parameters.
        config.epochs = epoch;
        println!(
            "Saving model after epoch {epoch} to {}…",
            args.model_directory.display(),
        );
        save_checkpoint(model.clone(), &config, &args.model_directory)?;
        let saved_at = Local::now().to_rfc3339();
        println!(
            "Saved model after epoch {epoch} to {} at {saved_at}.",
            args.model_directory.display(),
        );
    }

    Ok(())
}

// Stage a complete checkpoint before atomically replacing its final files.
fn save_checkpoint<B: Backend>(
    model: Model<B>,
    config: &ModelConfig,
    directory: &Path,
) -> Result<(), Box<dyn Error>> {
    // Keep temporary and final files on the same filesystem so persistence stays atomic.
    fs::create_dir_all(directory)?;
    let parent = directory
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let staged_config = TempFileBuilder::new()
        .suffix(".json")
        .tempfile_in(parent)?
        .into_temp_path();
    let staged_model = TempFileBuilder::new()
        .suffix(".mpk")
        .tempfile_in(parent)?
        .into_temp_path();

    // Finish serializing both temporary files before exposing either one to inference.
    config.save(&staged_config)?;
    model.save_file(staged_model.with_extension(""), &CompactRecorder::new())?;

    // Replace the model last so an interruption during serialization preserves the prior model.
    persist_file(&staged_config, &directory.join("model.json"))?;
    persist_file(&staged_model, &directory.join("model.mpk"))?;

    Ok(())
}

// Copy one staged artifact into a temporary file and atomically persist it at its destination.
fn persist_file(source: &Path, destination: &Path) -> Result<(), Box<dyn Error>> {
    // Follow the state-file persistence pattern while accommodating Burn-managed file names.
    let parent = destination
        .parent()
        .ok_or("the checkpoint destination must have a parent directory")?;
    let mut source = fs::File::open(source)?;
    let mut temp_file = NamedTempFile::new_in(parent)?;
    io::copy(&mut source, &mut temp_file)?;
    temp_file.flush()?;
    temp_file.persist(destination)?;

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
        let loss = CrossEntropyLossConfig::new()
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
                        Some(PriceOutcome::UpperFirst)
                    } else if *future_price < lower_limit {
                        Some(PriceOutcome::LowerFirst)
                    } else {
                        None
                    }
                })
                .unwrap_or(PriceOutcome::Neither);

            // Normalize only the returns presented to the model, leaving the price label exact.
            let inputs = log_returns(&window[..=inputs])
                .into_iter()
                .map(|value| (value - mean) / deviation)
                .collect();

            SeriesItem { inputs, target }
        })
        .collect()
}

// These metrics summarize multiclass predictions and each outcome's retrieval quality.
struct ClassificationMetrics {
    loss: f32,
    correct: usize,
    total: usize,
    true_positives: [usize; OUTCOME_COUNT],
    predicted_positives: [usize; OUTCOME_COUNT],
    actual_positives: [usize; OUTCOME_COUNT],
}

impl ClassificationMetrics {
    // Calculate the share of labels classified correctly.
    fn accuracy(&self) -> f32 {
        usize_to_f32(self.correct) / usize_to_f32(self.total)
    }

    // Calculate positive predictive value, defining an empty prediction set as zero.
    fn precision(&self, outcome: PriceOutcome) -> f32 {
        let class = outcome as usize;
        if self.predicted_positives[class] == 0 {
            return 0.0_f32;
        }
        usize_to_f32(self.true_positives[class]) / usize_to_f32(self.predicted_positives[class])
    }

    // Calculate the share of actual positives detected, defining an empty class as zero.
    fn recall(&self, outcome: PriceOutcome) -> f32 {
        let class = outcome as usize;
        if self.actual_positives[class] == 0 {
            return 0.0_f32;
        }
        usize_to_f32(self.true_positives[class]) / usize_to_f32(self.actual_positives[class])
    }
}

// Report comparable training and validation metrics at one point in optimization.
fn report_metrics(
    label: &str,
    training: &ClassificationMetrics,
    validation: &ClassificationMetrics,
) {
    println!(
        concat!(
            "{}: train loss {:.4}, accuracy {:.2}%; validation loss {:.4}, accuracy {:.2}%; ",
            "lower precision {:.2}%, recall {:.2}%; upper precision {:.2}%, recall {:.2}%; ",
            "neither precision {:.2}%, recall {:.2}%",
        ),
        label,
        training.loss,
        training.accuracy() * 100.0_f32,
        validation.loss,
        validation.accuracy() * 100.0_f32,
        validation.precision(PriceOutcome::LowerFirst) * 100.0_f32,
        validation.recall(PriceOutcome::LowerFirst) * 100.0_f32,
        validation.precision(PriceOutcome::UpperFirst) * 100.0_f32,
        validation.recall(PriceOutcome::UpperFirst) * 100.0_f32,
        validation.precision(PriceOutcome::Neither) * 100.0_f32,
        validation.recall(PriceOutcome::Neither) * 100.0_f32,
    );
}

// Measure multiclass loss and classification quality without autodiff.
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
        true_positives: [0; OUTCOME_COUNT],
        predicted_positives: [0; OUTCOME_COUNT],
        actual_positives: [0; OUTCOME_COUNT],
    };
    for batch in loader.iter() {
        let item_count = batch.targets.dims()[0];
        let predictions = model.forward(batch.inputs);
        let loss = CrossEntropyLossConfig::new()
            .init(&model.devices()[0])
            .forward(predictions.clone(), batch.targets.clone());
        total_loss += loss.into_scalar().elem::<f32>() * usize_to_f32(item_count);
        let logits = predictions.into_data().to_vec::<f32>().unwrap();
        let targets = batch.targets.into_data().to_vec::<i32>().unwrap();

        // Count classifications directly so the reporting logic remains easy to inspect.
        for (class_logits, target) in logits.as_chunks::<OUTCOME_COUNT>().0.iter().zip(targets) {
            let prediction = class_logits
                .iter()
                .enumerate()
                .max_by(|(_, left), (_, right)| left.total_cmp(right))
                .map(|(class, _)| class)
                .unwrap();
            let target = usize::try_from(target).unwrap();
            metrics.correct += usize::from(prediction == target);
            metrics.true_positives[target] += usize::from(prediction == target);
            metrics.predicted_positives[prediction] += 1;
            metrics.actual_positives[target] += 1;
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
    use super::{
        PriceOutcome, parse_dropout, parse_price_series, parse_training_prices, prepare_data,
        windows,
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
        assert!(
            data.training
                .iter()
                .all(|item| item.target == PriceOutcome::UpperFirst),
        );
    }

    #[test]
    fn label_future_price_crossings() {
        // Distinguish strict upper-first, lower-first, and unresolved future paths.
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
        assert_eq!(crossing[0].target, PriceOutcome::UpperFirst);
        assert_eq!(no_gain[0].target, PriceOutcome::Neither);
        assert_eq!(loss_before_gain[0].target, PriceOutcome::LowerFirst);
        assert_eq!(loss_after_gain[0].target, PriceOutcome::UpperFirst);
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
