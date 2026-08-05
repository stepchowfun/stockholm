use burn::{
    backend::{Autodiff, NdArray, ndarray::NdArrayDevice},
    data::{
        dataloader::{DataLoader, DataLoaderBuilder, batcher::Batcher},
        dataset::InMemDataset,
    },
    module::AutodiffModule,
    nn::{
        Linear, LinearConfig, Relu,
        loss::{MseLoss, Reduction::Mean},
    },
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
};
use clap::Args as ClapArgs;
use std::{error::Error, fs, marker::PhantomData, path::PathBuf, sync::Arc};

// These arguments configure a time-series training run.
#[derive(ClapArgs)]
pub struct Args {
    /// CSV files used to train the model.
    #[arg(long, required = true, num_args = 1..)]
    training_paths: Vec<PathBuf>,

    /// CSV files used only to validate the model.
    #[arg(long, required = true, num_args = 1..)]
    validation_paths: Vec<PathBuf>,

    /// Number of recent returns supplied to the model.
    #[arg(long, default_value_t = 20, value_parser = parse_positive_usize)]
    inputs: usize,

    /// Number of future returns predicted by the model.
    #[arg(long, default_value_t = 5, value_parser = parse_positive_usize)]
    outputs: usize,

    /// Number of examples processed in each optimization step.
    #[arg(long, default_value_t = 64, value_parser = parse_positive_usize)]
    batch_size: usize,

    /// Number of complete passes through the training dataset.
    #[arg(long, default_value_t = 50, value_parser = parse_positive_usize)]
    epochs: usize,

    /// Step size used by the Adam optimizer.
    #[arg(long, default_value_t = 1e-3, value_parser = parse_positive_f64)]
    learning_rate: f64,

    /// Seed used for model initialization and training-data shuffling.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Directory where the trained model and its metadata will be saved.
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
struct Model<B: Backend> {
    input: Linear<B>,
    hidden: Linear<B>,
    output: Linear<B>,
    activation: Relu,
}

// This configuration defines the model dimensions.
#[derive(Config, Debug)]
struct ModelConfig {
    inputs: usize,
    hidden: usize,
    outputs: usize,
}

impl ModelConfig {
    // Initialize every model layer on the selected device.
    fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            input: LinearConfig::new(self.inputs, self.hidden).init(device),
            hidden: LinearConfig::new(self.hidden, self.hidden).init(device),
            output: LinearConfig::new(self.hidden, self.outputs).init(device),
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> Model<B> {
    // Apply the two hidden transformations and output projection.
    fn forward(&self, inputs: Tensor<B, 2>) -> Tensor<B, 2> {
        let values = self.activation.forward(self.input.forward(inputs));
        let values = self.activation.forward(self.hidden.forward(values));
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
    let data = prepare_data(
        &training_series,
        &validation_series,
        args.inputs,
        args.outputs,
    )?;

    // Use Burn's portable CPU backend for deterministic local training.
    let device = NdArrayDevice::Cpu;
    train::<Autodiff<NdArray>>(&device, args, &data)?;

    Ok(())
}

// Load opening-price returns while preserving every file as an independent series.
fn load_series(paths: &[PathBuf]) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    paths
        .iter()
        .map(|path| {
            let contents = fs::read_to_string(path)
                .map_err(|error| format!("failed to read {}: {error}", path.display()))?;
            let prices = parse_prices(&contents)
                .map_err(|error| format!("failed to parse {}: {error}", path.display()))?;
            Ok(log_returns(&prices))
        })
        .collect()
}

// Train the model, evaluate it chronologically, and save its artifacts.
fn train<B: AutodiffBackend>(
    device: &B::Device,
    args: &Args,
    data: &PreparedData,
) -> Result<(), Box<dyn Error>> {
    // Seed Burn and construct a modest model based on the requested dimensions.
    B::seed(device, args.seed);
    let hidden = (args.inputs + args.outputs).next_power_of_two().max(32);
    let config = ModelConfig::new(args.inputs, hidden, args.outputs);
    let mut model = config.init::<B>(device);
    let mut optimizer = AdamConfig::new().init::<B, Model<B>>();

    // Scan all windows while shuffling only the training order each epoch.
    let training_loader = DataLoaderBuilder::new(SeriesBatcher::<B>::new())
        .batch_size(args.batch_size)
        .shuffle(args.seed)
        .num_workers(1)
        .build(InMemDataset::new(data.training.clone()));
    let validation_loader = DataLoaderBuilder::new(SeriesBatcher::<B::InnerBackend>::new())
        .batch_size(args.batch_size)
        .num_workers(1)
        .build(InMemDataset::new(data.validation.clone()));

    // Optimize mean-squared error and display validation progress each epoch.
    println!(
        "Training on {} windows and validating on {} windows…",
        data.training.len(),
        data.validation.len(),
    );
    for epoch in 1..=args.epochs {
        let training_loss = train_epoch(
            &mut model,
            &mut optimizer,
            &training_loader,
            args.learning_rate,
        );
        let validation_loss = validation_loss(&model.valid(), &validation_loader);
        println!(
            "Epoch {:>2}/{}: train RMSE {:.2} bps, validation RMSE {:.2} bps",
            epoch,
            args.epochs,
            training_loss.sqrt() * data.deviation * 10_000.0,
            validation_loss.sqrt() * data.deviation * 10_000.0,
        );
    }

    // Compare the model against predicting no future price movement.
    let final_loss = validation_loss(&model.valid(), &validation_loader);
    let baseline_loss = baseline_loss(&data.validation, data.mean, data.deviation);
    println!(
        "Validation RMSE: model {:.2} bps, no-change baseline {:.2} bps",
        final_loss.sqrt() * data.deviation * 10_000.0,
        baseline_loss.sqrt() * data.deviation * 10_000.0,
    );

    // Save both the Burn model and the values needed to reconstruct price forecasts.
    fs::create_dir_all(&args.model_directory)?;
    model.save_file(args.model_directory.join("model"), &CompactRecorder::new())?;
    config.save(args.model_directory.join("model.json"))?;
    save_metadata(args, data)?;
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
) -> f32
where
    O: Optimizer<Model<B>, B>,
{
    // Accumulate sample-weighted loss while applying each gradient update.
    let mut total_loss = 0.0_f32;
    let mut total_items = 0_usize;
    for batch in loader.iter() {
        let item_count = batch.targets.dims()[0];
        let predictions = model.forward(batch.inputs);
        let loss = MseLoss::new().forward(predictions, batch.targets, Mean);
        total_loss += loss.clone().into_scalar().elem::<f32>() * usize_to_f32(item_count);
        total_items += item_count;

        // Associate gradients with model parameters before applying Adam.
        let gradients = GradientsParams::from_grads(loss.backward(), model);
        *model = optimizer.step(learning_rate, model.clone(), gradients);
    }

    total_loss / usize_to_f32(total_items)
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

// Parse finite positive opening prices from a historical-data CSV file.
fn parse_prices(contents: &str) -> Result<Vec<f32>, Box<dyn Error>> {
    // Locate the opening-price column by name so column order remains explicit.
    let mut reader = csv::Reader::from_reader(contents.as_bytes());
    let headers = reader.headers()?;
    let open_index = headers
        .iter()
        .position(|header| header == "open")
        .ok_or("the CSV file must contain an open column")?;

    // Attach line numbers to malformed values so input problems are actionable.
    let mut prices = Vec::new();
    for (index, result) in reader.records().enumerate() {
        let record = result?;
        let line = index + 2;
        let value = record
            .get(open_index)
            .ok_or_else(|| format!("missing opening price on line {line}"))?;
        let price: f32 = value
            .parse()
            .map_err(|error| format!("invalid opening price on line {line}: {error}"))?;
        if !price.is_finite() || price <= 0.0 {
            return Err(format!("opening price on line {line} must be finite and positive").into());
        }
        prices.push(price);
    }

    Ok(prices)
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

// Convert raw prices into stationary relative changes.
fn log_returns(prices: &[f32]) -> Vec<f32> {
    prices
        .windows(2)
        .map(|pair| (pair[1] / pair[0]).ln())
        .collect()
}

// Normalize from training files and create windows within every independent series.
fn prepare_data(
    training_series: &[Vec<f32>],
    validation_series: &[Vec<f32>],
    inputs: usize,
    outputs: usize,
) -> Result<PreparedData, Box<dyn Error>> {
    // Require every file to contain at least one complete model window.
    for (kind, series) in [
        ("training", training_series),
        ("validation", validation_series),
    ] {
        if series.is_empty() {
            return Err(format!("at least one {kind} file is required").into());
        }
        for (index, returns) in series.iter().enumerate() {
            if returns.len() < inputs + outputs {
                return Err(format!(
                    "{kind} file {} is too short for {inputs} inputs and {outputs} outputs",
                    index + 1,
                )
                .into());
            }
        }
    }

    // Fit normalization exclusively to returns from the training files.
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

    // Normalize and window each file separately so no example crosses a file boundary.
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
    let training = prepare(training_series);
    let validation = prepare(validation_series);

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

// Measure the normalized error from predicting zero return at every horizon.
fn baseline_loss(items: &[SeriesItem], mean: f32, deviation: f32) -> f32 {
    // Zero raw return has this value after applying the training normalization.
    let prediction = -mean / deviation;
    let squared_error = items
        .iter()
        .flat_map(|item| item.targets.iter())
        .map(|target| (target - prediction).powi(2))
        .sum::<f32>();
    let target_count = items.iter().map(|item| item.targets.len()).sum::<usize>();
    squared_error / usize_to_f32(target_count)
}

// Convert collection sizes for floating-point averages where minor rounding is acceptable.
#[allow(clippy::cast_precision_loss)]
fn usize_to_f32(value: usize) -> f32 {
    value as f32
}

// Persist preprocessing values required to convert predicted returns back to prices.
fn save_metadata(args: &Args, data: &PreparedData) -> Result<(), Box<dyn Error>> {
    // Use a simple text format that remains readable without model tooling.
    let metadata = format!(
        concat!(
            "inputs={}\noutputs={}\nbatch_size={}\nepochs={}\n",
            "learning_rate={}\nseed={}\nreturn_mean={}\nreturn_deviation={}\n",
        ),
        args.inputs,
        args.outputs,
        args.batch_size,
        args.epochs,
        args.learning_rate,
        args.seed,
        data.mean,
        data.deviation,
    );
    fs::write(args.model_directory.join("metadata.txt"), metadata)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{baseline_loss, log_returns, parse_prices, prepare_data};
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
        assert_eq!(args.inputs, 20);
        assert_eq!(args.outputs, 5);
        assert_eq!(args.batch_size, 64);
        assert_eq!(args.epochs, 50);
        assert!((args.learning_rate - 1e-3).abs() < f64::EPSILON);
        assert_eq!(args.seed, 42);
        assert_eq!(args.model_directory, PathBuf::from("model"));
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

        assert_eq!(prices, vec![100.0, 101.5, 99.0]);
    }

    #[test]
    fn reject_nonpositive_prices() {
        // Confirm logarithmic preprocessing rejects values outside its domain.
        let error = parse_prices("date,open\n1,100\n2,0\n").unwrap_err();

        assert!(error.to_string().contains("finite and positive"));
    }

    #[test]
    fn create_separate_training_and_validation_windows() {
        // Confirm complete files contribute only to their selected dataset.
        let prices = (1_u16..=101).map(f32::from).collect::<Vec<_>>();
        let returns = log_returns(&prices);
        let series = std::slice::from_ref(&returns);
        let data = prepare_data(series, series, 4, 2).unwrap();

        assert_eq!(data.training.len(), 95);
        assert_eq!(data.validation.len(), 95);
        assert!(baseline_loss(&data.validation, data.mean, data.deviation).is_finite());
    }

    #[test]
    fn keep_file_windows_separate() {
        // Confirm combining files does not add windows across their boundary.
        let prices = (1_u16..=101).map(f32::from).collect::<Vec<_>>();
        let returns = log_returns(&prices);
        let series = [returns.clone(), returns];
        let data = prepare_data(&series, &series, 4, 2).unwrap();

        assert_eq!(data.training.len(), 190);
        assert_eq!(data.validation.len(), 190);
    }
}
