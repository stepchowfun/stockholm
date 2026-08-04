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

// These constants configure the initial training workflow.
const BATCH_SIZE: usize = 64;
const EPOCHS: usize = 50;
const LEARNING_RATE: f64 = 1e-3;
const SEED: u64 = 42;
const TRAINING_PERCENT: usize = 80;

// These arguments configure a time-series training run.
#[derive(ClapArgs)]
pub struct Args {
    /// Text file containing one positive stock price per line.
    path: PathBuf,

    /// Number of recent returns supplied to the model.
    #[arg(long, default_value_t = 20, value_parser = parse_positive_usize)]
    inputs: usize,

    /// Number of future returns predicted by the model.
    #[arg(long, default_value_t = 5, value_parser = parse_positive_usize)]
    outputs: usize,

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
    // Load and validate every raw price before deriving returns.
    let contents = fs::read_to_string(&args.path)?;
    let prices = parse_prices(&contents)?;
    let returns = log_returns(&prices);
    let data = prepare_data(&returns, args.inputs, args.outputs)?;

    // Use Burn's portable CPU backend for deterministic local training.
    let device = NdArrayDevice::Cpu;
    train::<Autodiff<NdArray>>(&device, args, &data)?;

    Ok(())
}

// Train the model, evaluate it chronologically, and save its artifacts.
fn train<B: AutodiffBackend>(
    device: &B::Device,
    args: &Args,
    data: &PreparedData,
) -> Result<(), Box<dyn Error>> {
    // Seed Burn and construct a modest model based on the requested dimensions.
    B::seed(device, SEED);
    let hidden = (args.inputs + args.outputs).next_power_of_two().max(32);
    let config = ModelConfig::new(args.inputs, hidden, args.outputs);
    let mut model = config.init::<B>(device);
    let mut optimizer = AdamConfig::new().init::<B, Model<B>>();

    // Scan all windows while shuffling only the training order each epoch.
    let training_loader = DataLoaderBuilder::new(SeriesBatcher::<B>::new())
        .batch_size(BATCH_SIZE)
        .shuffle(SEED)
        .num_workers(1)
        .build(InMemDataset::new(data.training.clone()));
    let validation_loader = DataLoaderBuilder::new(SeriesBatcher::<B::InnerBackend>::new())
        .batch_size(BATCH_SIZE)
        .num_workers(1)
        .build(InMemDataset::new(data.validation.clone()));

    // Optimize mean-squared error and display validation progress each epoch.
    println!(
        "Training on {} windows and validating on {} windows…",
        data.training.len(),
        data.validation.len(),
    );
    for epoch in 1..=EPOCHS {
        let training_loss = train_epoch(&mut model, &mut optimizer, &training_loader);
        let validation_loss = validation_loss(&model.valid(), &validation_loader);
        println!(
            "Epoch {epoch:>2}/{EPOCHS}: train RMSE {:.6}, validation RMSE {:.6}",
            training_loss.sqrt(),
            validation_loss.sqrt(),
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
        *model = optimizer.step(LEARNING_RATE, model.clone(), gradients);
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

// Parse one finite positive stock price from each nonempty input line.
fn parse_prices(contents: &str) -> Result<Vec<f32>, Box<dyn Error>> {
    // Attach line numbers to malformed values so input problems are actionable.
    let mut prices = Vec::new();
    for (index, line) in contents.lines().enumerate() {
        let value = line.trim();
        if value.is_empty() {
            continue;
        }
        let price: f32 = value
            .parse()
            .map_err(|error| format!("invalid price on line {}: {error}", index + 1))?;
        if !price.is_finite() || price <= 0.0 {
            return Err(format!("price on line {} must be finite and positive", index + 1).into());
        }
        prices.push(price);
    }

    // At least two prices are required to derive one return.
    if prices.len() < 2 {
        return Err("the input file must contain at least two prices".into());
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

// Convert raw prices into stationary relative changes.
fn log_returns(prices: &[f32]) -> Vec<f32> {
    prices
        .windows(2)
        .map(|pair| (pair[1] / pair[0]).ln())
        .collect()
}

// Split chronologically, normalize from training history, and create every window.
fn prepare_data(
    returns: &[f32],
    inputs: usize,
    outputs: usize,
) -> Result<PreparedData, Box<dyn Error>> {
    // Reserve the final portion exclusively for future validation targets.
    let split = returns.len() * TRAINING_PERCENT / 100;
    if split < inputs + outputs || returns.len() - split < outputs || split < inputs {
        return Err(format!(
            "need more prices for {inputs} inputs, {outputs} outputs, and an 80/20 split",
        )
        .into());
    }

    // Fit normalization only to data available before the validation boundary.
    let mean = returns[..split].iter().sum::<f32>() / usize_to_f32(split);
    let variance = returns[..split]
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f32>()
        / usize_to_f32(split);
    let deviation = variance.sqrt();
    if !deviation.is_finite() || deviation <= f32::EPSILON {
        return Err("training returns must have nonzero finite variance".into());
    }
    let normalized = returns
        .iter()
        .map(|value| (value - mean) / deviation)
        .collect::<Vec<_>>();

    // Keep all training targets before the split and validation targets after it.
    let training = windows(&normalized[..split], inputs, outputs);
    let validation = windows(&normalized[split - inputs..], inputs, outputs);

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
        "inputs={}\noutputs={}\nreturn_mean={}\nreturn_deviation={}\n",
        args.inputs,
        args.outputs,
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
        let cli = Cli::try_parse_from(["stockholm", "train", "prices.txt"]).unwrap();

        let Some(Subcommand::Train(args)) = cli.command else {
            panic!("expected train subcommand");
        };
        assert_eq!(args.path, PathBuf::from("prices.txt"));
        assert_eq!(args.inputs, 20);
        assert_eq!(args.outputs, 5);
        assert_eq!(args.model_directory, PathBuf::from("model"));
    }

    #[test]
    fn parse_price_lines() {
        // Confirm whitespace and blank lines do not alter valid observations.
        let prices = parse_prices("100\n 101.5 \n\n99\n").unwrap();

        assert_eq!(prices, vec![100.0, 101.5, 99.0]);
    }

    #[test]
    fn reject_nonpositive_prices() {
        // Confirm logarithmic preprocessing rejects values outside its domain.
        let error = parse_prices("100\n0\n").unwrap_err();

        assert!(error.to_string().contains("finite and positive"));
    }

    #[test]
    fn create_chronological_windows() {
        // Confirm validation targets begin strictly after the training boundary.
        let prices = (1_u16..=101).map(f32::from).collect::<Vec<_>>();
        let returns = log_returns(&prices);
        let data = prepare_data(&returns, 4, 2).unwrap();

        assert_eq!(data.training.len(), 75);
        assert_eq!(data.validation.len(), 19);
        assert!(baseline_loss(&data.validation, data.mean, data.deviation).is_finite());
    }
}
