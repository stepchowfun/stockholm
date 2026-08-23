#!/usr/bin/env ruby

# Load only standard-library tools so plotting requires no extra gems.
require "csv"
require "fileutils"
require "optparse"
require "tempfile"
require "time"

# Keep generated labels in the same time zone as the market-hours data.
ENV["TZ"] = "America/New_York"

# Resolve all default paths relative to the repository containing this script.
ROOT = File.expand_path(__dir__)
DEFAULT_INPUTS = File.join(ROOT, "historical_data", "validation", "*.csv")
DEFAULT_MODEL_DIRECTORY = File.join(ROOT, "model")
DEFAULT_OUTPUT_DIRECTORY = File.join(ROOT, "inference")

# Reserve fixed margins for two vertical axes and the time labels.
LEFT = 82.0
RIGHT = 1508.0
TOP = 58.0
BOTTOM = 638.0

# Parse optional destinations while treating positional arguments as input files.
options = {
  model_directory: DEFAULT_MODEL_DIRECTORY,
  output_directory: DEFAULT_OUTPUT_DIRECTORY,
}
OptionParser.new do |parser|
  parser.banner = "Usage: ./plot-inference.rb [options] [CSV ...]"
  parser.on("--model-directory PATH", "Model directory (default: model)") do |path|
    options[:model_directory] = File.expand_path(path)
  end
  parser.on("--output-directory PATH", "SVG directory (default: inference)") do |path|
    options[:output_directory] = File.expand_path(path)
  end
end.parse!

# Plot the complete validation set unless explicit CSV paths were supplied.
input_paths = ARGV.empty? ? Dir[DEFAULT_INPUTS].sort : ARGV.map { |path| File.expand_path(path) }
abort "No input CSV files were found." if input_paths.empty?
FileUtils.mkdir_p(options[:output_directory])

# Generate one inference CSV and SVG overlay for each historical data file.
input_paths.each do |input_path|
  name = File.basename(input_path, ".csv")
  output_path = File.join(options[:output_directory], "#{name}.svg")

  # Keep the bulky probability data only until its SVG has been rendered.
  Tempfile.create(["stockholm-inference-", ".csv"]) do |predictions|
    predictions.close
    command = [
      "cargo",
      "run",
      "--release",
      "--",
      "infer",
      "--model-directory",
      options[:model_directory],
      "--input-path",
      input_path,
      "--output-path",
      predictions.path,
    ]
    abort "Inference failed for #{input_path}." unless system(*command, chdir: ROOT)

    # Match each prediction to the open price at the same timestamp.
    prices = CSV.foreach(input_path, headers: true).each_with_object({}) do |row, values|
      values[row.fetch("date").to_i] = row.fetch("open").to_f
    end
    rows = []
    CSV.foreach(predictions.path, headers: true) do |row|
      timestamp = row.fetch("timestamp").to_i
      price = prices[timestamp]
      next unless price

      difference = row.fetch("upper_probability").to_f - row.fetch("lower_probability").to_f
      rows << [timestamp, price, difference]
    end
    abort "Inference produced no timestamped prices for #{input_path}." if rows.empty?

    # Scale prices independently while centering the signed probability axis on zero.
    timestamps = rows.map(&:first)
    prices = rows.map { |row| row[1] }
    differences = rows.map { |row| row[2] }
    price_min, price_max = prices.minmax
    price_padding = [(price_max - price_min) * 0.05, 0.01].max
    price_min -= price_padding
    price_max += price_padding
    difference_extent = [differences.map(&:abs).max, 0.01].max

    # Convert observations into coordinates within the common plot area.
    x = lambda do |timestamp|
      LEFT + (timestamp - timestamps.first).fdiv(timestamps.last - timestamps.first) * (RIGHT - LEFT)
    end
    price_y = lambda do |price|
      BOTTOM - (price - price_min).fdiv(price_max - price_min) * (BOTTOM - TOP)
    end
    difference_y = lambda do |difference|
      BOTTOM - (difference + difference_extent).fdiv(2.0 * difference_extent) * (BOTTOM - TOP)
    end
    path = lambda do |values, y|
      rows.each_index.map do |index|
        format("%s %.2f %.2f", index.zero? ? "M" : "L", x.call(timestamps[index]), y.call(values[index]))
      end.join(" ")
    end

    # Draw axes, grid lines, and labels directly as a portable SVG.
    svg = String.new
    svg << %(<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="700" viewBox="0 0 1600 700"><rect width="100%" height="100%" fill="white"/>)
    svg << %(<text x="800" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">SOXL price and directional probability difference — #{name.sub("SOXL-", "")}</text>\n)
    6.times do |index|
      ratio = index / 5.0
      y = BOTTOM - ratio * (BOTTOM - TOP)
      price = price_min + ratio * (price_max - price_min)
      svg << %(<line x1="#{LEFT}" y1="#{y}" x2="#{RIGHT}" y2="#{y}" stroke="#e5e7eb" stroke-width=".6"/><text x="72" y="#{y + 4}" text-anchor="end" font-family="sans-serif" font-size="12">#{format("%.2f", price)}</text>\n)
    end
    5.times do |index|
      ratio = index / 4.0
      y = BOTTOM - ratio * (BOTTOM - TOP)
      difference = -difference_extent + ratio * 2.0 * difference_extent
      svg << %(<text x="1518" y="#{y + 4}" font-family="sans-serif" font-size="12" fill="#2563eb">#{format("%.2f", difference)}</text>\n)
    end
    7.times do |index|
      ratio = index / 6.0
      timestamp = timestamps.first + ((timestamps.last - timestamps.first) * ratio).round
      x_coordinate = LEFT + ratio * (RIGHT - LEFT)
      label = Time.at(timestamp).strftime("%-I:%M %p")
      svg << %(<line x1="#{x_coordinate}" y1="#{TOP}" x2="#{x_coordinate}" y2="#{BOTTOM}" stroke="#f3f4f6" stroke-width=".6"/><text x="#{x_coordinate}" y="661" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#374151">#{label}</text>\n)
    end

    # Overlay thin price and directional lines so fine details remain visible when zoomed.
    zero_y = difference_y.call(0.0)
    svg << %(<line x1="#{LEFT}" y1="#{zero_y}" x2="#{RIGHT}" y2="#{zero_y}" stroke="#6b7280" stroke-width=".7" stroke-dasharray="5 4"/>\n)
    svg << %(<path d="#{path.call(prices, price_y)}" fill="none" stroke="#111827" stroke-width=".4"/>\n)
    svg << %(<path d="#{path.call(differences, difference_y)}" fill="none" stroke="#2563eb" stroke-width=".4"/>\n)
    svg << %(<rect x="#{LEFT}" y="#{TOP}" width="#{RIGHT - LEFT}" height="#{BOTTOM - TOP}" fill="none" stroke="#9ca3af" stroke-width=".8"/>)
    svg << %(<text x="20" y="348" transform="rotate(-90 20 348)" text-anchor="middle" font-family="sans-serif" font-size="13">Open price</text>)
    svg << %(<text x="1582" y="348" transform="rotate(90 1582 348)" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#2563eb">Upper − lower probability</text>)
    svg << %(<line x1="94" y1="44" x2="124" y2="44" stroke="#111827" stroke-width=".4"/><text x="130" y="48" font-family="sans-serif" font-size="12">Open price</text>)
    svg << %(<line x1="227" y1="44" x2="257" y2="44" stroke="#2563eb" stroke-width=".4"/><text x="263" y="48" font-family="sans-serif" font-size="12">Upper − lower</text>)
    svg << %(<line x1="378" y1="44" x2="408" y2="44" stroke="#6b7280" stroke-dasharray="5 4"/><text x="414" y="48" font-family="sans-serif" font-size="12">Zero</text></svg>)

    # Publish the completed chart at its stable output path.
    File.write(output_path, svg)
    puts "Saved plot to #{output_path}."
  end
end
