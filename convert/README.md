# LC0 Training Data to Apache Beam Converter

This package provides utilities for converting Leela Chess Zero (LC0) training data in V6, V7, and V7B formats to structures compatible with Apache Beam.

## Features

- Read LC0 training chunk files (.gz) in V6, V7, and V7B formats
- Convert training records to dictionaries
- Encode dictionaries back to binary format
- Process records in parallel using Apache Beam
- Export data in TFRecord, JSON, or text formats

## Installation

### Requirements

- Python 3.7+
- numpy
- apache-beam
- tensorflow (optional, for TFRecord support)

### Setup

1. Clone this repository or copy the `convert` directory to your project.
2. Install the required dependencies:

```bash
pip install numpy apache-beam
# Optional for TFRecord support
pip install tensorflow
```

## Usage

### Basic Conversion

```bash
# Convert a single file
python convert.py --input path/to/chunk.gz --output path/to/output.tfrecord

# Convert a directory of files
python convert.py --input_dir path/to/chunks --output_dir path/to/output --pattern "*.gz"

# Use Apache Beam for distributed processing
python convert.py --input_dir path/to/chunks --output path/to/output.tfrecord --use_beam
```

### Output Formats

The converter supports the following output formats:

- `tfrecord` (default): TensorFlow TFRecord format
- `json`: JSON format (one record per line)
- `text`: Plain text format

```bash
# Convert to JSON format
python convert.py --input path/to/chunk.gz --output path/to/output.json --output_format json

# Convert to text format
python convert.py --input path/to/chunk.gz --output path/to/output.txt --output_format text
```

### V7 Fields

By default, V7/V7B-specific fields are excluded from the output. To include them:

```bash
python convert.py --input path/to/chunk.gz --output path/to/output.tfrecord --v7_fields
```

### Sampling

To reduce the size of the output, you can sample the records:

```bash
# Sample 10% of the records
python convert.py --input path/to/chunk.gz --output path/to/output.tfrecord --sample_rate 0.1
```

### Advanced Options

For more options:

```bash
python convert.py --help
```

## Examples

See `example.py` for usage examples:

```bash
# Simple example
python example.py --input path/to/chunk.gz --example simple

# Batch processing example
python example.py --input_dir path/to/chunks --example batch

# Encoding example
python example.py --input path/to/chunk.gz --example encode
```

## API Usage

### Reading Training Records

```python
from convert.chunk_parser import read_chunks

# Read records from a file
for record in read_chunks("path/to/chunk.gz"):
    # Process the binary record
    pass
```

### Decoding Records

```python
from convert.lc0_coder import LC0TrainingDataCoder

# Create a coder
coder = LC0TrainingDataCoder(support_v7=False)

# Decode a record
record_dict = coder.decode(record_bytes)

# Access fields
version = record_dict['version']
best_q = record_dict['best_q']
probabilities = record_dict['probabilities']
```

### Encoding Records

```python
from convert.lc0_coder import LC0TrainingDataCoder

# Create a coder
coder = LC0TrainingDataCoder()

# Create or modify a dictionary
record_dict = {
    'version': 6,
    'input_format': 1,
    # ... other required fields
}

# Encode the record
encoded_bytes = coder.encode(record_dict)
```

### Using with Apache Beam

```python
import apache_beam as beam
from convert.chunk_parser import ParseChunksToRecords
from convert.lc0_coder import LC0TrainingDataCoder

with beam.Pipeline() as pipeline:
    records = (
        pipeline
        | "Create file paths" >> beam.Create(["path/to/chunk.gz"])
        | "Read files" >> beam.io.ReadFromText()
        | "Parse records" >> beam.ParDo(ParseChunksToRecords())
        | "Decode records" >> beam.Map(lambda x: LC0TrainingDataCoder().decode(x))
        # Further processing...
    )
```

## Field Reference

### V6 Fields

- `version`: Format version (6)
- `input_format`: Input format (1)
- `probabilities`: Policy vector
- `planes`: Board state planes
- `castling_us_ooo`, `castling_us_oo`, `castling_them_ooo`, `castling_them_oo`: Castling rights
- `side_to_move`: Side to move
- `rule50_count`: Rule 50 counter
- `invariance_info`: Invariance information
- `result`: Game result
- `root_q`, `best_q`: Q value at root and best move
- `root_d`, `best_d`: Draw probability at root and best move
- `root_m`, `best_m`: MLH at root and best move
- `plies_left`: Actual plies left in game

### V7 Additional Fields

When `--v7_fields` is enabled:

- `v7_result_q`, `v7_result_d`: Game result Q and D values
- `v7_played_q`, `v7_played_d`, `v7_played_m`: Q, D, M values of played move
- `v7_orig_q`, `v7_orig_d`, `v7_orig_m`: Original neural network outputs
- `v7_visits`: Number of visits
- `v7_played_idx`, `v7_best_idx`: Move indices
- `v7_pol_kld`: Policy KL divergence
- `v7_q_st`, `v7_d_st`: Q and D state values
- `v7_opp_played_idx`, `v7_next_played_idx`: Opponent and next played move indices

### V7B Additional Fields

When `--v7_fields` is enabled:

- `v7b_opp_probs`: Opponent move probabilities
- `v7b_next_probs`: Next move probabilities
- `v7b_fut`: Future board states

## License

This project is licensed under the GPL v3.0 License - see the LICENSE file for details. 