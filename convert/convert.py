#!/usr/bin/env python3
"""
LC0 Training Data Converter

This script converts LC0 training data (V6) to Apache Beam compatible format.
It can process individual files or directories of chunk files.

Example usage:
    # Convert a single file
    python convert.py --input path/to/chunk.gz --output path/to/output.tfrecord
    
    # Convert a directory of files
    python convert.py --input_dir path/to/chunks --output_dir path/to/output --pattern "*.gz" 
    
    # Use Apache Beam for distributed processing
    python convert.py --input_dir path/to/chunks --output path/to/output.tfrecord --use_beam
"""

import os
import sys
import argparse
import logging
from typing import List, Dict, Any, Optional, Union, Tuple

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions
from apache_beam.io import filesystems

from lc0_coder import LC0TrainingDataCoder
from chunk_parser import ParseChunksToRecords, get_chunk_files, read_chunks


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Convert LC0 training data to Apache Beam format')
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', type=str, help='Input chunk file')
    input_group.add_argument('--input_dir', type=str, help='Directory containing input chunk files')
    
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument('--output', type=str, help='Output file path')
    output_group.add_argument('--output_dir', type=str, help='Directory for output files')
    
    parser.add_argument('--pattern', type=str, default='*.gz', 
                        help='File pattern for input files when using --input_dir')
    parser.add_argument('--recursive', action='store_true',
                        help='Search input directory recursively')
    parser.add_argument('--sample_rate', type=float, default=1.0,
                        help='Sample rate for positions (0.0 to 1.0)')
    parser.add_argument('--max_files', type=int, default=None,
                        help='Maximum number of files to process')
    parser.add_argument('--use_beam', action='store_true',
                        help='Use Apache Beam for distributed processing')
    parser.add_argument('--output_format', type=str, choices=['tfrecord', 'text', 'json'], 
                        default='tfrecord', help='Output format')
    
    # Apache Beam specific options
    parser.add_argument('--runner', type=str, default='DirectRunner',
                        help='Apache Beam runner to use')
    parser.add_argument('--temp_location', type=str, default=None,
                        help='Temporary location for Apache Beam')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of workers for Apache Beam')
    
    return parser.parse_args()


def get_input_files(args) -> List[str]:
    """Get list of input files based on command line arguments."""
    if args.input:
        return [args.input]
    elif args.input_dir:
        files = get_chunk_files(args.input_dir, args.pattern, args.recursive)
        if not files:
            raise ValueError(f"No files matching pattern '{args.pattern}' found in {args.input_dir}")
        
        # Sort for deterministic behavior
        files.sort()
        
        # Limit number of files if specified
        if args.max_files:
            files = files[:args.max_files]
        
        return files
    else:
        raise ValueError("Either --input or --input_dir must be specified")


def convert_record(record: bytes, include_v7_fields: bool = False) -> Dict[str, Any]:
    """Convert a binary training record to a dictionary.
    
    Args:
        record: Binary training record
        include_v7_fields: Whether to include V7-specific fields
        
    Returns:
        Dictionary representation of the record
    """
    coder = LC0TrainingDataCoder(support_v7=False)
    decoded = coder.decode(record)
    
    # V7 fields will not be included since support_v7 is False
    return decoded


class ProcessChunk(beam.DoFn):
    """Process chunks and convert records using Apache Beam."""
    
    def __init__(self, include_v7_fields: bool = False, sample_rate: float = 1.0):
        self.include_v7_fields = include_v7_fields
        self.sample_rate = sample_rate
        self.coder = LC0TrainingDataCoder(support_v7=False)
    
    def process(self, element):
        """Process chunk and yield converted records."""
        # Parse the chunk
        parser = ParseChunksToRecords(sample_rate=self.sample_rate)
        
        # For each record in the chunk
        for record in parser.process(element):
            # Decode and yield
            yield self.coder.decode(record)


def write_records_direct(records, output_path: str, output_format: str):
    """Write records directly without using Apache Beam.
    
    Args:
        records: Iterable of training records
        output_path: Path to write records to
        output_format: Output format (tfrecord, json, text)
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    if output_format == 'tfrecord':
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("tensorflow is required for tfrecord output. Install it with pip install tensorflow.")
        
        with tf.io.TFRecordWriter(output_path) as writer:
            for record in records:
                # Convert dictionary to tf.train.Example
                features = {}
                for key, value in record.items():
                    if isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], (int, float)):
                        # For numeric arrays
                        features[key] = tf.train.Feature(
                            float_list=tf.train.FloatList(value=value)
                        )
                    elif isinstance(value, (int, bool)):
                        features[key] = tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[value])
                        )
                    elif isinstance(value, float):
                        features[key] = tf.train.Feature(
                            float_list=tf.train.FloatList(value=[value])
                        )
                    elif isinstance(value, bytes):
                        features[key] = tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[value])
                        )
                    elif isinstance(value, str):
                        features[key] = tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=[value.encode('utf-8')])
                        )
                    elif isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], bytes):
                        # For byte arrays
                        features[key] = tf.train.Feature(
                            bytes_list=tf.train.BytesList(value=value)
                        )
                
                example = tf.train.Example(
                    features=tf.train.Features(feature=features)
                )
                writer.write(example.SerializeToString())
    
    elif output_format == 'json':
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        def prepare_for_json(record):
            import numpy as np
            result = {}
            for key, value in record.items():
                if isinstance(value, np.ndarray):
                    result[key] = value.tolist()
                elif isinstance(value, bytes):
                    # Convert bytes to base64
                    import base64
                    result[key] = base64.b64encode(value).decode('utf-8')
                else:
                    result[key] = value
            return result
        
        with open(output_path, 'w') as f:
            for record in records:
                f.write(json.dumps(prepare_for_json(record)) + '\n')
    
    elif output_format == 'text':
        with open(output_path, 'w') as f:
            for record in records:
                # Simple text format - just key-value pairs
                f.write(str(record) + '\n')
    
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def run_direct_conversion(args):
    """Run direct conversion without Apache Beam.
    
    Args:
        args: Command line arguments
    """
    logging.info("Running direct conversion...")
    
    input_files = get_input_files(args)
    logging.info(f"Processing {len(input_files)} files...")
    
    if args.output_dir:
        # Process each file separately
        for input_file in input_files:
            basename = os.path.basename(input_file)
            basename = os.path.splitext(basename)[0]  # Remove extension
            if basename.endswith('.gz'):
                basename = os.path.splitext(basename)[0]  # Remove .gz extension if present
            
            output_path = os.path.join(args.output_dir, f"{basename}.{args.output_format}")
            
            logging.info(f"Converting {input_file} to {output_path}...")
            
            # Read records from the file
            records = (convert_record(record, False) 
                      for record in read_chunks(input_file))
            
            # Write records
            write_records_direct(records, output_path, args.output_format)
    else:
        # Process all files and write to a single output
        logging.info(f"Converting {len(input_files)} files to {args.output}...")
        
        # Read records from all files
        def read_all_records():
            for input_file in input_files:
                yield from (convert_record(record, False) 
                           for record in read_chunks(input_file))
        
        # Write records
        write_records_direct(read_all_records(), args.output, args.output_format)


def run_beam_conversion(args):
    """Run conversion using Apache Beam.
    
    Args:
        args: Command line arguments
    """
    logging.info("Running Apache Beam conversion...")
    
    # Create pipeline options
    options = PipelineOptions()
    options.view_as(SetupOptions).save_main_session = True
    
    # Set runner
    if args.runner:
        options = PipelineOptions(['--runner', args.runner])
    
    # Set temp location if provided
    if args.temp_location:
        options.view_as(SetupOptions).temp_location = args.temp_location
    
    # Set number of workers if provided
    if args.num_workers:
        options = PipelineOptions(['--direct_num_workers', str(args.num_workers)])
    
    # Get input files
    input_files = get_input_files(args)
    logging.info(f"Processing {len(input_files)} files...")
    
    # Run pipeline
    with beam.Pipeline(options=options) as pipeline:
        # Read input files
        records = (
            pipeline
            | "Create file paths" >> beam.Create(input_files)
            | "Read files" >> beam.FlatMap(lambda filepath: read_chunks(filepath))
            | "Parse records" >> beam.ParDo(ProcessChunk(include_v7_fields=False, 
                                                       sample_rate=args.sample_rate))
        )
        
        if args.output_format == 'tfrecord':
            try:
                from apache_beam.io.tfrecordio import WriteToTFRecord
                from apache_beam.ml.transforms.tft_only import convert_to_example
            except ImportError:
                raise ImportError("apache_beam.io.tfrecordio is required for tfrecord output.")
            
            # Convert to TF Examples and write
            (records
             | "To TF Examples" >> beam.Map(lambda record: convert_to_example(record))
             | "Write TFRecord" >> WriteToTFRecord(args.output)
            )
        
        elif args.output_format == 'json':
            import json
            
            def to_json(record):
                import numpy as np
                # Convert numpy arrays to lists for JSON serialization
                result = {}
                for key, value in record.items():
                    if isinstance(value, np.ndarray):
                        result[key] = value.tolist()
                    elif isinstance(value, bytes):
                        # Convert bytes to base64
                        import base64
                        result[key] = base64.b64encode(value).decode('utf-8')
                    else:
                        result[key] = value
                return json.dumps(result)
            
            (records
             | "To JSON" >> beam.Map(to_json)
             | "Write JSON" >> beam.io.WriteToText(args.output, file_name_suffix='.json')
            )
        
        else:  # text format
            (records
             | "To string" >> beam.Map(str)
             | "Write text" >> beam.io.WriteToText(args.output, file_name_suffix='.txt')
            )


def main():
    """Main entry point."""
    setup_logging()
    args = parse_args()
    
    # Validate arguments
    if args.sample_rate < 0.0 or args.sample_rate > 1.0:
        raise ValueError("Sample rate must be between 0.0 and 1.0")
    
    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Run conversion
    if args.use_beam:
        run_beam_conversion(args)
    else:
        run_direct_conversion(args)


if __name__ == "__main__":
    main() 