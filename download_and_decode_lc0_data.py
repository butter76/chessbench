#!/usr/bin/env python3
"""
Leela Chess Zero Training Data Downloader and Decoder

This script downloads Leela Chess Zero training data .bag files from a Wasabi/S3 bucket,
and decodes them using the decode_lc0_data function.
"""

import os
import sys
import argparse
import logging
import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm
import csv

# Import custom modules
from searchless_chess.src.bagz import BagReader
from searchless_chess.src.constants import decode_lc0_data, LC0DataRecord


def setup_logging(verbose=False):
    """Set up logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_s3_client(access_key, secret_key, region="us-west-1", endpoint_url=None):
    """
    Create and return an S3 client configured for Wasabi
    
    Args:
        access_key: S3/Wasabi access key
        secret_key: S3/Wasabi secret key
        region: S3/Wasabi region (e.g., 'us-east-1')
        endpoint_url: Optional custom endpoint URL for Wasabi
        
    Returns:
        boto3 S3 client
    """
    if endpoint_url is None:
        # Default Wasabi endpoint format
        endpoint_url = f'https://s3.{region}.wasabisys.com'
    
    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )
        return s3_client
    except Exception as e:
        logging.error(f"Failed to create S3 client: {e}")
        raise


def list_bag_files(s3_client, bucket_name, prefix=''):
    """
    List all .bag files in the specified bucket with optional prefix
    
    Args:
        s3_client: boto3 S3 client
        bucket_name: Name of the bucket to list files from
        prefix: Optional prefix to filter files (e.g., 'lc0_data_20230701')
        
    Returns:
        List of object keys for .bag files
    """
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        bag_files = []
        
        # Use paginator to handle buckets with many files
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    if obj['Key'].endswith('.bag'):
                        bag_files.append(obj['Key'])
        
        return bag_files
    except ClientError as e:
        logging.error(f"Error listing files in bucket {bucket_name}: {e}")
        raise


def download_bag_file(s3_client, bucket_name, object_key, output_dir):
    """
    Download a .bag file from S3/Wasabi
    
    Args:
        s3_client: boto3 S3 client
        bucket_name: Name of the bucket
        object_key: Key of the object to download
        output_dir: Directory to save the downloaded file
        
    Returns:
        Path to the downloaded file
    """
    local_path = os.path.join(output_dir, os.path.basename(object_key))
    
    # Skip if file already exists
    if os.path.exists(local_path):
        logging.info(f"File already exists: {local_path}")
        return local_path
    
    try:
        # Get file size for progress tracking
        response = s3_client.head_object(Bucket=bucket_name, Key=object_key)
        file_size = response['ContentLength']
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Download with progress bar
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Downloading {object_key}") as pbar:
            # Define a callback for progress updates
            def download_progress(bytes_amount):
                pbar.update(bytes_amount)
            
            # Download the file
            s3_client.download_file(
                bucket_name, 
                object_key, 
                local_path, 
                Callback=download_progress
            )
        
        logging.info(f"Downloaded {object_key} to {local_path}")
        return local_path
    except ClientError as e:
        logging.error(f"Error downloading {object_key}: {e}")
        if os.path.exists(local_path):
            os.remove(local_path)
        raise

def decode_bag_file(bag_path, max_records=None, print_sample=False):
    """
    Decode a .bag file containing LC0 data records
    
    Args:
        bag_path: Path to the .bag file
        max_records: Maximum number of records to process (None for all)
        print_sample: Whether to print sample records
        
    Returns:
        Number of records processed
    """
    try:
        # Open the bag file
        reader = BagReader(bag_path)
        record_count = len(reader)
        logging.info(f"Bag file contains {record_count} records")
        
        # Limit records if specified
        if max_records is not None and max_records < record_count:
            record_count = max_records
            logging.info(f"Processing first {max_records} records")
        
        # Process records with progress bar
        processed = 0
        games = 0
        deduped = 0
        dist = {'-1.0': 0, '0.0': 0, '1.0': 0}
        prev_records = []
        dedup = set()
        with tqdm(total=record_count, desc="Decoding records") as pbar:
            for i in range(record_count):
                # Get and decode the record
                record_bytes = reader[i]
                record = decode_lc0_data(record_bytes)
                
                if record.fen == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1":
                    games += 1
                    if len(prev_records) > 0:
                        last_record = prev_records[-1]
                        if last_record.root_q > 0.5:
                            outcome = 1
                        elif last_record.root_q < -0.5:
                            outcome = -1
                        else:
                            outcome = 0
                        outcome *= (-1) ** (last_record.plies_left % 2)
                        for prev_record in prev_records:
                            new_record = prev_record._replace(result=outcome * (-1) ** (prev_record.plies_left % 2))
                            
                        prev_records = []
                    
                
                b = record.fen.split(' ')[0]
                if b not in dedup:
                    deduped += 1
                    dedup.add(b)
                    prev_records.append(record)
                
                processed += 1
                pbar.update(1)
        
        return processed, games, dist, deduped
    
    except Exception as e:
        logging.error(f"Error decoding bag file {bag_path}: {e}")
        raise


def print_record(record: LC0DataRecord, index: int):
    """
    Pretty print a LC0DataRecord
    
    Args:
        record: The record to print
        index: Record index for display
    """
    print(f"\n=== Record {index} ===")
    print(f"FEN: {record.fen}")
    print(f"Move: {record.move}")
    print(f"Result: {record.result}")
    print(f"Root Q/D: {record.root_q:.4f}/{record.root_d:.4f}")
    print(f"Played Q/D: {record.played_q:.4f}/{record.played_d:.4f}")
    print(f"Plies left: {record.plies_left}")
    print(f"Policy (top 5 of {len(record.policy)}):")
    
    # Sort policy by probability (descending) and print top 5
    sorted_policy = sorted(record.policy, key=lambda x: x[1], reverse=True)
    for move, prob in sorted_policy[:5]:
        print(f"  {move}: {prob:.6f}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Download and decode LC0 data from Wasabi/S3')
    
    # S3/Wasabi configuration
    parser.add_argument('--access-key', required=True, help='S3/Wasabi access key')
    parser.add_argument('--secret-key', required=True, help='S3/Wasabi secret key')
    parser.add_argument('--region', default='us-west-1', help='S3/Wasabi region (default: us-east-1)')
    parser.add_argument('--endpoint-url', help='Custom S3 endpoint URL (optional, default: Wasabi endpoint for region)')
    parser.add_argument('--bucket', required=True, help='S3/Wasabi bucket name')
    
    # File selection
    parser.add_argument('--prefix', default='', help='Prefix for listing files (optional)')
    parser.add_argument('--key', help='Specific object key to download (optional, if not specified will list available files)')
    
    # Output options
    parser.add_argument('--output-dir', default='downloaded_data', help='Directory for downloaded files')
    parser.add_argument('--csv', action='store_true', help='Output decoded data to CSV')
    parser.add_argument('--max-records', type=int, help='Maximum number of records to process')
    parser.add_argument('--print-samples', action='store_true', help='Print sample records')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)
    
    try:
        # Create S3 client
        s3_client = create_s3_client(
            args.access_key, 
            args.secret_key, 
            args.region,
            args.endpoint_url
        )
        
        # Ensure output directory exists
        os.makedirs(args.output_dir, exist_ok=True)
        
        # If specific key provided, download and process it
        if args.key:
            logging.info(f"Downloading specific file: {args.key}")
            bag_path = download_bag_file(s3_client, args.bucket, args.key, args.output_dir)
            
            # Define CSV output path if requested
            csv_path = None
            if args.csv:
                csv_name = os.path.splitext(os.path.basename(args.key))[0] + '.csv'
                csv_path = os.path.join(args.output_dir, csv_name)
            
            # Decode the bag file
            record_count, game_count, dist, deduped = decode_bag_file(
                bag_path, 
                max_records=args.max_records,
                print_sample=args.print_samples
            )
            
            print(f"Successfully processed {record_count} records from {args.key} ({game_count} games), average {record_count / game_count} records per game")
            print(f"Result distribution: {dist}")
            print(f"Deduped {deduped} records, {deduped / record_count * 100:.2f}% of total")
        # If no key provided, list available files
        else:
            logging.info(f"Listing .bag files in bucket {args.bucket} with prefix '{args.prefix}'")
            bag_files = list_bag_files(s3_client, args.bucket, args.prefix)
            
            if not bag_files:
                logging.info(f"No .bag files found in bucket {args.bucket} with prefix '{args.prefix}'")
                return
            
            print(f"\nFound {len(bag_files)} .bag files in bucket {args.bucket}:")
            for i, key in enumerate(bag_files):
                print(f"{i+1}. {key}")
            
            print("\nTo download and decode a specific file, run:")
            print(f"python {sys.argv[0]} --access-key YOUR_KEY --secret-key YOUR_SECRET --bucket {args.bucket} --key FILE_KEY")
    
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)
    
    logging.info("Done!")


if __name__ == "__main__":
    main() 