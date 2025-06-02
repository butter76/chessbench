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
import threading
import queue
import time

# Import custom modules
from searchless_chess.src.bagz import BagReader, BagWriter
from searchless_chess.src.constants import decode_lc0_data, LC0DataRecord, encode_lc0_data


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


def download_worker(s3_client, bucket_name, bag_files, buffer_dir, file_queue, max_buffer_size=10):
    """
    Background worker to download bag files while maintaining a buffer
    
    Args:
        s3_client: boto3 S3 client
        bucket_name: Name of the bucket
        bag_files: List of bag file keys to download
        buffer_dir: Directory to store downloaded files
        file_queue: Queue to put downloaded filenames
        max_buffer_size: Maximum number of files to keep in buffer
    """
    try:
        for bag_file in bag_files:
            # Wait until buffer has space
            while True:
                buffer_files = [f for f in os.listdir(buffer_dir) if f.endswith('.bag')]
                if len(buffer_files) < max_buffer_size:
                    break
                time.sleep(1)  # Wait 1 second before checking again
            
            # Download the file
            try:
                local_path = download_bag_file(s3_client, bucket_name, bag_file, buffer_dir)
                file_queue.put(local_path)
                logging.info(f"Downloaded and queued: {bag_file}")
            except Exception as e:
                logging.error(f"Failed to download {bag_file}: {e}")
                continue
        
        # Signal completion by putting None in queue
        file_queue.put(None)
        logging.info("Download worker completed")
        
    except Exception as e:
        logging.error(f"Download worker error: {e}")
        file_queue.put(None)


def upload_bag_file(s3_client, bucket_name, local_path, object_key):
    """
    Upload a processed .bag file to S3/Wasabi
    
    Args:
        s3_client: boto3 S3 client
        bucket_name: Name of the bucket
        local_path: Path to the local file to upload
        object_key: Key to use for the uploaded object
    """
    try:
        # Get file size for progress tracking
        file_size = os.path.getsize(local_path)
        
        # Upload with progress bar
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Uploading {object_key}") as pbar:
            # Define a callback for progress updates
            def upload_progress(bytes_amount):
                pbar.update(bytes_amount)
            
            # Upload the file
            s3_client.upload_file(
                local_path,
                bucket_name, 
                object_key,
                Callback=upload_progress
            )
        
        logging.info(f"Uploaded {local_path} to {object_key}")
        
    except ClientError as e:
        logging.error(f"Error uploading {local_path}: {e}")
        raise


def decode_bag_file(bag_path, output_bag, dedup, print_sample=False):
    """
    Decode a .bag file containing LC0 data records
    
    Args:
        bag_path: Path to the .bag file
        print_sample: Whether to print sample records
        
    Returns:
        Number of records processed
    """
    try:
        # Open the bag file
        reader = BagReader(bag_path)
        record_count = len(reader)
        
        with BagWriter(output_bag) as writer:
            # Process records with progress bar
            processed = 0
            games = 0
            deduped = 0
            prev_records = []
            count = 0
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
                                count += 1
                                if count % 3 != 0:
                                    continue
                                new_record = prev_record._replace(result=outcome * (-1) ** (prev_record.plies_left % 2))
                                writer.write(encode_lc0_data(new_record))
                            prev_records = []
                    
                    b = record.fen.split(' ')[0]
                    if b not in dedup:
                        deduped += 1
                        dedup.add(b)
                        prev_records.append(record)
                    
                    processed += 1
                    if i % 1000 == 0:
                        pbar.update(1000)
    
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
    parser.add_argument('--bucket', required=True, help='S3/Wasabi bucket name for downloading')
    parser.add_argument('--output-bucket', required=True, help='S3/Wasabi bucket name for uploading processed files (optional, defaults to same as --bucket)')
    
    # File selection
    parser.add_argument('--prefix', default='', help='Prefix for listing files (optional)')
    parser.add_argument('--key', help='Specific object key to download (optional, if not specified will list available files)')
    
    # Output options
    parser.add_argument('--output-dir', default='downloaded_data', help='Directory for downloaded files')
    parser.add_argument('--output-bag', default='processed_data', help='Directory for processed bag files')
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
        os.makedirs(args.output_bag, exist_ok=True)
        
        # Create buffer directory for downloads
        buffer_dir = os.path.join(args.output_dir, 'buffer')
        os.makedirs(buffer_dir, exist_ok=True)
        
        # Global dedup set shared across all processing
        global_dedup = set()
        
        # Set output bucket (default to same as input bucket)
        output_bucket = args.output_bucket if args.output_bucket else args.bucket
        
        # If specific key provided, download and process it
        if args.key:
            logging.info(f"Downloading and processing specific file: {args.key}")
            bag_path = download_bag_file(s3_client, args.bucket, args.key, args.output_dir)
            output_bag = os.path.join(args.output_bag, f"processed_{os.path.basename(args.key)}")
            
            # Decode the bag file
            decode_bag_file(
                bag_path, 
                output_bag,
                dedup=global_dedup,
            )
            
            # Upload processed file to S3
            upload_key = f"processed_{args.key}"
            upload_bag_file(s3_client, output_bucket, output_bag, upload_key)
            
        # If no key provided, process all files with streaming download
        else:
            logging.info(f"Listing .bag files in bucket {args.bucket} with prefix '{args.prefix}'")
            bag_files = list_bag_files(s3_client, args.bucket, args.prefix)
            
            if not bag_files:
                logging.info(f"No .bag files found in bucket {args.bucket} with prefix '{args.prefix}'")
                return
            
            logging.info(f"Found {len(bag_files)} .bag files to process")
            
            # Set up queue for communication between threads
            file_queue = queue.Queue()
            
            # Start download worker thread
            download_thread = threading.Thread(
                target=download_worker,
                args=(s3_client, args.bucket, bag_files, buffer_dir, file_queue),
                daemon=True
            )
            download_thread.start()
            
            # Process files as they become available
            processed_count = 0
            while True:
                # Get next downloaded file
                bag_path = file_queue.get()
                
                # None signals completion
                if bag_path is None:
                    break
                
                try:
                    # Process the bag file
                    output_bag = os.path.join(args.output_bag, f"processed_{os.path.basename(bag_path)}")
                    logging.info(f"Processing {bag_path}")
                    
                    decode_bag_file(
                        bag_path,
                        output_bag,
                        dedup=global_dedup
                    )
                    
                    # Upload processed file to S3
                    original_key = os.path.relpath(bag_path, buffer_dir)
                    upload_key = f"processed_{original_key}"
                    upload_bag_file(s3_client, output_bucket, output_bag, upload_key)
                    
                    processed_count += 1
                    logging.info(f"Completed processing {processed_count}/{len(bag_files)}: {os.path.basename(bag_path)}")
                    
                except Exception as e:
                    logging.error(f"Error processing {bag_path}: {e}")
                
                finally:
                    # Delete the processed file to free up buffer space
                    try:
                        os.remove(bag_path)
                        logging.debug(f"Deleted processed file: {bag_path}")
                    except Exception as e:
                        logging.warning(f"Failed to delete {bag_path}: {e}")
            
            # Wait for download thread to complete
            download_thread.join()
            logging.info(f"Completed processing all {processed_count} files")
    
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)
    
    logging.info("Done!")


if __name__ == "__main__":
    main() 