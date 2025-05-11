#!/usr/bin/env python3
"""
Leela Chess Zero Training Data Processor

This script downloads Leela Chess Zero training data from July 1st, 2023 onwards,
processes it using the simple_convert_example function, and stores the output in bagz files.
"""

import os
import glob
import tarfile
import requests
import datetime
import time
import re
import shutil
import threading
import queue
import tempfile
import multiprocessing
from multiprocessing import Process, Queue, Event, Manager, Value
from ctypes import c_int
from tqdm import tqdm
import concurrent.futures
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import logging
import json
import argparse

# Import custom modules
from searchless_chess.convert.example import simple_convert_example
from searchless_chess.src.bagz import BagWriter, BagReader
from searchless_chess.src.constants import LC0DataRecord, encode_lc0_data, decode_lc0_data


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def fetch_tar_file_list(base_url="https://storage.lczero.org/files/training_data/test80/"):
    """
    Fetch list of tar files from Leela's storage, filtering for dates after July 1, 2023
    
    Args:
        base_url: Base URL for the Leela Chess Zero training data
        
    Returns:
        List of URLs for tar files dated after July 1, 2023
    """
    logging.info(f"Fetching file list from {base_url}")
    response = requests.get(base_url)
    file_pattern = r'(training-run1-test80-\d{8}-\d{4}\.tar)'
    files = re.findall(file_pattern, response.text)
    
    # Filter files after July 1, 2023
    cutoff_date = datetime.datetime(2023, 7, 1)
    filtered_files = []
    
    for file in files:
        # Extract date from filename (format: training-run1-test80-YYYYMMDD-HHMM.tar)
        date_str = file.split('-')[3]
        hour_str = file.split('-')[4].split('.')[0]
        
        file_date = datetime.datetime.strptime(f"{date_str} {hour_str}", "%Y%m%d %H%M")
        if file_date >= cutoff_date:
            filtered_files.append(file)
    
    sorted_files = sorted(filtered_files)  # Sort by date/time
    logging.info(f"Found {len(sorted_files)} tar files after July 1, 2023")
    
    return [f"{base_url}{file}" for file in sorted_files]


def download_tar(url, download_dir, max_retries=3, retry_delay=5):
    """
    Download a tar file to the specified directory with retry logic
    
    Args:
        url: URL of the tar file to download
        download_dir: Directory to save the downloaded file
        max_retries: Maximum number of retry attempts
        retry_delay: Delay in seconds between retries
        
    Returns:
        Path to the downloaded file
    """
    filename = os.path.basename(url)
    filepath = os.path.join(download_dir, filename)
    
    if os.path.exists(filepath):
        logging.info(f"File already exists: {filepath}")
        return filepath
    
    for attempt in range(max_retries):
        try:
            logging.info(f"Downloading {url} to {filepath} (attempt {attempt+1}/{max_retries})")
            
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                
                # Check for rate limiting response
                if r.status_code == 429 or r.status_code == 529:
                    logging.warning(f"Rate limited (status {r.status_code}). Retrying after delay.")
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                
                with open(filepath, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            logging.info(f"Downloaded {filename}")
            return filepath
            
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                logging.warning(f"Download attempt {attempt+1} failed: {e}. Retrying after {retry_delay} seconds...")
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            else:
                logging.error(f"Download failed after {max_retries} attempts: {e}")
                raise
    
    raise Exception(f"Failed to download {url} after {max_retries} attempts")


def extract_tar(tar_path, extract_dir):
    """
    Extract tar file to directory
    
    Args:
        tar_path: Path to the tar file
        extract_dir: Directory to extract the contents to
        
    Returns:
        List of paths to the extracted .gz files
    """
    os.makedirs(extract_dir, exist_ok=True)
    logging.info(f"Extracting {tar_path} to {extract_dir}")
    
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=extract_dir)
    
    gz_files = glob.glob(f"{extract_dir}/**/*.gz", recursive=True)
    logging.info(f"Extracted {len(gz_files)} .gz files")
    return gz_files


def process_data_parallel(tar_urls, num_download_workers=4, num_process_workers=56, 
                         output_dir="processed_data", temp_dir="temp_data"):
    """
    Process multiple tar files in parallel
    
    Args:
        tar_urls: List of URLs to tar files
        num_download_workers: Number of parallel download workers
        num_process_workers: Number of parallel processing workers
        output_dir: Directory for output files
        temp_dir: Directory for temporary files
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    
    logging.info(f"Processing {len(tar_urls)} tar files with {num_download_workers} download workers and {num_process_workers} process workers")
    
    # Set up shared objects for coordination between processes
    manager = Manager()
    
    # Create a shared dict for statistics
    stats = manager.dict({
        'downloads_started': 0,
        'downloads_completed': 0,
        'downloads_failed': 0,
        'processing_completed': 0,
        'processing_failed': 0,
        'records_processed': 0
    })
    
    # Create a shared queue for downloaded files waiting to be processed
    process_queue = manager.Queue()
    
    # Create an event to signal when all downloads are complete
    downloads_complete = manager.Event()
    
    # Create a lock for thread-safe operations
    download_lock = threading.Lock()
    
    # Set of files currently being downloaded
    files_in_progress = set()
    
    # Queue for URLs to download
    download_queue = queue.Queue()
    for url in tar_urls:
        download_queue.put(url)
    
    # Function to download tar files (runs in threads)
    def download_worker():
        while True:
            try:
                # Get the next URL to download with a timeout
                url = download_queue.get(timeout=1)
            except queue.Empty:
                # No more files to download
                return
            
            tar_name = os.path.basename(url)
            date_part = tar_name.split('-')[3]  # Extract YYYYMMDD
            hour_part = tar_name.split('-')[4].split('.')[0]  # Extract HHMM
            
            # Create output bag file path for this date
            output_bag = os.path.join(output_dir, f"lc0_data_{date_part}_{hour_part}.bag")
            tar_path = os.path.join(temp_dir, tar_name)
            
            # Use a lock to check if file is already being downloaded
            with download_lock:
                # Skip if output already exists
                if os.path.exists(output_bag):
                    logging.info(f"Output file already exists: {output_bag}")
                    download_queue.task_done()
                    continue
                
                # Skip if tar file is already downloaded or being downloaded
                if os.path.exists(tar_path) or tar_name in files_in_progress:
                    logging.info(f"File is already being downloaded: {tar_name}")
                    download_queue.task_done()
                    continue
                
                # Mark this file as in progress
                files_in_progress.add(tar_name)
                stats['downloads_started'] += 1
            
            # Download the file
            success = False
            try:
                # Create a temporary marker file to indicate download in progress
                marker_file = f"{tar_path}.downloading"
                with open(marker_file, 'w') as f:
                    f.write(f"Download started at {datetime.datetime.now()}")
                
                # Download the file
                try:
                    tar_path = download_tar(url, temp_dir)
                    # Add to process queue for multiprocessing workers
                    process_queue.put((url, tar_path))
                    stats['downloads_completed'] += 1
                    success = True
                    logging.info(f"Successfully downloaded: {tar_name} (Queued for processing)")
                except Exception as e:
                    stats['downloads_failed'] += 1
                    logging.error(f"Failed to download {url}: {e}")
                
                # Remove the marker file
                if os.path.exists(marker_file):
                    os.remove(marker_file)
                
                # Add a delay between downloads to avoid rate limiting
                time.sleep(2)
            finally:
                # Mark this file as no longer in progress
                with download_lock:
                    if tar_name in files_in_progress:
                        files_in_progress.remove(tar_name)
                
                # Mark task as done
                download_queue.task_done()
    
    # Function to process downloaded files (runs in separate processes)
    def process_worker(process_queue, downloads_complete, stats, worker_id):
        # Configure logging for this process
        logging.basicConfig(
            level=logging.INFO,
            format=f'%(asctime)s - Process-{worker_id} - %(levelname)s - %(message)s'
        )
        
        logging.info(f"Process worker {worker_id} started")
        
        while True:
            try:
                # Check if all downloads are complete and queue is empty
                if downloads_complete.is_set() and process_queue.empty():
                    logging.info(f"Process worker {worker_id} shutting down - no more files to process")
                    break
                
                try:
                    # Get the next file to process with a timeout
                    url, tar_path = process_queue.get(timeout=1)
                except (queue.Empty, EOFError):
                    # No files to process yet, wait and try again
                    time.sleep(0.5)
                    continue
                
                tar_name = os.path.basename(url)
                date_part = tar_name.split('-')[3]  # Extract YYYYMMDD
                hour_part = tar_name.split('-')[4].split('.')[0]  # Extract HHMM
                
                logging.info(f"Processing {tar_name} on worker {worker_id}")
                
                # Create output bag file path and extract directory
                output_bag = os.path.join(output_dir, f"lc0_data_{date_part}_{hour_part}.bag")
                extract_dir = os.path.join(temp_dir, f"extract_{date_part}_{hour_part}_{worker_id}")
                
                try:
                    # Extract .gz files
                    gz_files = extract_tar(tar_path, extract_dir)
                    
                    # Open the bag file once for this tar file
                    with BagWriter(output_bag) as writer:
                        # Process each .gz file and write directly to the bag
                        total_records = 0
                        success_count = 0
                        
                        for gz_file in gz_files:
                            try:
                                # Process the .gz file directly
                                logging.info(f"Worker {worker_id}: Processing {os.path.basename(gz_file)}")
                                records = simple_convert_example(gz_file)
                                
                                if records:
                                    # Write records directly to the bag file
                                    for record in records:
                                        serialized_record = encode_lc0_data(record)
                                        writer.write(serialized_record)
                                    
                                    total_records += len(records)
                                    success_count += 1
                                    logging.info(f"Worker {worker_id}: Processed {len(records)} records from {os.path.basename(gz_file)}")
                            except Exception as e:
                                logging.error(f"Worker {worker_id}: Error processing {gz_file}: {e}")
                    
                    # Update statistics
                    stats['records_processed'] += total_records
                    
                    logging.info(f"Worker {worker_id}: Successfully processed {success_count}/{len(gz_files)} files from {tar_name} with {total_records} total records")
                    
                    # Clean up
                    logging.info(f"Worker {worker_id}: Cleaning up temporary files for {tar_name}")
                    shutil.rmtree(extract_dir)
                    os.remove(tar_path)
                    
                    stats['processing_completed'] += 1
                    
                except Exception as e:
                    logging.error(f"Worker {worker_id}: Error processing {tar_name}: {e}")
                    # Clean up if possible
                    if os.path.exists(extract_dir):
                        shutil.rmtree(extract_dir)
                    if os.path.exists(tar_path):
                        os.remove(tar_path)
                    
                    stats['processing_failed'] += 1
            
            except Exception as e:
                logging.error(f"Worker {worker_id}: Unhandled error: {e}")
    
    # Start download threads
    download_threads = []
    for i in range(num_download_workers):
        thread = threading.Thread(target=download_worker)
        thread.daemon = True
        thread.start()
        download_threads.append(thread)
    
    # Start processing processes
    process_processes = []
    for i in range(num_process_workers):
        p = Process(
            target=process_worker, 
            args=(process_queue, downloads_complete, stats, i)
        )
        p.daemon = True
        p.start()
        process_processes.append(p)
    
    # Wait for all downloads to complete
    download_queue.join()
    logging.info("All downloads completed or skipped")
    
    # Signal to processing processes that downloads are complete
    downloads_complete.set()
    
    # Wait for process queue to be empty
    while not process_queue.empty():
        logging.info(f"Waiting for processing to complete. Items left: ~{process_queue.qsize()}")
        time.sleep(5)
    
    # Give processes some time to finish current work
    time.sleep(5)
    
    # Terminate and join processes
    for p in process_processes:
        p.terminate()
    for p in process_processes:
        p.join()
    
    # Join download threads
    for thread in download_threads:
        thread.join(timeout=1)
    
    # Print statistics
    logging.info("=== Processing Statistics ===")
    logging.info(f"Downloads started: {stats['downloads_started']}")
    logging.info(f"Downloads completed: {stats['downloads_completed']}")
    logging.info(f"Downloads failed: {stats['downloads_failed']}")
    logging.info(f"Files processed: {stats['processing_completed']}")
    logging.info(f"Processing failures: {stats['processing_failed']}")
    logging.info(f"Total records processed: {stats['records_processed']}")
    
    return stats['processing_completed']


def run_beam_pipeline(input_pattern, output_path):
    """
    Run Apache Beam pipeline to consolidate and process all data
    
    Args:
        input_pattern: Pattern to match input files
        output_path: Path to write the output
    """
    logging.info(f"Running Beam pipeline on {input_pattern}")
    
    options = PipelineOptions([
        '--runner=DirectRunner',
        '--direct_num_workers=64',
        '--direct_running_mode=multi_processing'
    ])
    
    def process_record(record_bytes):
        # Decode the record from bytes
        record = decode_lc0_data(record_bytes)
        # Process the record further if needed
        return record
    
    with beam.Pipeline(options=options) as p:
        # Read all bag files
        records = (p 
                  | 'ReadBagFiles' >> beam.Create(input_pattern)
                  | 'ReadRecords' >> beam.FlatMap(lambda file_path: BagReader(file_path))
                  | 'ProcessRecords' >> beam.Map(process_record))
        
        # Write output to a single bag file
        records | 'EncodeLc0Data' >> beam.Map(encode_lc0_data) | 'WriteToFile' >> beam.io.WriteToFile(output_path)
    
    logging.info(f"Beam pipeline completed, output written to {output_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process Leela Chess Zero training data')
    
    parser.add_argument('--output_dir', type=str, default='processed_data',
                        help='Directory for output files')
    parser.add_argument('--temp_dir', type=str, default='temp_data',
                        help='Directory for temporary files')
    parser.add_argument('--download_workers', type=int, default=4,
                        help='Number of parallel download workers')
    parser.add_argument('--process_workers', type=int, default=56,
                        help='Number of parallel processing workers')
    parser.add_argument('--beam_output', type=str, default=None,
                        help='Run Beam pipeline and output to this file')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging()
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)
    
    # Get list of tar files to download
    tar_urls = fetch_tar_file_list()
    
    # Process all tar files in parallel
    process_data_parallel(
        tar_urls,
        num_download_workers=args.download_workers,
        num_process_workers=args.process_workers,
        output_dir=args.output_dir,
        temp_dir=args.temp_dir
    )
    
    # Run Beam pipeline if requested
    if args.beam_output:
        input_pattern = os.path.join(args.output_dir, "*.bag")
        run_beam_pipeline(input_pattern, args.beam_output)
    
    logging.info("All processing complete!")


def process_tar_file(tar_url, temp_dir, output_dir):
    """
    Download, extract, process, and clean up a single tar file
    
    Args:
        tar_url: URL of the tar file
        temp_dir: Directory for temporary files
        output_dir: Directory for output files
        
    Returns:
        Path to the output bag file
    """
    tar_name = os.path.basename(tar_url)
    date_part = tar_name.split('-')[3]  # Extract YYYYMMDD
    hour_part = tar_name.split('-')[4].split('.')[0]  # Extract HHMM
    
    # Create output bag file path for this date
    output_bag = os.path.join(output_dir, f"lc0_data_{date_part}_{hour_part}.bag")
    extract_dir = os.path.join(temp_dir, f"extract_{date_part}_{hour_part}")
    
    try:
        # Skip if output already exists
        if os.path.exists(output_bag):
            logging.info(f"Output file already exists: {output_bag}")
            return output_bag
        
        # Download tar file with delay to avoid rate limiting
        logging.info(f"Downloading single file: {tar_url}")
        tar_path = download_tar(tar_url, temp_dir)
        
        # Extract .gz files
        gz_files = extract_tar(tar_path, extract_dir)
        
        # Open the bag file once for all gz files
        with BagWriter(output_bag) as writer:
            # Process each .gz file
            total_records = 0
            success_count = 0
            
            for gz_file in gz_files:
                try:
                    # Process the .gz file directly
                    logging.info(f"Processing {os.path.basename(gz_file)}")
                    records = simple_convert_example(gz_file)
                    
                    if records:
                        # Write records directly to the bag file
                        for record in records:
                            serialized_record = encode_lc0_data(record)
                            writer.write(serialized_record)
                        
                        total_records += len(records)
                        success_count += 1
                        logging.info(f"Processed {len(records)} records from {os.path.basename(gz_file)}")
                except Exception as e:
                    logging.error(f"Error processing {gz_file}: {e}")
        
        logging.info(f"Successfully processed {success_count}/{len(gz_files)} files from {tar_name} with {total_records} total records")
        
        # Clean up
        logging.info(f"Cleaning up temporary files for {tar_name}")
        shutil.rmtree(extract_dir)
        os.remove(tar_path)
        
        return output_bag
    
    except Exception as e:
        logging.error(f"Error processing {tar_url}: {e}")
        # Clean up if possible
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        if os.path.exists(tar_path):
            os.remove(tar_path)
        return None


if __name__ == "__main__":
    main() 