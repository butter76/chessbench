#!/usr/bin/env python3
"""
Test script for Leela Chess Zero data processing

This script downloads and processes a single tar file to verify the implementation.
"""

import os
import logging
import argparse
import time
from process_lc0_data import (
    setup_logging, download_tar, extract_tar, 
    process_gz_file, process_tar_file
)
from searchless_chess.src.bagz import BagReader
from searchless_chess.src.constants import decode_lc0_data

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test Leela Chess Zero data processing')
    
    parser.add_argument('--tar_url', type=str, 
                        default='https://storage.lczero.org/files/training_data/test80/training-run1-test80-20230701-0017.tar',
                        help='URL of the tar file to process')
    parser.add_argument('--output_dir', type=str, default='test_output',
                        help='Directory for output files')
    parser.add_argument('--temp_dir', type=str, default='test_temp',
                        help='Directory for temporary files')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    return parser.parse_args()

def verify_bag_file(bag_path):
    """Verify that the bag file contains valid records."""
    try:
        reader = BagReader(bag_path)
        record_count = len(reader)
        
        if record_count == 0:
            logging.error(f"Bag file {bag_path} is empty")
            return False
        
        # Check the first few records
        for i in range(min(5, record_count)):
            record_bytes = reader[i]
            if not record_bytes:
                logging.error(f"Record {i} in bag file {bag_path} is empty")
                return False
            
            # Try to decode the record to verify it's a valid LC0DataRecord
            try:
                record = decode_lc0_data(record_bytes)
                logging.info(f"Sample record {i}: FEN={record.fen}, moves={len(record.policy)}")
            except Exception as e:
                logging.error(f"Failed to decode record {i}: {e}")
                return False
        
        logging.info(f"Bag file {bag_path} contains {record_count} valid records")
        return True
    
    except Exception as e:
        logging.error(f"Error verifying bag file {bag_path}: {e}")
        return False

def test_download_only(tar_url, temp_dir):
    """Test only the download functionality."""
    logging.info("=== TESTING DOWNLOAD ONLY ===")
    start_time = time.time()
    
    try:
        tar_path = download_tar(tar_url, temp_dir)
        if not tar_path or not os.path.exists(tar_path):
            logging.error(f"Download failed: {tar_url}")
            return None
        
        tar_size = os.path.getsize(tar_path)
        logging.info(f"Download successful: {tar_path} ({tar_size / 1024 / 1024:.2f} MB)")
        return tar_path
    except Exception as e:
        logging.error(f"Download test failed: {e}")
        return None
    finally:
        elapsed = time.time() - start_time
        logging.info(f"Download test completed in {elapsed:.2f} seconds")

def test_extract_only(tar_path, extract_dir):
    """Test only the extraction functionality."""
    logging.info("=== TESTING EXTRACTION ONLY ===")
    start_time = time.time()
    
    try:
        gz_files = extract_tar(tar_path, extract_dir)
        if not gz_files:
            logging.error(f"Extraction failed: {tar_path}")
            return None
        
        logging.info(f"Extraction successful: {len(gz_files)} files extracted to {extract_dir}")
        return gz_files
    except Exception as e:
        logging.error(f"Extraction test failed: {e}")
        return None
    finally:
        elapsed = time.time() - start_time
        logging.info(f"Extraction test completed in {elapsed:.2f} seconds")

def test_process_only(gz_file, output_path):
    """Test processing a single .gz file."""
    logging.info("=== TESTING PROCESSING ONLY ===")
    start_time = time.time()
    
    try:
        success = process_gz_file(gz_file, output_path)
        if not success:
            logging.error(f"Processing failed: {gz_file}")
            return False
        
        logging.info(f"Processing successful: {gz_file} -> {output_path}")
        return True
    except Exception as e:
        logging.error(f"Processing test failed: {e}")
        return False
    finally:
        elapsed = time.time() - start_time
        logging.info(f"Processing test completed in {elapsed:.2f} seconds")

def main():
    """Main entry point."""
    args = parse_args()
    
    # Set up logging with more detail if verbose
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        setup_logging()
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.temp_dir, exist_ok=True)
    
    # Test phases individually for better diagnostics
    logging.info(f"Testing processing for {args.tar_url}")
    
    # First test download
    tar_path = test_download_only(args.tar_url, args.temp_dir)
    if not tar_path:
        logging.error("Download phase failed. Aborting test.")
        return
    
    # Test extraction
    extract_dir = os.path.join(args.temp_dir, "extract_test")
    gz_files = test_extract_only(tar_path, extract_dir)
    if not gz_files:
        logging.error("Extraction phase failed. Aborting test.")
        return
    
    # Test processing a single file
    if gz_files:
        single_output = os.path.join(args.output_dir, "single_file_test.bag")
        if test_process_only(gz_files[0], single_output):
            logging.info("Single file processing test passed.")
            
            # Verify the output file
            if verify_bag_file(single_output):
                logging.info("Single file output verification passed.")
            else:
                logging.error("Single file output verification failed.")
        else:
            logging.error("Single file processing failed.")
    
    # Now run the full process
    logging.info("=== RUNNING FULL PROCESS TEST ===")
    full_start_time = time.time()
    
    output_path = process_tar_file(args.tar_url, args.temp_dir, args.output_dir)
    
    if output_path and os.path.exists(output_path):
        logging.info(f"Successfully created output file: {output_path}")
        
        # Verify the output file
        if verify_bag_file(output_path):
            logging.info("Test completed successfully!")
        else:
            logging.error("Test failed: Invalid bag file")
    else:
        logging.error("Test failed: No output file created")
    
    full_elapsed = time.time() - full_start_time
    logging.info(f"Full process test completed in {full_elapsed:.2f} seconds")

if __name__ == "__main__":
    main()