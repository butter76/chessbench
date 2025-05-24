"""
LC0 Training Data Chunk Parser

This module provides utilities for parsing LC0 training data chunks
and extracting individual training records.
"""

import struct
import gzip
import os
import random
from typing import List, Iterator, Tuple, BinaryIO, Optional

import apache_beam as beam


class ParseChunksToRecords(beam.DoFn):
    """Split LC0 chunk files into individual training records.
    
    This DoFn processes LC0 training data chunk files, identifying the
    version of each chunk and splitting it into individual training records
    for downstream processing.
    
    Attributes:
        sample_rate: Rate at which to sample positions (default: 1.0)
        v6_size: Size of V6 record in bytes
        v7_size: Size of V7 record in bytes
        v7b_size: Size of V7B record in bytes
    """
    
    def __init__(self, sample_rate: float = 1.0):
        """Initialize the parser.
        
        Args:
            sample_rate: Sampling rate from 0.0 to 1.0. Use 1.0 to keep all positions.
        """
        self.sample_rate = sample_rate
        self.v6_size = 8356   # Size of V6 record in bytes
        self.v7_size = 8396   # Size of V7 record in bytes
        self.v7b_size = 16580  # Approximate size of V7B (may need adjustment)
    
    def process(self, element):
        """Process a binary chunk file into individual records.
        
        Args:
            element: File object or binary content from a chunk file
            
        Yields:
            Individual training records
        """
        # Handle different input types
        if hasattr(element, 'read'):
            # If it's a file-like object
            data = element.read()
        else:
            # If it's already bytes
            data = element
        
        # Skip empty files
        if not data:
            return
        
        # Determine record size from version
        version = data[:4]
        if version == struct.pack("i", 6):
            record_size = self.v6_size
        elif version == struct.pack("i", 7):
            record_size = self.v7_size
        elif version == struct.pack("i", 170):
            record_size = self.v7b_size
        else:
            # Try to interpret as a packed integer for the error message
            try:
                ver_int = struct.unpack('i', version)[0]
                raise ValueError(f"Unknown version: {ver_int}")
            except:
                raise ValueError(f"Unknown version format in chunk")
        
        # Split into records and yield each
        offset = 0
        while offset + record_size <= len(data):
            # Apply sampling
            if random.random() <= self.sample_rate:
                yield data[offset:offset+record_size]
            offset += record_size


def read_chunks(filepath: str, decompress: bool = True) -> Iterator[bytes]:
    """Read training records from a chunk file.
    
    Args:
        filepath: Path to the chunk file
        decompress: Whether to decompress gzip files
        
    Yields:
        Individual training records as byte strings
    """
    # Determine if decompression is needed
    is_gzip = filepath.endswith('.gz')
    
    # Open file with appropriate mode
    open_func = gzip.open if is_gzip and decompress else open
    mode = 'rb'
    
    parser = ParseChunksToRecords(sample_rate=1.0)
    
    with open_func(filepath, mode) as f:
        data = f.read()
        yield from parser.process(data)


def get_chunk_files(directory: str, pattern: str = "*.gz", recursive: bool = True) -> List[str]:
    """Get a list of chunk files in a directory.
    
    Args:
        directory: Directory containing chunk files
        pattern: File pattern to match (default: "*.gz")
        recursive: Whether to search recursively
        
    Returns:
        List of paths to chunk files
    """
    import glob
    
    search_pattern = os.path.join(directory, "**", pattern) if recursive else os.path.join(directory, pattern)
    return glob.glob(search_pattern, recursive=recursive)


class ChunkReader:
    """Reader for LC0 training chunks.
    
    This class provides an iterator interface for reading training records
    from chunk files, with optional sampling.
    
    Attributes:
        files: List of chunk files to read
        sample_rate: Sampling rate for positions
    """
    
    def __init__(self, files: List[str], sample_rate: float = 1.0):
        """Initialize the reader.
        
        Args:
            files: List of file paths to read
            sample_rate: Sampling rate from 0.0 to 1.0
        """
        self.files = files
        self.sample_rate = sample_rate
    
    def __iter__(self) -> Iterator[bytes]:
        """Iterate through all records in all chunk files.
        
        Yields:
            Individual training records as byte strings
        """
        # Process all files
        for filepath in self.files:
            parser = ParseChunksToRecords(sample_rate=self.sample_rate)
            
            # Determine if decompression is needed
            is_gzip = filepath.endswith('.gz')
            
            # Open file with appropriate mode
            open_func = gzip.open if is_gzip else open
            mode = 'rb'
            
            with open_func(filepath, mode) as f:
                data = f.read()
                yield from parser.process(data) 