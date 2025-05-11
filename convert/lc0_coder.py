"""
LC0 Training Data Coder for Apache Beam

This module provides a custom coder for Leela Chess Zero (LC0) training data
compatible with Apache Beam pipelines. It handles V6 format and can process
V7/V7B data by ignoring the additional fields.
"""

import struct
import numpy as np
from apache_beam.coders.coders import Coder




class LC0TrainingDataCoder(Coder):
    """Custom coder for LC0 training data in V6/V7 format.
    
    This coder handles serialization and deserialization of LC0 training data
    for use in Apache Beam pipelines. It supports V6 format fully and can
    convert V7/V7B data to V6-compatible format by ignoring additional fields.
    
    Attributes:
        support_v7: Whether to handle V7/V7B data (default: True)
    """
    
    # Constants for version identification
    V6_VERSION = struct.pack("i", 6)
    V7_VERSION = struct.pack("i", 7)
    V7B_VERSION = struct.pack("i", 170)
    
    # Struct formats for V6 and V7 data
    # V6 format is 8356 bytes, with additional fields beyond V5 (which was 8308 bytes)
    # The missing fields are: result_q, result_d, played_q, played_d, played_m, orig_q, orig_d, orig_m, visits, played_idx, best_idx, pol_kld
    V6_STRUCT_STRING = "4si7432s832sBBBBBBBbfffffffffffffffIHHff"
    V7_STRUCT_STRING = "4si7432s832sBBBBBBBbfffffffffffffffIHHfffHHffffffff"
    V7B_STRUCT_STRING = "4si7432s832sBBBBBBBbfffffffffffffffIHHfffHHffffffff" + "7432s" * 2 + str(12 * 8 * 16) + "s"
    
    def __init__(self, support_v7=True):
        """Initialize the coder.
        
        Args:
            support_v7: Whether to support V7/V7B data conversion
        """
        self.support_v7 = support_v7
        self.v6_struct = struct.Struct(self.V6_STRUCT_STRING)
        
        if support_v7:
            self.v7_struct = struct.Struct(self.V7_STRUCT_STRING)
            self.v7b_struct = struct.Struct(self.V7B_STRUCT_STRING)
    
    def encode(self, training_record):
        """Encodes a dictionary representation of training data into bytes.
        
        Args:
            training_record: Dictionary with training data fields
                Required keys depend on the format version.
            
        Returns:
            Encoded byte string
            
        Raises:
            ValueError: If unsupported version or missing required fields
        """
        # Pack data according to version
        version = training_record.get('version', 6)
        
        if version == 6:
            return self._encode_v6(training_record)
        elif (version == 7 or version == 170) and self.support_v7:
            # For V7 data, we convert to V6 format for compatibility
            return self._encode_v6_from_v7(training_record)
        else:
            raise ValueError(f"Unsupported version: {version}")
    
    def _encode_v6(self, record):
        """Encode V6 format training record
        
        Args:
            record: Dictionary with V6 training data fields
            
        Returns:
            Encoded binary data
        """
        # Required fields for V6 format
        try:
            # Pack using struct format
            return self.v6_struct.pack(
                struct.pack("i", 6),  # version
                record['input_format'],
                record['probabilities'].tobytes() if isinstance(record['probabilities'], np.ndarray) else record['probabilities'],
                record['planes'],
                record['castling_us_ooo'],
                record['castling_us_oo'],
                record['castling_them_ooo'],
                record['castling_them_oo'],
                record['side_to_move'],
                record['rule50_count'],
                record['invariance_info'],
                record['result'],
                record['root_q'],
                record['best_q'],
                record['root_d'],
                record['best_d'],
                record['root_m'],
                record['best_m'],
                record['plies_left'],
                # Additional V6 fields (not in V5)
                record.get('result_q', 0.0),
                record.get('result_d', 0.0),
                record.get('played_q', 0.0),
                record.get('played_d', 0.0),
                record.get('played_m', 0.0),
                record.get('orig_q', 0.0),
                record.get('orig_d', 0.0),
                record.get('orig_m', 0.0),
                record.get('visits', 0),
                record.get('played_idx', 0),
                record.get('best_idx', 0),
                record.get('pol_kld', 0.0),
                record.get('unused', 0.0)
            )
        except KeyError as e:
            raise ValueError(f"Missing required field for V6 format: {e}")
    
    def _encode_v6_from_v7(self, record):
        """Convert V7 record to V6 format and encode
        
        Args:
            record: Dictionary with V7 training data fields
            
        Returns:
            Encoded binary data in V6 format
        """
        # Create a copy of the record with only V6 fields
        v6_record = {
            'version': 6,
            'input_format': record['input_format'],
            'probabilities': record['probabilities'],
            'planes': record['planes'],
            'castling_us_ooo': record['castling_us_ooo'],
            'castling_us_oo': record['castling_us_oo'],
            'castling_them_ooo': record['castling_them_ooo'],
            'castling_them_oo': record['castling_them_oo'],
            'side_to_move': record['side_to_move'],
            'rule50_count': record['rule50_count'],
            'invariance_info': record['invariance_info'],
            'result': record['result'],
            'root_q': record['root_q'],
            'best_q': record['best_q'],
            'root_d': record['root_d'],
            'best_d': record['best_d'],
            'root_m': record['root_m'],
            'best_m': record['best_m'],
            'plies_left': record['plies_left'],
            # Additional V6 fields
            'result_q': record.get('v7_result_q', 0.0),
            'result_d': record.get('v7_result_d', 0.0),
            'played_q': record.get('v7_played_q', 0.0),
            'played_d': record.get('v7_played_d', 0.0),
            'played_m': record.get('v7_played_m', 0.0),
            'orig_q': record.get('v7_orig_q', 0.0),
            'orig_d': record.get('v7_orig_d', 0.0),
            'orig_m': record.get('v7_orig_m', 0.0),
            'visits': record.get('v7_visits', 0),
            'played_idx': record.get('v7_played_idx', 0),
            'best_idx': record.get('v7_best_idx', 0),
            'pol_kld': record.get('v7_pol_kld', 0.0),
            'unused': 0.0
        }
        
        return self._encode_v6(v6_record)
    
    def decode(self, encoded_bytes):
        """Decodes bytes into dictionary representation.
        
        Args:
            encoded_bytes: Binary LC0 training data record
            
        Returns:
            Dictionary with training data fields
            
        Raises:
            ValueError: If unsupported version is encountered
        """
        # First 4 bytes contain version
        version = encoded_bytes[:4]
        
        if version == self.V6_VERSION:
            return self._decode_v6(encoded_bytes)
        elif version == self.V7_VERSION and self.support_v7:
            return self._decode_v7(encoded_bytes)
        elif version == self.V7B_VERSION and self.support_v7:
            return self._decode_v7b(encoded_bytes)
        else:
            raise ValueError(f"Unsupported version in binary data")
    
    def _decode_v6(self, data):
        """Decode V6 format into dictionary
        
        Args:
            data: Binary V6 training record
            
        Returns:
            Dictionary with parsed fields
        """
        # Unpack binary data using struct format
        unpacked = self.v6_struct.unpack(data)
        
        # Convert to dictionary with appropriate field names
        result = {
            'version': struct.unpack('i', unpacked[0])[0],
            'input_format': unpacked[1],
            'probabilities': np.frombuffer(unpacked[2], dtype=np.float32).reshape(-1, 1858)[0],
            'planes': unpacked[3],
            'castling_us_ooo': unpacked[4],
            'castling_us_oo': unpacked[5],
            'castling_them_ooo': unpacked[6],
            'castling_them_oo': unpacked[7],
            'side_to_move': unpacked[8],
            'rule50_count': unpacked[9],
            'invariance_info': unpacked[10],
            'result': unpacked[11],
            'root_q': unpacked[12],
            'best_q': unpacked[13],
            'root_d': unpacked[14],
            'best_d': unpacked[15],
            'root_m': unpacked[16],
            'best_m': unpacked[17],
            'plies_left': unpacked[18],
            # Additional V6 fields
            'result_q': unpacked[19],
            'result_d': unpacked[20],
            'played_q': unpacked[21],
            'played_d': unpacked[22],
            'played_m': unpacked[23],
            'orig_q': unpacked[24],
            'orig_d': unpacked[25],
            'orig_m': unpacked[26],
            'visits': unpacked[27],
            'played_idx': unpacked[28],
            'best_idx': unpacked[29],
            'pol_kld': unpacked[30],
            'unused': unpacked[31]
        }
        
        # Calculate WDL values for convenience
        result['wdl'] = self._calculate_wdl(result['best_q'], result['best_d'])
        
        return result
    
    def _decode_v7(self, data):
        """Decode V7 format but return only V6 compatible fields
        
        Args:
            data: Binary V7 training record
            
        Returns:
            Dictionary with parsed fields (V6 compatible)
        """
        # Unpack binary data using V7 struct format
        unpacked = self.v7_struct.unpack(data)
        
        # Extract V6-compatible fields
        result = {
            'version': 6,  # Force V6 compatibility
            'input_format': unpacked[1],
            'probabilities': np.frombuffer(unpacked[2], dtype=np.float32).reshape(-1, 1858)[0],
            'planes': unpacked[3],
            'castling_us_ooo': unpacked[4],
            'castling_us_oo': unpacked[5],
            'castling_them_ooo': unpacked[6],
            'castling_them_oo': unpacked[7],
            'side_to_move': unpacked[8],
            'rule50_count': unpacked[9],
            'invariance_info': unpacked[10],
            'result': unpacked[11],
            'root_q': unpacked[12],
            'best_q': unpacked[13],
            'root_d': unpacked[14],
            'best_d': unpacked[15],
            'root_m': unpacked[16],
            'best_m': unpacked[17],
            'plies_left': unpacked[18],
            'result_q': unpacked[19],
            'result_d': unpacked[20],
            'played_q': unpacked[21],
            'played_d': unpacked[22],
            'played_m': unpacked[23],
            'orig_q': unpacked[24],
            'orig_d': unpacked[25],
            'orig_m': unpacked[26],
            'visits': unpacked[27],
            'played_idx': unpacked[28],
            'best_idx': unpacked[29],
            'pol_kld': unpacked[30]
        }
        
        # Add extra V7 fields with a v7_ prefix for clarity
        result.update({
            'v7_full_data': True,
            'v7_result_q': unpacked[19],
            'v7_result_d': unpacked[20],
            'v7_played_q': unpacked[21],
            'v7_played_d': unpacked[22],
            'v7_played_m': unpacked[23],
            'v7_orig_q': unpacked[24],
            'v7_orig_d': unpacked[25],
            'v7_orig_m': unpacked[26],
            'v7_visits': unpacked[27],
            'v7_played_idx': unpacked[28],
            'v7_best_idx': unpacked[29],
            'v7_pol_kld': unpacked[30],
            'v7_q_st': unpacked[31],
            'v7_d_st': unpacked[32],
            'v7_opp_played_idx': unpacked[33],
            'v7_next_played_idx': unpacked[34],
        })
        
        # Calculate WDL values for convenience
        result['wdl'] = self._calculate_wdl(result['best_q'], result['best_d'])
        result['v7_wdl'] = self._calculate_wdl(result['v7_result_q'], result['v7_result_d'])
        
        return result

    def _decode_v7b(self, data):
        """Decode V7B format but return only V6 compatible fields
        
        Args:
            data: Binary V7B training record
            
        Returns:
            Dictionary with parsed fields (V6 compatible)
        """
        # Unpack binary data using V7B struct format
        unpacked = self.v7b_struct.unpack(data)
        
        # First get V7 compatible data
        result = self._decode_v7(data[:self.v7_struct.size])
        
        # Add V7B specific fields
        result.update({
            'v7b_full_data': True,
            'v7b_opp_probs': np.frombuffer(unpacked[35], dtype=np.float32).reshape(-1, 1858)[0],
            'v7b_next_probs': np.frombuffer(unpacked[36], dtype=np.float32).reshape(-1, 1858)[0],
            'v7b_fut': unpacked[37],
        })
        
        return result
    
    def _calculate_wdl(self, q, d):
        """Calculate win/draw/loss probabilities from Q and D values
        
        Args:
            q: Q value (-1.0 to 1.0)
            d: Draw value (0.0 to 1.0)
            
        Returns:
            Tuple of (win, draw, loss) probabilities
        """
        win = 0.5 * (1.0 + q - d)
        loss = 0.5 * (1.0 - q - d)
        return (win, d, loss)
    
    def is_deterministic(self):
        """Whether this coder is deterministic.
        
        Returns:
            True as the encoding/decoding is deterministic
        """
        return True

    def estimate_size(self, value):
        """Estimate the encoded size of the value.
        
        Args:
            value: The value to be encoded
            
        Returns:
            The expected size in bytes after encoding
        """
        version = value.get('version', 6)
        
        if version == 6:
            return self.v6_struct.size
        elif version == 7:
            return self.v7_struct.size
        elif version == 170:  # V7B
            return self.v7b_struct.size
        else:
            return self.v6_struct.size  # Default to V6 size 