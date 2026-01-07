#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify content and format of reconstructed waveform data file
"""

import numpy as np
import os

def verify_reconstructed_waveforms():
    """Verify reconstructed waveform data file"""
    file_path = '../output/reconstructed_waveforms.npz'
    
    if not os.path.exists(file_path):
        print(f"❌ File does not exist: {file_path}")
        return False
    
    try:
        # Load data
        data = np.load(file_path)
        
        print("✅ Reconstructed waveform data file verification result:")
        print("=" * 50)
        
        # Check included arrays
        keys = list(data.keys())
        print(f"📊 Included arrays: {keys}")
        
        # Verify required arrays
        required_keys = ['wavelengths', 'Y_reconstructed']
        missing_keys = [key for key in required_keys if key not in keys]
        if missing_keys:
            print(f"❌ Missing required arrays: {missing_keys}")
            return False
        
        # Check wavelengths
        wavelengths = data['wavelengths']
        print(f"🌊 wavelengths shape: {wavelengths.shape}")
        print(f"🌊 wavelengths range: {wavelengths.min():.2f} - {wavelengths.max():.2f}")
        print(f"🌊 wavelengths dtype: {wavelengths.dtype}")
        
        # Check Y_reconstructed
        Y_reconstructed = data['Y_reconstructed']
        print(f"📈 Y_reconstructed shape: {Y_reconstructed.shape}")
        print(f"📈 Y_reconstructed range: {Y_reconstructed.min():.6f} - {Y_reconstructed.max():.6f}")
        print(f"📈 Y_reconstructed dtype: {Y_reconstructed.dtype}")
        
        # Check data dimension consistency
        if len(wavelengths) != Y_reconstructed.shape[1]:
            print(f"❌ Dimension mismatch: wavelengths length ({len(wavelengths)}) != Y_reconstructed columns ({Y_reconstructed.shape[1]})")
            return False
        
        # Show partial data
        print(f"🔍 wavelengths first 5 values: {wavelengths[:5]}")
        print(f"🔍 Y_reconstructed first frame first 5 values: {Y_reconstructed[0, :5]}")
        
        # Check if original data is included
        if 'Y_original' in keys:
            Y_original = data['Y_original']
            print(f"📊 Y_original shape: {Y_original.shape}")
            print("✅ Contains original data for comparison")
        else:
            print("⚠️  Does not contain original data (Y_original)")
        
        # Check data integrity
        if np.any(np.isnan(wavelengths)):
            print("❌ wavelengths contains NaN values")
            return False
        
        if np.any(np.isnan(Y_reconstructed)):
            print("❌ Y_reconstructed contains NaN values")
            return False
        
        print("=" * 50)
        print("✅ Data verification passed! Reconstructed waveform data format is correct and complete.")
        
        data.close()
        return True
        
    except Exception as e:
        print(f"❌ Error during verification: {e}")
        return False

if __name__ == "__main__":
    verify_reconstructed_waveforms()
