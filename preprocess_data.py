#!/usr/bin/env python3
"""
Main data processing script for 3D medical imaging
Consolidates all offline data processing into a single pipeline

New format: Files directly in directory with naming pattern: prefix_1csf.txt, prefix_1epi.txt, prefix_1t1.txt

Usage:
    python main_data.py <input_dir> <output_dir>
    
Example:
    python main_data.py dataset_135 output_data
    python main_data.py dataset_135 output_data --parallel --workers 4
"""

import os
import sys
import shutil
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

class DataProcessor:
    def __init__(self, input_dir, output_dir, max_workers=None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.pixel_size = 10
        self.max_workers = max_workers or max(1, multiprocessing.cpu_count() - 1)
        
    def load_2d_data(self, file_path):
        """Load 2D data from 3-column format (x y value)"""
        if not file_path.exists():
            print(f"Warning: {file_path} not found")
            return np.zeros((64, 64), dtype=np.float32)
            
        data = np.zeros((64, 64), dtype=np.float32)
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        x, y, val = int(parts[0]), int(parts[1]), float(parts[2])
                        if 0 <= x < 64 and 0 <= y < 64:
                            data[x, y] = val
                    except (ValueError, IndexError):
                        continue
                        
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
        return data

    def check_data_validity(self, csf_data, epi_data, t1_data):
        """Check if any of the data arrays are all zeros"""
        invalid_files = []
        
        if np.all(csf_data == 0):
            invalid_files.append("csf")
        if np.all(epi_data == 0):
            invalid_files.append("epi")
        if np.all(t1_data == 0):
            invalid_files.append("t1")
            
        return invalid_files
    
    def load_csf_labels(self, file_path):
        """Load CSF labels for z=0"""
        if not file_path.exists():
            print(f"Warning: {file_path} not found")
            return np.zeros((64, 64), dtype=bool)
            
        mask = np.zeros((64, 64), dtype=bool)
        csf_count = 0
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 4:
                        try:
                            x, y, z, val = map(int, parts)
                            if z == 0 and 0 <= x < 64 and 0 <= y < 64 and val == 1:
                                mask[x, y] = True
                                csf_count += 1
                        except ValueError:
                            continue
                        
            print(f"  Found {csf_count} CSF labels for z=0 slice")
                        
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
        return mask
    
    def generate_image(self, data, mask=None, add_labels=False):
        """Generate 640x640 image from 64x64 data"""
        target_size = 640
        pixel_size = target_size // 64
        
        # Upsample to 640x640
        upsampled = np.repeat(np.repeat(data, pixel_size, axis=0), pixel_size, axis=1)
        
        # Create RGB image
        if len(upsampled.shape) == 2:
            img_array = np.stack([upsampled, upsampled, upsampled], axis=-1)
        else:
            img_array = upsampled
            
        # Add hollow red squares if requested
        if add_labels and mask is not None:
            for idx1 in range(64):
                for idx2 in range(64):
                    if mask[idx1, idx2]:
                        # Calculate block boundaries
                        y_block_start = idx1 * pixel_size
                        x_block_start = idx2 * pixel_size
                        y_block_end = (idx1 + 1) * pixel_size
                        x_block_end = (idx2 + 1) * pixel_size
                        
                        # Make red border thicker and more visible
                        border_width = 3
                        y_outer_start = max(0, y_block_start - border_width)
                        x_outer_start = max(0, x_block_start - border_width)
                        y_outer_end = min(640, y_block_end + border_width)
                        x_outer_end = min(640, x_block_end + border_width)
                        
                        # Inner transparent area
                        y_inner_start = max(0, y_block_start + border_width)
                        x_inner_start = max(0, x_block_start + border_width)
                        y_inner_end = min(640, y_block_end - border_width)
                        x_inner_end = min(640, x_block_end - border_width)
                        
                        # Fill bright red border
                        img_array[y_outer_start:y_outer_end, x_outer_start:x_outer_end] = [1.0, 0.0, 0.0]
                        
                        # Make inner area transparent (restore original)
                        if y_inner_start < y_inner_end and x_inner_start < x_inner_end:
                            orig_val = data[idx1, idx2]
                            img_array[y_inner_start:y_inner_end, x_inner_start:x_inner_end] = [orig_val, orig_val, orig_val]
        
        return img_array
    
    def create_debug_image(self, subject_id):
        """Create 1280x1280 debug image for a subject"""
        # Load data from processed files
        subject_dir = self.output_dir / subject_id
        epi_data = self.load_2d_data(subject_dir / "epi.txt")
        t1_data = self.load_2d_data(subject_dir / "t1.txt")
        
        # Load CSF labels from original file (new format)
        csf_file = self.input_dir / f"{subject_id}_1csf.txt"
        csf_mask = self.load_csf_labels(csf_file)
        
        # Generate 4 images
        epi_img = self.generate_image(epi_data)
        t1_img = self.generate_image(t1_data)
        epi_labeled_img = self.generate_image(epi_data, csf_mask, add_labels=True)
        t1_labeled_img = self.generate_image(t1_data, csf_mask, add_labels=True)
        
        # Create 1280x1280 debug image
        debug_img = np.zeros((1280, 1280, 3), dtype=np.float32)
        
        # Top-left: epi
        debug_img[0:640, 0:640] = epi_img
        # Top-right: t1
        debug_img[0:640, 640:1280] = t1_img
        # Bottom-left: epi + csf
        debug_img[640:1280, 0:640] = epi_labeled_img
        # Bottom-right: t1 + csf
        debug_img[640:1280, 640:1280] = t1_labeled_img
        
        # Save debug image
        debug_path = subject_dir / "debug.png"
        fig, ax = plt.subplots(figsize=(12.8, 12.8), dpi=100)
        ax.imshow(debug_img)
        ax.set_axis_off()
        ax.set_position([0, 0, 1, 1])
        plt.savefig(debug_path, dpi=100, bbox_inches='tight', pad_inches=0, facecolor='black')
        plt.close()
        
        return debug_path
    
    def extract_z0_slice(self, input_file, output_file):
        """Extract z=0 slice and save as 3-column format (x y value)"""
        try:
            z0_data = {}
            max_val = 0
            min_val = float('inf')
            is_csf = "csf" in str(input_file).lower()
            
            with open(input_file, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        x, y, z, val = int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3])
                        if z == 0 and 0 <= x < 64 and 0 <= y < 64:
                            if is_csf:
                                # CSF labels: keep original 0/1 values
                                z0_data[(x, y)] = int(val)
                            else:
                                # EPI/T1 data: normalize values
                                z0_data[(x, y)] = val
                                max_val = max(max_val, val)
                                min_val = min(min_val, val)
                    except (ValueError, IndexError):
                        continue
            
            if is_csf:
                # CSF labels: use original values
                final_data = z0_data
            else:
                # EPI/T1 data: normalize values
                final_data = {}
                for (x, y), val in z0_data.items():
                    if max_val > min_val:
                        normalized_val = (val - min_val) / (max_val - min_val)
                    elif max_val > 0:
                        normalized_val = val / max_val
                    else:
                        normalized_val = 0.0
                    final_data[(x, y)] = normalized_val
            
            # Write 3-column format (x y value)
            with open(output_file, 'w') as f:
                for x in range(64):
                    for y in range(64):
                        val = final_data.get((x, y), 0.0)
                        f.write(f"{x} {y} {val}\n")
                        
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            # Create empty file on error
            with open(output_file, 'w') as f:
                for x in range(64):
                    for y in range(64):
                        f.write(f"{x} {y} 0.0\n")
    
    def process_subject(self, subject_id):
        """Process a single subject (new format - files directly in directory)"""
        output_subject_dir = self.output_dir / subject_id
        
        print(f"Processing subject: {subject_id}")
        
        # Create output directory
        output_subject_dir.mkdir(parents=True, exist_ok=True)
        
        # Process files using new format naming
        files_found = []
        
        # Check for new format files
        csf_file = self.input_dir / f"{subject_id}_1csf.txt"
        epi_file = self.input_dir / f"{subject_id}_1epi.txt"
        t1_file = self.input_dir / f"{subject_id}_1t1.txt"
        
        # Process CSF file
        if csf_file.exists():
            target_file = output_subject_dir / "csf.txt"
            self.extract_z0_slice(csf_file, target_file)
            files_found.append("csf.txt")
        
        # Process EPI file
        if epi_file.exists():
            target_file = output_subject_dir / "epi.txt"
            self.extract_z0_slice(epi_file, target_file)
            files_found.append("epi.txt")
        
        # Process T1 file
        if t1_file.exists():
            target_file = output_subject_dir / "t1.txt"
            self.extract_z0_slice(t1_file, target_file)
            files_found.append("t1.txt")
        
        # Check required files exist
        required_files = ["csf.txt", "epi.txt", "t1.txt"]
        missing_files = [f for f in required_files if f not in files_found]
        
        if missing_files:
            print(f"  Warning: Missing required files: {missing_files}")
            return False, False  # Return (success, valid_data)
        
        # Check data validity (all-zero check)
        csf_data = self.load_2d_data(output_subject_dir / "csf.txt")
        epi_data = self.load_2d_data(output_subject_dir / "epi.txt")
        t1_data = self.load_2d_data(output_subject_dir / "t1.txt")
        
        invalid_files = self.check_data_validity(csf_data, epi_data, t1_data)
        is_valid = len(invalid_files) == 0
        
        if not is_valid:
            print(f"  Warning: Invalid zero data for subject {subject_id}: {invalid_files}")
        
        # Generate debug image
        debug_path = self.create_debug_image(subject_id)
        print(f"  Generated debug image: {debug_path}")
        
        return True, is_valid  # Return (success, valid_data)
    
    def run_parallel(self):
        """Parallel processing pipeline (new format)"""
        if not self.input_dir.exists():
            print(f"Error: Input directory {self.input_dir} does not exist")
            return
            
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all unique subject prefixes from files
        subject_ids = set()
        for file in self.input_dir.iterdir():
            if file.is_file() and file.name.endswith(('.txt')):
                # Extract subject ID from filename (e.g., HC001_1csf.txt -> HC001)
                parts = file.name.rsplit('_', 1)
                if len(parts) == 2 and parts[1] in ['1csf.txt', '1epi.txt', '1t1.txt']:
                    subject_ids.add(parts[0])
        
        if not subject_ids:
            print("No subject files found in input directory")
            return
            
        subject_ids = sorted(subject_ids)
        print(f"Found {len(subject_ids)} subjects")
        print(f"Using {self.max_workers} parallel workers")
        
        # Prepare list file path
        list_file_path = Path(str(self.output_dir) + ".list")
        valid_subjects = []
        
        # Process subjects in parallel
        processed = 0
        failed = 0
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_subject = {
                executor.submit(self.process_subject, subject_id): subject_id 
                for subject_id in subject_ids
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_subject):
                subject_name = future_to_subject[future]
                try:
                    success, is_valid = future.result()
                    if success:
                        processed += 1
                        if is_valid:
                            valid_subjects.append(subject_name)
                        print(f"✓ Completed: {subject_name} ({processed}/{len(subject_ids)})")
                    else:
                        failed += 1
                        print(f"✗ Failed: {subject_name}")
                except Exception as e:
                    failed += 1
                    print(f"✗ Error processing {subject_name}: {e}")
        
        # Write valid subjects to list file (three-column format)
        with open(list_file_path, 'w') as f:
            for subject in sorted(valid_subjects):
                subject_dir = self.output_dir / subject
                epi_path = str(subject_dir / "epi.txt")
                t1_path = str(subject_dir / "t1.txt")
                csf_path = str(subject_dir / "csf.txt")
                f.write(f"{epi_path},{t1_path},{csf_path}\n")
        
        print(f"Completed processing {processed} subjects, {failed} failed")
        print(f"Valid subjects written to: {list_file_path}")

    def run(self):
        """Main processing pipeline (new format - sequential)"""
        if not self.input_dir.exists():
            print(f"Error: Input directory {self.input_dir} does not exist")
            return
            
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all unique subject prefixes from files
        subject_ids = set()
        for file in self.input_dir.iterdir():
            if file.is_file() and file.name.endswith(('.txt')):
                # Extract subject ID from filename (e.g., HC001_1csf.txt -> HC001)
                parts = file.name.rsplit('_', 1)
                if len(parts) == 2 and parts[1] in ['1csf.txt', '1epi.txt', '1t1.txt']:
                    subject_ids.add(parts[0])
        
        if not subject_ids:
            print("No subject files found in input directory")
            return
            
        subject_ids = sorted(subject_ids)
        print(f"Found {len(subject_ids)} subjects")
        
        # Prepare list file path
        list_file_path = Path(str(self.output_dir) + ".list")
        valid_subjects = []
        
        processed = 0
        for subject_id in subject_ids:
            success, is_valid = self.process_subject(subject_id)
            if success:
                processed += 1
                if is_valid:
                    valid_subjects.append(subject_id)
        
        # Write valid subjects to list file (three-column format)
        with open(list_file_path, 'w') as f:
            for subject in sorted(valid_subjects):
                subject_dir = self.output_dir / subject
                epi_path = str(subject_dir / "epi.txt")
                t1_path = str(subject_dir / "t1.txt")
                csf_path = str(subject_dir / "csf.txt")
                f.write(f"{epi_path},{t1_path},{csf_path}\n")
        
        print(f"Completed processing {processed} subjects")
        print(f"Valid subjects written to: {list_file_path}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python main_data.py <input_dir> <output_dir> [--parallel] [--workers N]")
        print("Examples:")
        print("  Sequential: python main_data.py dataset_135 output_data")
        print("  Parallel:   python main_data.py dataset_135 output_data --parallel")
        print("  Custom workers: python main_data.py dataset_135 output_data --parallel --workers 4")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Parse command line arguments
    use_parallel = "--parallel" in sys.argv
    max_workers = None
    
    if "--workers" in sys.argv:
        try:
            worker_index = sys.argv.index("--workers")
            max_workers = int(sys.argv[worker_index + 1])
        except (IndexError, ValueError):
            print("Warning: Invalid --workers value, using default")
    
    processor = DataProcessor(input_dir, output_dir, max_workers)
    
    if use_parallel:
        print("Starting parallel processing...")
        processor.run_parallel()
    else:
        print("Starting sequential processing...")
        processor.run()

if __name__ == "__main__":
    main()