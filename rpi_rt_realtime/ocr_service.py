#!/usr/bin/env python3
"""
ocr_service.py - OCR Service for real-time image processing
This script runs as a persistent OCR service that reads JPEG frames
from a named pipe and processes them using EasyOCR. It is designed to
work with the receiver application which handles the real-time constraints.
"""

import os
import sys
import time
import signal
import threading
import struct
import cv2
import numpy as np
import easyocr
import logging
from datetime import datetime
import json

# Configuration
PIPE_PATH = '/tmp/ocrpipe'  # Named pipe from receiver
LOG_FILE = 'ocr_service.log'  # Log file for OCR results
RESULTS_FILE = 'ocr_results.json'  # JSON file for OCR results
DEBUG_MODE = True  # Enable debug mode to save images
DEBUG_DIR = 'debug_frames'  # Directory to save debug images

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

# Global variables
running = True
reader = None
total_frames = 0
processed_frames = 0
ocr_times = []
results_lock = threading.Lock()
ocr_results = []

def signal_handler(sig, frame):
    """Handle interrupt signals for clean shutdown."""
    global running
    logging.info("Shutdown signal received")
    running = False

def initialize_ocr():
    """Initialize the EasyOCR reader."""
    global reader
    logging.info("Initializing EasyOCR...")
    
    # Languages to recognize - you can customize this based on your needs
    languages = ['en']
    
    # Initialize reader with GPU if available
    use_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0
    if use_gpu:
        logging.info("GPU detected, using CUDA")
    else:
        logging.info("No GPU detected, using CPU")
    
    reader = easyocr.Reader(languages, gpu=use_gpu)
    logging.info("EasyOCR initialized successfully")

def setup_debug_dir():
    """Set up directory for debug frames if needed."""
    if DEBUG_MODE:
        if not os.path.exists(DEBUG_DIR):
            os.makedirs(DEBUG_DIR)
            logging.info(f"Created debug directory at {DEBUG_DIR}")

def save_results():
    """Save OCR results to a JSON file."""
    with results_lock:
        with open(RESULTS_FILE, 'w') as f:
            json.dump(ocr_results, f, indent=2)
    logging.info(f"Saved {len(ocr_results)} results to {RESULTS_FILE}")

def perform_ocr(image):
    """Perform OCR on the given image."""
    global reader, processed_frames, ocr_times
    
    start_time = time.time()
    
    # Preprocess the image to improve OCR results
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Perform OCR
    try:
        results = reader.readtext(binary)
        
        # Process the results
        text_found = []
        for (bbox, text, prob) in results:
            if prob > 0.5:  # Filter by confidence
                text_found.append({
                    'text': text,
                    'confidence': float(prob),
                    'bbox': [
                        [float(pt[0]), float(pt[1])] for pt in bbox
                    ]
                })
        
        end_time = time.time()
        ocr_time = end_time - start_time
        ocr_times.append(ocr_time)
        
        # Save results
        timestamp = datetime.now().isoformat()
        result_entry = {
            'timestamp': timestamp, 
            'frame_id': processed_frames,
            'processing_time': ocr_time,
            'text_found': text_found
        }
        
        with results_lock:
            ocr_results.append(result_entry)
        
        # Log results
        text_summary = ', '.join([item['text'] for item in text_found])
        logging.info(f"Frame {processed_frames} - OCR time: {ocr_time:.3f}s - Text: {text_summary[:100]}")
        
        # Visualize results on debug image if enabled
        if DEBUG_MODE:
            debug_image = image.copy()
            for item in text_found:
                bbox = item['bbox']
                pts = np.array(bbox, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(debug_image, [pts], True, (0, 255, 0), 2)
                cv2.putText(debug_image, f"{item['text']} ({item['confidence']:.2f})", 
                            (int(bbox[0][0]), int(bbox[0][1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            debug_path = os.path.join(DEBUG_DIR, f"frame_{processed_frames}.jpg")
            cv2.imwrite(debug_path, debug_image)
        
        processed_frames += 1
        
        # Save results periodically
        if processed_frames % 10 == 0:
            save_results()
        
        return text_found
    
    except Exception as e:
        logging.error(f"OCR error: {str(e)}")
        return []

def read_frame_from_pipe():
    """Read a JPEG frame from the named pipe."""
    try:
        with open(PIPE_PATH, 'rb') as pipe:
            # Read frame size (4 bytes)
            size_data = pipe.read(4)
            if not size_data or len(size_data) < 4:
                return None
            
            frame_size = struct.unpack('!I', size_data)[0]
            if frame_size > 10000000:  # Sanity check (10MB)
                logging.warning(f"Invalid frame size: {frame_size}")
                return None
            
            # Read frame data
            frame_data = pipe.read(frame_size)
            if len(frame_data) != frame_size:
                logging.warning(f"Incomplete frame: got {len(frame_data)}, expected {frame_size}")
                return None
            
            # Decode JPEG data
            try:
                nparr = np.frombuffer(frame_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    logging.warning("Failed to decode JPEG data")
                    return None
                return frame
            except cv2.error as e:
                logging.error(f"OpenCV error: {str(e)}")
                return None
    
    except (IOError, struct.error) as e:
        logging.error(f"Pipe read error: {str(e)}")
        return None

def report_statistics():
    """Report OCR service statistics."""
    if processed_frames == 0:
        logging.info("No frames processed yet")
        return
    
    avg_time = sum(ocr_times) / len(ocr_times) if ocr_times else 0
    max_time = max(ocr_times) if ocr_times else 0
    min_time = min(ocr_times) if ocr_times else 0
    
    logging.info("--- OCR Service Statistics ---")
    logging.info(f"Total frames received: {total_frames}")
    logging.info(f"Frames processed: {processed_frames}")
    logging.info(f"Average processing time: {avg_time:.3f}s")
    logging.info(f"Maximum processing time: {max_time:.3f}s")
    logging.info(f"Minimum processing time: {min_time:.3f}s")

def main():
    """Main function."""
    global running, total_frames
    
    # Set up signal handling
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    logging.info("Starting OCR service")
    
    # Check for named pipe
    if not os.path.exists(PIPE_PATH):
        logging.error(f"Named pipe {PIPE_PATH} does not exist")
        logging.info("Creating named pipe...")
        try:
            os.mkfifo(PIPE_PATH)
        except OSError as e:
            logging.error(f"Failed to create named pipe: {str(e)}")
            return 1
    
    # Set up debug directory
    setup_debug_dir()
    
    # Initialize OCR
    try:
        initialize_ocr()
    except Exception as e:
        logging.error(f"Failed to initialize OCR: {str(e)}")
        return 1
    
    logging.info(f"OCR service listening on pipe: {PIPE_PATH}")
    
    # Main processing loop
    try:
        while running:
            # Read frame from pipe
            frame = read_frame_from_pipe()
            
            if frame is not None:
                total_frames += 1
                logging.debug(f"Received frame {total_frames}: {frame.shape}")
                
                # Process frame with OCR
                perform_ocr(frame)
            
            # Small sleep to avoid CPU spin
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
    
    # Final statistics and cleanup
    report_statistics()
    save_results()
    logging.info("OCR service shutdown complete")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())