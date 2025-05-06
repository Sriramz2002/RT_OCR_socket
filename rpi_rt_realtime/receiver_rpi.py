#!/usr/bin/env python3
"""
receiver_rpi.py - Real-time OCR Frame Receiver with WCET Analysis
This script handles receiving JPEG frames over TCP socket and performs
worst-case execution time (WCET) analysis while forwarding frames to 
the OCR service via a named pipe.
"""

import socket
import os
import time
import signal
import threading
import csv
import numpy as np
from datetime import datetime
import fcntl
import struct
import sched
import sys

# Configuration
TCP_IP = '0.0.0.0'  # Listen on all interfaces
TCP_PORT = 5001   # Default port for receiving frames
PIPE_PATH = '/tmp/ocrpipe'  # Named pipe to OCR service
BUFFER_SIZE = 65536  # Socket buffer size
MAX_FRAME_SIZE = 100000  # Maximum expected JPEG frame size
REAL_TIME_PRIORITY = 99  # Maximum real-time priority for SCHED_OTHER
LOG_INTERVAL = 5.0  # How often to log statistics (seconds)
CSV_LOG_PATH = 'timing_stats.csv'  # Path to save timing statistics
DEADLINE_MS = 100  # Deadline in milliseconds for frame processing

# Global variables
frames_received = 0
frames_processed = 0
deadline_misses = 0
execution_times = []
jitter_values = []
running = True
stats_lock = threading.Lock()

# Create a scheduler for periodic logging
scheduler = sched.scheduler(time.time, time.sleep)

def setup_socket():
    """Setup and return a TCP socket for receiving frames."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((TCP_IP, TCP_PORT))
    sock.listen(1)
    print(f"Listening for connections on {TCP_IP}:{TCP_PORT}")
    return sock

def setup_named_pipe():
    """Setup the named pipe for communication with OCR service."""
    try:
        if not os.path.exists(PIPE_PATH):
            os.mkfifo(PIPE_PATH)
            print(f"Created named pipe at {PIPE_PATH}")
        else:
            print(f"Using existing named pipe at {PIPE_PATH}")
    except OSError as e:
        print(f"Failed to create named pipe: {e}")
        sys.exit(1)

def set_realtime_priority():
    """
    Set real-time priority for the current process.
    Note: This uses SCHED_OTHER as specified, but this is not 
    a true real-time scheduler. For true real-time, SCHED_FIFO 
    would be more appropriate.
    """
    try:
        import os
        os.nice(-20)  # Set highest nice value for SCHED_OTHER
        print("Set process priority to highest available for SCHED_OTHER")
    except (ImportError, OSError) as e:
        print(f"Failed to set real-time priority: {e}")

def write_to_pipe(frame_data):
    """Write frame data to the named pipe."""
    try:
        with open(PIPE_PATH, 'wb', buffering=0) as pipe:
            # Write frame size as a 4-byte integer
            frame_size = len(frame_data)
            pipe.write(struct.pack('!I', frame_size))
            
            # Write frame data
            pipe.write(frame_data)
            pipe.flush()
        return True
    except BrokenPipeError:
        print("Error: OCR service not reading from pipe")
        return False
    except OSError as e:
        print(f"Error writing to pipe: {e}")
        return False

def receive_frame(conn):
    """
    Receive a frame from the socket connection and measure execution time.
    Returns the execution time in milliseconds or None if there was an error.
    """
    global frames_received, frames_processed, deadline_misses
    
    try:
        # First, receive the frame size (4 bytes)
        size_data = conn.recv(4)
        if not size_data or len(size_data) < 4:
            return None
        
        frame_size = struct.unpack('!I', size_data)[0]
        if frame_size > MAX_FRAME_SIZE:
            print(f"Warning: Frame size {frame_size} exceeds maximum {MAX_FRAME_SIZE}")
            return None
        
        # Start timing
        start_time = time.time()
        
        # Receive the frame data
        frame_data = b''
        bytes_received = 0
        
        while bytes_received < frame_size:
            packet = conn.recv(min(BUFFER_SIZE, frame_size - bytes_received))
            if not packet:
                return None
            frame_data += packet
            bytes_received += len(packet)
        
        # Forward the frame to the OCR service
        write_success = write_to_pipe(frame_data)
        
        # End timing
        end_time = time.time()
        execution_time_ms = (end_time - start_time) * 1000
        
        # Update statistics
        with stats_lock:
            frames_received += 1
            if write_success:
                frames_processed += 1
            if execution_time_ms > DEADLINE_MS:
                deadline_misses += 1
            execution_times.append(execution_time_ms)
        
        return execution_time_ms
    
    except (ConnectionError, socket.timeout, struct.error) as e:
        print(f"Error receiving frame: {e}")
        return None

def log_statistics():
    """Log timing statistics to console and CSV file."""
    global execution_times, jitter_values
    
    with stats_lock:
        if not execution_times:
            print("No execution data to log yet")
            scheduler.enter(LOG_INTERVAL, 1, log_statistics, ())
            return
            
        # Calculate statistics
        avg_time = np.mean(execution_times)
        max_time = np.max(execution_times)
        min_time = np.min(execution_times)
        p95_time = np.percentile(execution_times, 95)
        p99_time = np.percentile(execution_times, 99)
        
        # Calculate jitter (variation in execution times)
        if len(execution_times) > 1:
            jitter = np.std(execution_times)
            jitter_values.append(jitter)
        else:
            jitter = 0
        
        # Calculate deadline miss rate
        miss_rate = (deadline_misses / frames_received) * 100 if frames_received > 0 else 0
        
        # Console logging
        print(f"\n--- Timing Statistics at {datetime.now()} ---")
        print(f"Frames received: {frames_received}")
        print(f"Frames processed: {frames_processed}")
        print(f"WCET (worst-case): {max_time:.2f} ms")
        print(f"Average execution time: {avg_time:.2f} ms")
        print(f"Minimum execution time: {min_time:.2f} ms")
        print(f"95th percentile: {p95_time:.2f} ms")
        print(f"99th percentile: {p99_time:.2f} ms")
        print(f"Jitter (std dev): {jitter:.2f} ms")
        print(f"Deadline misses: {deadline_misses} ({miss_rate:.2f}%)")
        
        # CSV logging
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(CSV_LOG_PATH, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            if csvfile.tell() == 0:  # File is empty, write header
                writer.writerow(['Timestamp', 'Frames_Received', 'Frames_Processed', 
                                'WCET_ms', 'Avg_Time_ms', 'Min_Time_ms', 'P95_ms', 'P99_ms', 
                                'Jitter_ms', 'Deadline_Misses', 'Miss_Rate_Percent'])
            writer.writerow([timestamp, frames_received, frames_processed, 
                            max_time, avg_time, min_time, p95_time, p99_time, 
                            jitter, deadline_misses, miss_rate])
        
        # Reset for next interval if needed
        # execution_times = []  # Uncomment to reset each interval
    
    # Schedule next log if still running
    if running:
        scheduler.enter(LOG_INTERVAL, 1, log_statistics, ())

def logger_thread_func():
    """Thread function for logging statistics."""
    scheduler.enter(LOG_INTERVAL, 1, log_statistics, ())
    scheduler.run()

def receiver_thread_func():
    """Thread function for receiving frames."""
    sock = setup_socket()
    
    while running:
        print("Waiting for connection...")
        conn, addr = sock.accept()
        print(f"Connection from {addr}")
        
        try:
            while running:
                execution_time = receive_frame(conn)
                if execution_time is None:
                    break
                # Optional: short sleep to avoid CPU hogging
                # time.sleep(0.001)
        except Exception as e:
            print(f"Connection error: {e}")
        finally:
            conn.close()
    
    sock.close()

def signal_handler(sig, frame):
    """Handle interrupt signals."""
    global running
    print("\nShutting down...")
    running = False

def main():
    """Main function."""
    # Set up signal handling
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Set up CSV log file
    with open(CSV_LOG_PATH, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Timestamp', 'Frames_Received', 'Frames_Processed', 
                        'WCET_ms', 'Avg_Time_ms', 'Min_Time_ms', 'P95_ms', 'P99_ms', 
                        'Jitter_ms', 'Deadline_Misses', 'Miss_Rate_Percent'])
    
    # Set up named pipe for OCR service
    setup_named_pipe()
    
    # Set real-time priority
    set_realtime_priority()
    
    # Start threads
    logger = threading.Thread(target=logger_thread_func)
    logger.daemon = True
    logger.start()
    
    receiver = threading.Thread(target=receiver_thread_func)
    receiver.daemon = True
    receiver.start()
    
    print("Receiver started. Press Ctrl+C to exit.")
    
    # Keep main thread alive
    try:
        while running:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    
    # Wait for threads to finish
    receiver.join(timeout=2)
    logger.join(timeout=2)
    
    print("\nFinal statistics:")
    log_statistics()
    print("Receiver shutdown complete.")

if __name__ == "__main__":
    main()
