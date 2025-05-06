/**
 * receiver_rt.cpp - Real-time OCR Frame Receiver
 * 
 * This application implements a real-time frame receiver using POSIX threads
 * with SCHED_FIFO scheduling policy to ensure deterministic execution.
 * It receives JPEG frames over TCP and forwards them to an OCR service
 * via a named pipe while maintaining real-time constraints.
 */

 #include <iostream>
 #include <string>
 #include <cstring>
 #include <vector>
 #include <unistd.h>
 #include <fcntl.h>
 #include <sys/socket.h>
 #include <netinet/in.h>
 #include <arpa/inet.h>
 #include <sys/types.h>
 #include <sys/stat.h>
 #include <sys/mman.h>
 #include <pthread.h>
 #include <sched.h>
 #include <signal.h>
 #include <atomic>
 #include <chrono>
 #include <deque>
 #include <mutex>
 #include <fstream>
 #include <iomanip>
 #include <ctime>
 #include <syslog.h>
 
 // Configuration
 constexpr int TCP_PORT = 5678;
 constexpr char PIPE_PATH[] = "/tmp/ocrpipe";
 constexpr size_t BUFFER_SIZE = 65536;
 constexpr size_t MAX_FRAME_SIZE = 100000;
 constexpr double DEADLINE_MS = 100.0;
 constexpr int RECEIVER_PRIORITY = 99;    // Highest RT priority
 constexpr int LOGGER_PRIORITY = 50;      // Medium RT priority
 constexpr int LOG_INTERVAL_SEC = 5;
 constexpr char CSV_LOG_PATH[] = "rt_timing_stats.csv";
 
 // CPU affinity configuration
 constexpr int RECEIVER_CPU_CORE = 0;    // Pin receiver thread to CPU 0
 constexpr int LOGGER_CPU_CORE = 3;      // Pin logger thread to CPU 3
 
 // Custom exception for initialization failures
 class InitializationError : public std::runtime_error {
 public:
     InitializationError(const char* msg) : std::runtime_error(msg) {}
 };
 
 // Timing statistics structure
 struct TimingStats {
     std::chrono::steady_clock::time_point start_time;
     std::chrono::steady_clock::time_point end_time;
     double execution_time_ms;
     bool deadline_missed;
 };
 
 // Global variables
 std::atomic<bool> running{true};
 std::atomic<uint64_t> frames_received{0};
 std::atomic<uint64_t> frames_processed{0};
 std::atomic<uint64_t> deadline_misses{0};
 std::mutex stats_mutex;
 std::deque<TimingStats> timing_history;
 
 // Thread IDs
 pthread_t receiver_thread_id;
 pthread_t logger_thread_id;
 
 // Forward declarations
 void* receiver_thread_func(void* arg);
 void* logger_thread_func(void* arg);
 void signal_handler(int sig);
 
 /**
  * Set up signal handlers for clean termination
  */
 void setup_signal_handlers() {
     struct sigaction sa;
     memset(&sa, 0, sizeof(sa));
     sa.sa_handler = signal_handler;
     sigaction(SIGINT, &sa, nullptr);
     sigaction(SIGTERM, &sa, nullptr);
 }
 
 /**
  * Set CPU affinity for a thread
  */
 void set_cpu_affinity(pthread_t thread, int cpu_id) {
     cpu_set_t cpuset;
     CPU_ZERO(&cpuset);
     CPU_SET(cpu_id, &cpuset);
     
     int result = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
     if (result != 0) {
         syslog(LOG_ERR, "Failed to set CPU affinity: %s", strerror(result));
     } else {
         syslog(LOG_INFO, "Thread affinity set to CPU %d", cpu_id);
     }
 }
 
 /**
  * Set up real-time attributes for a thread
  */
 void configure_rt_thread(pthread_attr_t* attr, int priority) {
     struct sched_param param;
     
     // Initialize thread attributes
     int result = pthread_attr_init(attr);
     if (result != 0) {
         throw InitializationError("Failed to initialize thread attributes");
     }
     
     // Set scheduling policy to FIFO (real-time)
     result = pthread_attr_setschedpolicy(attr, SCHED_FIFO);
     if (result != 0) {
         throw InitializationError("Failed to set scheduling policy");
     }
     
     // Set priority
     param.sched_priority = priority;
     result = pthread_attr_setschedparam(attr, &param);
     if (result != 0) {
         throw InitializationError("Failed to set thread priority");
     }
     
     // Use explicit scheduling attributes
     result = pthread_attr_setinheritsched(attr, PTHREAD_EXPLICIT_SCHED);
     if (result != 0) {
         throw InitializationError("Failed to set explicit scheduling");
     }
 }
 
 /**
  * Set up a TCP socket for receiving frames
  */
 int setup_socket() {
     int sock_fd = socket(AF_INET, SOCK_STREAM, 0);
     if (sock_fd < 0) {
         throw InitializationError("Failed to create socket");
     }
     
     // Allow reuse of local addresses
     int opt = 1;
     if (setsockopt(sock_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
         close(sock_fd);
         throw InitializationError("Failed to set socket options");
     }
     
     // Bind to local address
     struct sockaddr_in serv_addr;
     memset(&serv_addr, 0, sizeof(serv_addr));
     serv_addr.sin_family = AF_INET;
     serv_addr.sin_addr.s_addr = INADDR_ANY;
     serv_addr.sin_port = htons(TCP_PORT);
     
     if (bind(sock_fd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
         close(sock_fd);
         throw InitializationError("Failed to bind socket");
     }
     
     // Listen for connections
     if (listen(sock_fd, 5) < 0) {
         close(sock_fd);
         throw InitializationError("Failed to listen on socket");
     }
     
     return sock_fd;
 }
 
 /**
  * Set up the named pipe for communication with OCR service
  */
 void setup_named_pipe() {
     // Check if pipe already exists
     struct stat st;
     if (stat(PIPE_PATH, &st) == 0) {
         if (!S_ISFIFO(st.st_mode)) {
             unlink(PIPE_PATH);  // Remove if not a pipe
         } else {
             syslog(LOG_INFO, "Using existing named pipe at %s", PIPE_PATH);
             return;
         }
     }
     
     // Create the named pipe
     if (mkfifo(PIPE_PATH, 0666) < 0) {
         throw InitializationError("Failed to create named pipe");
     }
     
     syslog(LOG_INFO, "Created named pipe at %s", PIPE_PATH);
 }
 
 /**
  * Write frame data to the named pipe
  */
 bool write_to_pipe(const std::vector<uint8_t>& frame_data) {
     // Open pipe in non-blocking write mode
     int pipe_fd = open(PIPE_PATH, O_WRONLY | O_NONBLOCK);
     if (pipe_fd < 0) {
         syslog(LOG_ERR, "Failed to open pipe for writing: %s", strerror(errno));
         return false;
     }
     
     // Write frame size first (4 bytes)
     uint32_t frame_size = frame_data.size();
     uint32_t network_order_size = htonl(frame_size);
     
     ssize_t written = write(pipe_fd, &network_order_size, sizeof(network_order_size));
     if (written != sizeof(network_order_size)) {
         syslog(LOG_ERR, "Failed to write frame size to pipe: %s", strerror(errno));
         close(pipe_fd);
         return false;
     }
     
     // Write frame data
     written = write(pipe_fd, frame_data.data(), frame_data.size());
     if (written != static_cast<ssize_t>(frame_data.size())) {
         syslog(LOG_ERR, "Failed to write frame data to pipe: %s", strerror(errno));
         close(pipe_fd);
         return false;
     }
     
     close(pipe_fd);
     return true;
 }
 
 /**
  * Receive and process a single frame
  */
 bool receive_frame(int conn_fd, std::vector<uint8_t>& buffer) {
     // Start timing
     TimingStats stats;
     stats.start_time = std::chrono::steady_clock::now();
     
     // First, receive the frame size (4 bytes)
     uint32_t frame_size_network;
     ssize_t bytes_received = recv(conn_fd, &frame_size_network, sizeof(frame_size_network), 0);
     
     if (bytes_received != sizeof(frame_size_network)) {
         return false;
     }
     
     // Convert network byte order to host byte order
     uint32_t frame_size = ntohl(frame_size_network);
     
     // Check if frame size is valid
     if (frame_size > MAX_FRAME_SIZE || frame_size == 0) {
         syslog(LOG_WARNING, "Invalid frame size received: %u", frame_size);
         return false;
     }
     
     // Resize buffer to fit the frame
     buffer.resize(frame_size);
     
     // Receive frame data
     size_t total_received = 0;
     while (total_received < frame_size) {
         bytes_received = recv(conn_fd, buffer.data() + total_received, 
                               frame_size - total_received, 0);
         
         if (bytes_received <= 0) {
             return false;
         }
         
         total_received += bytes_received;
     }
     
     // Forward frame to OCR service
     bool write_success = write_to_pipe(buffer);
     
     // End timing
     stats.end_time = std::chrono::steady_clock::now();
     std::chrono::duration<double, std::milli> duration = stats.end_time - stats.start_time;
     stats.execution_time_ms = duration.count();
     stats.deadline_missed = stats.execution_time_ms > DEADLINE_MS;
     
     // Update statistics
     frames_received++;
     if (write_success) {
         frames_processed++;
     }
     if (stats.deadline_missed) {
         deadline_misses++;
     }
     
     // Store timing data
     {
         std::lock_guard<std::mutex> lock(stats_mutex);
         timing_history.push_back(stats);
         
         // Limit history size to avoid excessive memory usage
         if (timing_history.size() > 1000) {
             timing_history.pop_front();
         }
     }
     
     return true;
 }
 
 /**
  * Main receiver thread function
  */
 void* receiver_thread_func(void* arg) {
     (void)arg;  // Unused
     
     syslog(LOG_INFO, "Receiver thread started");
     
     // Pre-allocate buffer for frame data
     std::vector<uint8_t> buffer;
     buffer.reserve(MAX_FRAME_SIZE);
     
     try {
         int server_fd = setup_socket();
         syslog(LOG_INFO, "Socket listening on port %d", TCP_PORT);
         
         while (running) {
             syslog(LOG_INFO, "Waiting for connection...");
             
             // Accept new connection
             struct sockaddr_in client_addr;
             socklen_t client_len = sizeof(client_addr);
             int conn_fd = accept(server_fd, (struct sockaddr*)&client_addr, &client_len);
             
             if (conn_fd < 0) {
                 if (running) {
                     syslog(LOG_ERR, "Failed to accept connection: %s", strerror(errno));
                 }
                 continue;
             }
             
             char client_ip[INET_ADDRSTRLEN];
             inet_ntop(AF_INET, &client_addr.sin_addr, client_ip, INET_ADDRSTRLEN);
             syslog(LOG_INFO, "Connection from %s:%d", client_ip, ntohs(client_addr.sin_port));
             
             // Process frames from this connection
             while (running) {
                 if (!receive_frame(conn_fd, buffer)) {
                     break;
                 }
             }
             
             close(conn_fd);
             syslog(LOG_INFO, "Connection closed");
         }
         
         close(server_fd);
     }
     catch (const std::exception& e) {
         syslog(LOG_ERR, "Receiver thread error: %s", e.what());
     }
     
     syslog(LOG_INFO, "Receiver thread exiting");
     return nullptr;
 }
 
 /**
  * Calculate statistics from timing data
  */
 void calculate_statistics(double& avg_time, double& max_time, double& min_time, 
                           double& p95_time, double& p99_time, double& jitter) {
     std::lock_guard<std::mutex> lock(stats_mutex);
     
     if (timing_history.empty()) {
         avg_time = max_time = min_time = p95_time = p99_time = jitter = 0.0;
         return;
     }
     
     // Extract execution times
     std::vector<double> times;
     times.reserve(timing_history.size());
     
     for (const auto& stat : timing_history) {
         times.push_back(stat.execution_time_ms);
     }
     
     // Calculate statistics
     avg_time = 0.0;
     max_time = times[0];
     min_time = times[0];
     
     for (double time : times) {
         avg_time += time;
         max_time = std::max(max_time, time);
         min_time = std::min(min_time, time);
     }
     
     avg_time /= times.size();
     
     // Calculate jitter (standard deviation)
     double variance = 0.0;
     for (double time : times) {
         double diff = time - avg_time;
         variance += diff * diff;
     }
     
     jitter = std::sqrt(variance / times.size());
     
     // Sort for percentiles
     std::sort(times.begin(), times.end());
     
     // Calculate percentiles
     size_t p95_index = times.size() * 0.95;
     size_t p99_index = times.size() * 0.99;
     
     p95_time = times[p95_index];
     p99_time = times[p99_index];
 }
 
 /**
  * Log timing statistics
  */
 void log_statistics() {
     double avg_time, max_time, min_time, p95_time, p99_time, jitter;
     calculate_statistics(avg_time, max_time, min_time, p95_time, p99_time, jitter);
     
     uint64_t f_recv = frames_received.load();
     uint64_t f_proc = frames_processed.load();
     uint64_t d_miss = deadline_misses.load();
     
     double miss_rate = (f_recv > 0) ? (static_cast<double>(d_miss) / f_recv * 100.0) : 0.0;
     
     // Log to syslog
     syslog(LOG_INFO, "--- Timing Statistics ---");
     syslog(LOG_INFO, "Frames received: %lu", f_recv);
     syslog(LOG_INFO, "Frames processed: %lu", f_proc);
     syslog(LOG_INFO, "WCET (worst-case): %.2f ms", max_time);
     syslog(LOG_INFO, "Average execution time: %.2f ms", avg_time);
     syslog(LOG_INFO, "Minimum execution time: %.2f ms", min_time);
     syslog(LOG_INFO, "95th percentile: %.2f ms", p95_time);
     syslog(LOG_INFO, "99th percentile: %.2f ms", p99_time);
     syslog(LOG_INFO, "Jitter (std dev): %.2f ms", jitter);
     syslog(LOG_INFO, "Deadline misses: %lu (%.2f%%)", d_miss, miss_rate);
     
     // Get current timestamp
     auto now = std::chrono::system_clock::now();
     auto now_time_t = std::chrono::system_clock::to_time_t(now);
     std::tm now_tm;
     localtime_r(&now_time_t, &now_tm);
     
     // Log to CSV
     std::ofstream csv_file(CSV_LOG_PATH, std::ios::app);
     if (csv_file.is_open()) {
         // Write header if file is empty
         csv_file.seekp(0, std::ios::end);
         if (csv_file.tellp() == 0) {
             csv_file << "Timestamp,Frames_Received,Frames_Processed,"
                      << "WCET_ms,Avg_Time_ms,Min_Time_ms,P95_ms,P99_ms,"
                      << "Jitter_ms,Deadline_Misses,Miss_Rate_Percent\n";
         }
         
         char time_buf[64];
         std::strftime(time_buf, sizeof(time_buf), "%Y-%m-%d %H:%M:%S", &now_tm);
         
         csv_file << time_buf << ","
                  << f_recv << ","
                  << f_proc << ","
                  << std::fixed << std::setprecision(2)
                  << max_time << ","
                  << avg_time << ","
                  << min_time << ","
                  << p95_time << ","
                  << p99_time << ","
                  << jitter << ","
                  << d_miss << ","
                  << miss_rate << "\n";
         
         csv_file.close();
     } else {
         syslog(LOG_ERR, "Failed to open CSV log file: %s", CSV_LOG_PATH);
     }
     
     // Also log to console
     std::cout << "\n--- Timing Statistics ---" << std::endl;
     std::cout << "Frames received: " << f_recv << std::endl;
     std::cout << "Frames processed: " << f_proc << std::endl;
     std::cout << "WCET (worst-case): " << max_time << " ms" << std::endl;
     std::cout << "Average execution time: " << avg_time << " ms" << std::endl;
     std::cout << "Minimum execution time: " << min_time << " ms" << std::endl;
     std::cout << "95th percentile: " << p95_time << " ms" << std::endl;
     std::cout << "99th percentile: " << p99_time << " ms" << std::endl;
     std::cout << "Jitter (std dev): " << jitter << " ms" << std::endl;
     std::cout << "Deadline misses: " << d_miss << " (" << miss_rate << "%)" << std::endl;
 }
 
 /**
  * Logger thread function
  */
 void* logger_thread_func(void* arg) {
     (void)arg;  // Unused
     
     syslog(LOG_INFO, "Logger thread started");
     
     while (running) {
         // Sleep for the log interval
         sleep(LOG_INTERVAL_SEC);
         
         // Log statistics
         if (running) {
             log_statistics();
         }
     }
     
     syslog(LOG_INFO, "Logger thread exiting");
     return nullptr;
 }
 
 /**
  * Signal handler for graceful shutdown
  */
 void signal_handler(int sig) {
     (void)sig;  // Unused
     running = false;
     syslog(LOG_INFO, "Shutdown signal received");
 }
 
 int main() {
     try {
         // Initialize syslog
         openlog("receiver_rt", LOG_PID | LOG_CONS, LOG_USER);
         syslog(LOG_INFO, "Starting OCR receiver application");
         
         // Set up signal handlers
         setup_signal_handlers();
         
         // Lock all memory to prevent paging
         if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
             syslog(LOG_WARNING, "Failed to lock memory: %s", strerror(errno));
         } else {
             syslog(LOG_INFO, "Memory locked to prevent paging");
         }
         
         // Set up named pipe
         setup_named_pipe();
         
         // Initialize empty CSV log file
         std::ofstream csv_file(CSV_LOG_PATH, std::ios::trunc);
         if (csv_file.is_open()) {
             csv_file << "Timestamp,Frames_Received,Frames_Processed,"
                      << "WCET_ms,Avg_Time_ms,Min_Time_ms,P95_ms,P99_ms,"
                      << "Jitter_ms,Deadline_Misses,Miss_Rate_Percent\n";
             csv_file.close();
         } else {
             syslog(LOG_WARNING, "Failed to initialize CSV log file");
         }
         
         // Create threads with real-time attributes
         pthread_attr_t receiver_attr, logger_attr;
         
         // Configure receiver thread (high priority)
         configure_rt_thread(&receiver_attr, RECEIVER_PRIORITY);
         int result = pthread_create(&receiver_thread_id, &receiver_attr, receiver_thread_func, nullptr);
         if (result != 0) {
             throw InitializationError("Failed to create receiver thread");
         }
         
         // Set CPU affinity for receiver thread
         set_cpu_affinity(receiver_thread_id, RECEIVER_CPU_CORE);
         
         // Configure logger thread (medium priority)
         configure_rt_thread(&logger_attr, LOGGER_PRIORITY);
         result = pthread_create(&logger_thread_id, &logger_attr, logger_thread_func, nullptr);
         if (result != 0) {
             throw InitializationError("Failed to create logger thread");
         }
         
         // Set CPU affinity for logger thread
         set_cpu_affinity(logger_thread_id, LOGGER_CPU_CORE);
         
         // Clean up thread attributes
         pthread_attr_destroy(&receiver_attr);
         pthread_attr_destroy(&logger_attr);
         
         std::cout << "OCR Receiver started. Press Ctrl+C to exit." << std::endl;
         
         // Wait for threads to finish
         pthread_join(receiver_thread_id, nullptr);
         pthread_join(logger_thread_id, nullptr);
         
         // Final log
         std::cout << "\nFinal statistics:" << std::endl;
         log_statistics();
         
         syslog(LOG_INFO, "OCR receiver application shutdown complete");
         closelog();
         
         return 0;
     }
     catch (const std::exception& e) {
         syslog(LOG_ERR, "Fatal error: %s", e.what());
         std::cerr << "Fatal error: " << e.what() << std::endl;
         closelog();
         return 1;
     }
 }