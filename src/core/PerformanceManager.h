#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <mutex>
#include <limits>

// Singleton class for tracking performance metrics across the application.
// Usage:
//   PerformanceManager::instance().startTimer("PhotonTracing");
//   // ... do work ...
//   PerformanceManager::instance().stopTimer("PhotonTracing");
//
// At the end:
//   PerformanceManager::instance().exportMetrics("PerformanceMetrics.txt");

class PerformanceManager
{
public:
    // Get the singleton instance
    static PerformanceManager& instance()
    {
        static PerformanceManager inst;
        return inst;
    }

    // Delete copy/move constructors for singleton
    PerformanceManager(const PerformanceManager&) = delete;
    PerformanceManager& operator=(const PerformanceManager&) = delete;
    PerformanceManager(PerformanceManager&&) = delete;
    PerformanceManager& operator=(PerformanceManager&&) = delete;

    // Start timing a task (creates entry if doesn't exist)
    void startTimer(const std::string& taskName)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        TaskMetrics& task = tasks_[taskName];
        task.startTime = std::chrono::high_resolution_clock::now();
        task.isRunning = true;
    }

    // Stop timing a task and accumulate the duration
    void stopTimer(const std::string& taskName)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = tasks_.find(taskName);
        if (it == tasks_.end() || !it->second.isRunning)
        {
            std::cerr << "PerformanceManager: Warning - stopTimer called for '" 
                      << taskName << "' which was not started." << std::endl;
            return;
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double, std::milli>(endTime - it->second.startTime).count();
        
        it->second.totalTimeMs += duration;
        it->second.callCount++;
        it->second.lastDurationMs = duration;
        it->second.minDurationMs = (std::min)(it->second.minDurationMs, duration);
        it->second.maxDurationMs = (std::max)(it->second.maxDurationMs, duration);
        it->second.isRunning = false;
    }

    // Record a single measurement directly (for external timing)
    void recordTime(const std::string& taskName, double durationMs)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        TaskMetrics& task = tasks_[taskName];
        task.totalTimeMs += durationMs;
        task.callCount++;
        task.lastDurationMs = durationMs;
        task.minDurationMs = (std::min)(task.minDurationMs, durationMs);
        task.maxDurationMs = (std::max)(task.maxDurationMs, durationMs);
    }

    // Get the last recorded duration for a task (in milliseconds)
    double getLastDuration(const std::string& taskName) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = tasks_.find(taskName);
        if (it != tasks_.end())
            return it->second.lastDurationMs;
        return 0.0;
    }

    // Get total accumulated time for a task (in milliseconds)
    double getTotalTime(const std::string& taskName) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = tasks_.find(taskName);
        if (it != tasks_.end())
            return it->second.totalTimeMs;
        return 0.0;
    }

    // Get call count for a task
    unsigned int getCallCount(const std::string& taskName) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = tasks_.find(taskName);
        if (it != tasks_.end())
            return it->second.callCount;
        return 0;
    }

    // Reset all metrics
    void reset()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        tasks_.clear();
    }

    // Generate metrics report as a string (for ExporterManager to write)
    // configSummary: optional configuration info to include in report header
    std::string generateMetricsReport(const std::string& configSummary = "") const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        std::ostringstream oss;

        // Header
        oss << "================================================================================\n";
        oss << "                        PERFORMANCE METRICS REPORT                              \n";
        oss << "================================================================================\n\n";

        // Get current time for the report
        auto now = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now);
        oss << "Generated: " << std::ctime(&time_t_now);

        // Include configuration if provided
        if (!configSummary.empty())
        {
            oss << "\n" << configSummary << "\n";
        }
        oss << "\n";

        // Sort tasks by total time (descending)
        std::vector<std::pair<std::string, TaskMetrics>> sortedTasks(tasks_.begin(), tasks_.end());
        std::sort(sortedTasks.begin(), sortedTasks.end(),
            [](const std::pair<std::string, TaskMetrics>& a, const std::pair<std::string, TaskMetrics>& b) 
            { return a.second.totalTimeMs > b.second.totalTimeMs; });

        // Calculate grand total
        double grandTotal = 0.0;
        for (size_t i = 0; i < sortedTasks.size(); ++i)
        {
            grandTotal += sortedTasks[i].second.totalTimeMs;
        }

        oss << "--------------------------------------------------------------------------------\n";
        oss << std::left << std::setw(35) << "Task Name"
            << std::right << std::setw(12) << "Total (ms)"
            << std::setw(10) << "Calls"
            << std::setw(12) << "Avg (ms)"
            << std::setw(12) << "Min (ms)"
            << std::setw(12) << "Max (ms)"
            << std::setw(8) << "%" << "\n";
        oss << "--------------------------------------------------------------------------------\n";

        for (size_t i = 0; i < sortedTasks.size(); ++i)
        {
            const std::string& name = sortedTasks[i].first;
            const TaskMetrics& metrics = sortedTasks[i].second;
            
            double avgTime = metrics.callCount > 0 ? metrics.totalTimeMs / metrics.callCount : 0.0;
            double percentage = grandTotal > 0 ? (metrics.totalTimeMs / grandTotal) * 100.0 : 0.0;
            double minTime = (metrics.minDurationMs == (std::numeric_limits<double>::max)()) ? 0.0 : metrics.minDurationMs;

            oss << std::left << std::setw(35) << name
                << std::right << std::fixed << std::setprecision(2)
                << std::setw(12) << metrics.totalTimeMs
                << std::setw(10) << metrics.callCount
                << std::setw(12) << avgTime
                << std::setw(12) << minTime
                << std::setw(12) << metrics.maxDurationMs
                << std::setw(7) << percentage << "%\n";
        }

        oss << "--------------------------------------------------------------------------------\n";
        oss << std::left << std::setw(35) << "TOTAL"
            << std::right << std::fixed << std::setprecision(2)
            << std::setw(12) << grandTotal << " ms"
            << " (" << (grandTotal / 1000.0) << " seconds)\n";
        oss << "================================================================================\n";

        return oss.str();
    }

    // Check if there are any recorded metrics
    bool hasMetrics() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return !tasks_.empty();
    }

    // Print summary to console
    void printSummary() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        
        std::cout << "\n=== Performance Summary ===" << std::endl;
        
        std::vector<std::pair<std::string, TaskMetrics>> sortedTasks(tasks_.begin(), tasks_.end());
        std::sort(sortedTasks.begin(), sortedTasks.end(),
            [](const std::pair<std::string, TaskMetrics>& a, const std::pair<std::string, TaskMetrics>& b) 
            { return a.second.totalTimeMs > b.second.totalTimeMs; });

        for (size_t i = 0; i < sortedTasks.size(); ++i)
        {
            const std::string& name = sortedTasks[i].first;
            const TaskMetrics& metrics = sortedTasks[i].second;
            double avgTime = metrics.callCount > 0 ? metrics.totalTimeMs / metrics.callCount : 0.0;
            std::cout << "  " << name << ": " 
                      << std::fixed << std::setprecision(2) << metrics.totalTimeMs << " ms total, "
                      << metrics.callCount << " calls, "
                      << avgTime << " ms avg" << std::endl;
        }
        std::cout << "===========================" << std::endl;
    }

private:
    PerformanceManager() = default;
    ~PerformanceManager() = default;

    struct TaskMetrics
    {
        std::chrono::high_resolution_clock::time_point startTime;
        double totalTimeMs = 0.0;
        double lastDurationMs = 0.0;
        double minDurationMs = (std::numeric_limits<double>::max)();
        double maxDurationMs = 0.0;
        unsigned int callCount = 0;
        bool isRunning = false;
    };

    mutable std::mutex mutex_;
    std::unordered_map<std::string, TaskMetrics> tasks_;
};

// RAII helper for automatic timing
// Usage: { ScopedTimer timer("MyTask"); /* work */ } // automatically stops when scope exits
class ScopedTimer
{
public:
    explicit ScopedTimer(const std::string& taskName)
        : taskName_(taskName)
    {
        PerformanceManager::instance().startTimer(taskName_);
    }

    ~ScopedTimer()
    {
        PerformanceManager::instance().stopTimer(taskName_);
    }

    // Disable copy/move
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;

private:
    std::string taskName_;
};

// Convenience macros for common usage
#define PERF_START(name) PerformanceManager::instance().startTimer(name)
#define PERF_STOP(name) PerformanceManager::instance().stopTimer(name)
#define PERF_SCOPED(name) ScopedTimer _scopedTimer##__LINE__(name)
