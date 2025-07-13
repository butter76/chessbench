#!/bin/bash

# Chess Search Tree Viewer GUI Runner
# This script sets up and runs the GUI application

set -e

echo "Chess Search Tree Viewer GUI"
echo "============================"

# Check if we're in the gui directory
if [ ! -f "package.json" ]; then
    echo "Error: Please run this script from the gui/ directory"
    echo "cd gui && ./run_gui.sh"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed"
    echo "Please install Node.js from https://nodejs.org/"
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "Error: npm is not installed"
    echo "Please install npm (usually comes with Node.js)"
    exit 1
fi

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Check if user wants to generate sample logs
echo
echo "Options:"
echo "1. Start GUI with existing logs"
echo "2. Generate sample logs and start GUI"
echo "3. Generate sample logs only"
echo

read -p "Choose an option (1-3): " choice

case $choice in
    1)
        echo "Starting GUI..."
        npm start
        ;;
    2)
        echo "Generating sample logs..."
        if [ -f "capture_search_logs.py" ]; then
            python3 capture_search_logs.py
            echo "Sample logs generated. Starting GUI..."
            npm start
        else
            echo "Error: capture_search_logs.py not found"
            exit 1
        fi
        ;;
    3)
        echo "Generating sample logs..."
        if [ -f "capture_search_logs.py" ]; then
            python3 capture_search_logs.py
            echo "Sample logs generated. You can now start the GUI with: npm start"
        else
            echo "Error: capture_search_logs.py not found"
            exit 1
        fi
        ;;
    *)
        echo "Invalid option. Starting GUI..."
        npm start
        ;;
esac 