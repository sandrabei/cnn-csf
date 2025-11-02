"""Logging utility module"""
import sys
from pathlib import Path


class Tee:
    """Class for simultaneous output to stdout and file, no buffering delay"""
    def __init__(self, filename):
        self.file = open(filename, 'w', buffering=1)  # Line buffering, no delay
        self.stdout = sys.stdout
        sys.stdout = self
    
    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()  # Flush immediately
        self.stdout.flush()  # Ensure stdout also outputs immediately
    
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    
    def close(self):
        if self.file:
            self.file.close()
        sys.stdout = self.stdout