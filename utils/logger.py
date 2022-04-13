import sys
import os


class Print_Logger(object):
    def __init__(self, filename="default.log", stream=sys.stdout):
        self.terminal = stream
        filename = os.path.join('logs', filename)
        self.log = open(filename, "a+")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
