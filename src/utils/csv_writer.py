import logging
import os
from csv import DictWriter
from typing import Any, Dict, List

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CSVWriter:
    def __init__(self, filename: str, columns: List[str], write_interval: int = 10):
        self.filename = filename
        # Delete filename if exists
        if os.path.exists(filename):
            logger.info(f"Deleting {filename} as it already exists. Creating bak file.")
            exit_code = os.system(f"mv {filename} {filename}.bak")
            if exit_code != 0:
                logger.warning(f"Failed to create bak file for {filename}. Exit code {exit_code}.")

        # create dirs if not exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.columns = columns
        self.rows: List[Dict[str, Any]] = []
        self.header_written = False
        self.write_interval = write_interval

    def add_row(self, row: Dict[str, Any]):
        self.rows.append(row)
        if len(self.rows) >= self.write_interval:
            self.write()

    def write(self):
        logger.info(f"Writing {len(self.rows)} rows to {self.filename}.")
        with open(self.filename, "a") as f:
            writer = DictWriter(f, fieldnames=self.columns)
            if not self.header_written:
                writer.writeheader()
                self.header_written = True
            writer.writerows(self.rows)
            self.rows = []
