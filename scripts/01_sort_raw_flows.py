#!/usr/bin/env python3
"""Etape [1] : tri chronologique des exports Zeek bruts (data/raw/part-*.csv)."""

from ids_pipeline.preprocessing.sort_flows import main

if __name__ == "__main__":
    main()
