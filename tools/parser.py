#!/usr/bin/env python3
# parser.py
# Takes text files from raw_data converts it into a specific format
# and then puts it into training_data
# parser.py Copyright (C) 2021 Sof
# ^The GPL3 told me to put that there <shrug>

from pathlib import Path


def get_line_list(file_path: Path):
    """
    Get a list of each single line, stripped.
    :param file_path: A Path object of the file
    :return: List
    """
    lines = list()

    with file_path.open('r') as file:
        # Loop until a line without nothing ('') shows up. (EOF)
        while line := file.readline():
            lines.append(line.strip())

    return lines


def main(raw_folder, parsed_folder, size=16):
    """
    Parses data from all text files in the raw folder into a format of
    string\tlength
    e.g. (when size=16):
        Input file:
            "Macaroni\n"

        Output file:
            "Macaroni        \t8"

    :param raw_folder:
    :param parsed_folder:
    :param size: The length to pad the strings to
    """
    # Loop through all files, parsing them and writing to parsed_folder
    # with the same filename
    raw_folder_path = Path(raw_folder)

    # Make sure we get actual files
    raw_files = [f for f in raw_folder_path.iterdir() if f.is_file()]

    for file in raw_files:
        print(get_line_list(file))


if __name__ == "__main__":
    main("raw_data", "training_data")
