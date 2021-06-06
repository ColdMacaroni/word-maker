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


def pad_right(string, length, pad_char=' '):
    """
    Pads the string to the given length, can specify the character
    :param string: A string
    :param length: An int of the desired length
    :param pad_char: The character to pad with. Default = ' '
    :return: A string. len(new string) == length
    """
    new_string = string

    # In case the string is already that size
    if len(new_string) < length:
        # Add the necessary amount of spaces
        new_string += pad_char * (length - len(string))

    # >= In case the string was bigger than the length
    assert len(new_string) >= length, f"String is {len(new_string)}, not {length}!"

    return new_string


def main(raw_folder, parsed_folder, str_len=16, sep_char=';'):
    """
    Parses data from all text files in the raw folder into a format of
    string{sep_char}length
    e.g. (when size=16 and sep_char=';'):
        Input file:
            "Macaroni\n"

        Output file:
            "Macaroni        ;8"

    :param raw_folder:
    :param parsed_folder:
    :param str_len: The length to pad the strings to
    :param sep_char: The character used to separate the word and its
                     length
    """
    # Loop through all files, parsing them and writing to parsed_folder
    # with the same filename
    input_path = Path(raw_folder)
    output_path = Path(parsed_folder)

    # Make sure we get actual files
    raw_files = [f for f in input_path.iterdir() if f.is_file()]

    for file in raw_files:
        # Obtain lines. (Each should be a word!)
        all_words = get_line_list(file)

        # Filter out those bigger than the specified size
        words = [w for w in all_words if len(w) <= str_len]

        # Store the length *before* padding
        word_lens = [len(word) for word in words]

        # Pad words to str_len
        padded_words = [
            pad_right(word, str_len, pad_char=' ') for word in words
        ]

        # Put the lengths and padded words together for easier writing
        # I couldnt think of a better name
        words_n_lens = zip(padded_words, word_lens)

        # I'm not sure how the / works but this is the same as saying
        # output_file = Path(f"{output_path}/{file.name}")
        output_file = output_path / file.name

        # Clean the file. 'w' mode empties the file by default.
        if output_file.exists():
            with output_file.open('w'):
                pass

        with output_file.open('a') as output:
            for word, length in words_n_lens:
                output.write(f"{word}{sep_char}{length}")


if __name__ == "__main__":
    main("raw_data", "training_data")
