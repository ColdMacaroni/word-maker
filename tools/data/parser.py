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


def translator(string, t_dict):
    """
    Translates strings into a normalized numbers for the NN
    "and             " -> "0, 0.5, 0.11538461538461539, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1"
    'a' = 0                   == (ord('a') - ord('a')) / 26
    'n' = 0.5                 == (ord('n') - ord('a')) / 26
    'd' = 0.11538461538461539 == (ord('d') - ord('a')) / 26
    ' ' = 1

    :param string: The string to be translated
    :param t_dict: Dictionary to follow for translations
    """
    new_chars = list()

    for char in string:
        # Have to convert into str to be able to use .join
        new_chars.append(str(t_dict[char.lower()]))

    return ','.join(new_chars)


def main(raw_folder, parsed_folder, str_len=16, sep_char=';'):
    """
    Parses data from all text files in the raw folder into a format of
    string{sep_char}length
    e.g. (when size=16 and sep_char=';'):
        Input file:
            "Macaroni\\n"

        Turns into:
            "Macaroni        ;8"

        Output file (Only one newline):
            "0.46153846153846156, 0.0, 0.07692307692307693,
             0.0, 0.6538461538461539, 0.5384615384615384,
             0.5, 0.3076923076923077, 1, 1, 1, 1, 1, 1, 1, 1;8\\n"

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

    # Generate dictionary for translations.
    # Each letter is a normalized number between 0 and 26,
    # 26 being a space.
    # 26 is not a magic #, its the # of letters in the alphabet. ok?
    tr_dict = dict()
    for i in range(0, 26):
        # This will go a-z
        letter = chr(ord('a') + i)

        tr_dict[letter] = i/26

    tr_dict[' '] = 1.0

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

        tr_words = [translator(word, tr_dict) for word in padded_words]

        # Put the lengths and padded words together for easier writing
        # I couldnt think of a better name
        words_n_lens = zip(tr_words, word_lens)

        # I'm not sure how the / works but this is the same as saying
        # output_file = Path(f"{output_path}/{file.name}")
        output_file = output_path / file.name

        # Clean the file. 'w' mode empties the file by default.
        if output_file.exists():
            with output_file.open('w'):
                pass

        with output_file.open('a') as output:
            for word, length in words_n_lens:
                output.write(f"{word}{sep_char}{length}\n")


if __name__ == "__main__":
    main("raw_data", "training_data")
