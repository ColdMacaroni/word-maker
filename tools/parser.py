##
# parser.py
# Takes text files from raw_data converts it into a specific format
# and then puts it into training_data
# parser.py Copyright (C) 2021 Sof
# ^The GPL3 told me to put that there <shrug>

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
    ...


if __name__ == "__main__":
    main("raw_data", "training_data")
