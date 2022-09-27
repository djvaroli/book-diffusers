import typing as t


def line_by_line_generator(filepath: str, mode: str = "r+") -> t.Generator[str, None, None]:
    """Returns a generator of sentences from the source file.

    Args:
        filepath (str): source file with text.

    Returns:
        t.Generator[str]: generator that yields individual sentences from
            the specified file.
    """
    with open(filepath, mode) as f:
        for line in f:
            yield line


def read_whole_file(filepath: str, mode: str = "r+") -> str:
    """Reads the contents of the entire file in-memory.

    Args:
        filepath (str): path to file.
        mode (str, optional): mode to open file in. Defaults to "r+".

    Returns:
        str: contents of file as a single string.
    """
    with open(filepath, mode) as f:
        content = f.read()
    return content


def split_into_sentences(contents: str, sep: str = ".") -> t.List[str]:
    """Splits contents of a given file into individual sentences using specified separator.

    Args:
        contents (str): file contents.
        sep (str, optional): character to split contents on. Defaults to ".".

    Returns:
        t.List[str]: list of sentences making up `contents`.
    """
    return contents.split(sep)
