import unidecode

def title_parser(title):
    title = unidecode.unidecode(title)
    return title