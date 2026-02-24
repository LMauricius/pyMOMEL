import pyMOMEL
from pyMOMEL import Number


def pprint(val, multiline: bool, indent=0):
    """Pretty print parsed MOMEL data as aligned Python objects."""
    pad = "  " * indent
    end = "\n" if multiline else ""

    text: list[str] = []

    if isinstance(val, dict):
        if not val:
            text.append("{}")
        else:
            w = max(len(repr(k)) for k in val)
            text.append("{" + end)
            for k, v in val.items():
                if multiline:
                    text.append(pad + "  ")
                text.append(f"{repr(k) + ':':<{w + 1}} ")
                if is_simple(v):
                    text.append(pprint(v, False) + ", " + end)
                else:
                    text.append(pprint(v, multiline, indent + 2) + ", " + end)
            text.append(pad + "}")
    elif isinstance(val, list):
        if not val:
            text.append("[]")
        else:
            text.append("[" + end)
            for item in val:
                if multiline:
                    text.append(pad + "  ")
                if is_simple(item):
                    text.append(pprint(item, False) + ", " + end)
                else:
                    text.append(pprint(item, multiline, indent + 1) + ", " + end)
            text.append(pad + "]")
    elif isinstance(val, tuple):
        if len(val) == 0:
            text.append("()")
        else:
            text.append("(" + end)
            for item in val:
                if multiline:
                    text.append(pad + "  ")
                if is_simple(item):
                    text.append(pprint(item, False) + ", " + end)
                else:
                    text.append(pprint(item, multiline, indent + 1) + ", " + end)
            text.append(pad + ")")
    elif isinstance(val, Number):
        text.append(repr(val.value) + val.unit)
    else:
        text.append(repr(val))

    return "".join(text)


def is_simple(val):
    """True if val can be printed on one line."""
    if isinstance(val, (Number, str, int, float)):
        return True
    if isinstance(val, tuple):
        return all(is_simple(v) for v in val) and len(pprint(val, False)) < 80
    if isinstance(val, list):
        return all(is_simple(v) for v in val) and len(pprint(val, False)) < 80
    return False


TESTS = [
    """
# This file contains some data...
# Edit it as needed!

Record title: "Users that visited us on this beautiful day"
Metadata: {
    Date: YMD 2025 2 14
    Time: HM 04 32 pm
}
Users: [
    {
        Name:            "John Smith"
        Weight:          75kg
        Distance:        5km
        Duration:        1h 6min 46s
        Math result:     6.02_+23
        Favourite color: rgb 65 127 240
        Scores: [
            9.8
            8.5
            9.1
        ]
        Remark: "
                 Was fun.
                 Would do it again.
                "
    }
    {
        Name:            "Alice Johnson"
        Weight:          60kg
        Distance:        8.3km
        Duration:        1h 45min 12s
        Math result:     3.14
        Favourite color: rgb 240 98 146
        Scores: [
            9.2
            8.9
            9.6
        ]
        Remark: "
                 Enjoyed the scenic route.
                 Planning to bring friends next time.
                "
    }
]
Today's mentioned trivia: {
    Mass of Earth: 5.97219_+24kg
    Tallest statue: "Statue of Unity"
    Most remote inhabited archipelago: "Tristan da Cunha"
}
""",
]

for testStr in TESTS:
    print("==== TEST ================================================================")
    print(testStr)
    print("--------------------------------------------------------------------------")
    try:
        res = pyMOMEL.parse(testStr)
        print(pprint(res, True))
    except Exception as e:
        print(repr(e))
