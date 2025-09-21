import re
VALID = [
 r"^go to \w+( \d+)?$", r"^open \w+( \d+)?$", r"^close \w+( \d+)?$",
 r"^take \w+( \d+)? from \w+( \d+)?$", r"^(put|place) \w+( \d+)? (in|on) \w+( \d+)?$",
 r"^use \w+( \d+)?$", r"^heat \w+( \d+)? with microwave 1$", r"^cool \w+( \d+)? with fridge 1$",
 r"^examine \w+( \d+)?$", r"^look$", r"^inventory$"
]
def guard(s:str)->str:
    a=s.strip().splitlines()[0].lower()
    return a if any(re.match(p,a) for p in VALID) else "look"
