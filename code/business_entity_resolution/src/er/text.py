"""Deterministic text transforms; no downloaded models or business lookups."""
import hashlib
import re
import unicodedata
from functools import lru_cache
from anyascii import anyascii

SUFFIXES = frozenset("inc incorporated corp corporation llc ltd limited pvt private co company llp plc sarl sas sa eurl".split())
ADDRESS_WORDS = {"street": "st", "road": "rd", "avenue": "ave", "boulevard": "blvd",
                 "lane": "ln", "drive": "dr", "floor": "fl", "building": "bldg",
                 "apartment": "apt", "suite": "ste"}


def unicode_text(value):
    # Preserve Indic combining marks in the original-script view.
    value = unicodedata.normalize("NFKC", value or "").casefold().replace("&", " and ")
    return " ".join("".join(c if c.isalnum() or unicodedata.category(c).startswith("M") else " "
                            for c in value).split())


def normalize(value, address=False):
    words = re.findall(r"[a-z0-9]+", anyascii(unicode_text(value)).lower())
    if address:
        words = [ADDRESS_WORDS.get(w, w) for w in words]
    return " ".join(words)


def core_name(name):
    words = [w for w in name.split() if w not in SUFFIXES and w != "and"]
    return " ".join(sorted(words or name.split()))


@lru_cache(maxsize=100000)
def phonetic(name):
    result = set()
    for word in name.split():
        if len(word) < 3 or word in SUFFIXES:
            continue
        word = word.replace("ph", "f").replace("sh", "s").replace("ch", "s")
        word = word.translate(str.maketrans({"q": "k", "c": "k", "w": "v", "z": "s"}))
        word = re.sub(r"[aeiouyh]", "", word)
        word = re.sub(r"(.)\1+", r"\1", word)
        if len(word) >= 2:
            result.add("p" + word[:6])
    return " ".join(sorted(result))


def country_key(country):
    return "c" + hashlib.blake2s(country.strip().casefold().encode(), digest_size=12).hexdigest()


def record(row):
    name = normalize(row["business_name"])
    address = normalize(row["business_address"], address=True)
    return {"eid": row["entity_id"], "country": country_key(row["country"]),
            "name": name, "address": address, "core": core_name(name),
            "uname": unicode_text(row["business_name"]),
            "uaddress": unicode_text(row["business_address"])}


def fold(eid, seed=42):
    number = int.from_bytes(hashlib.blake2b(f"{seed}:{eid}".encode(), digest_size=8).digest(), "little")
    # Separate model fitting, threshold tuning, and untouched evaluation entities.
    return ("fit", "tune", "eval")[0 if number % 10 < 6 else 1 if number % 10 < 8 else 2]
