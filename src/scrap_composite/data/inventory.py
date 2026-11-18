"""Shared eligibility rules for the current object inventory."""

# 38 and 39 are storage pressure vessels in the source catalog, despite the
# oil-collector names used by the reconstruction library.
STORAGE_TANK_IDS = frozenset({'5', '6', '7', '38', '39'})


def is_storage_tank(number):
    return str(number).lstrip('0') in STORAGE_TANK_IDS
