"""
Tests for the map margin around each trajectory's bounding box.

The map has to extend past the recorded track by however far the *filter* can wander off it,
or a geophysical update lands off-map and is skipped. Under `conf/*_denied.toml` GNSS is
withheld for 120 s at a time and the solution runs unaided throughout, which is the first
scenario that reaches the edge.

The failure is silent at the preprocessing end -- the map is written successfully, and the
run dies later, per trajectory, on `OutOfMapBounds`. So these are the checks that have to
hold here.
"""

from __future__ import annotations

import math

import pytest
from analysis.plotting import inflate_bounds
from analysis.preprocess import DEFAULT_MAP_MARGIN_KM, METRES_PER_DEGREE, pad_bounds

# Philadelphia-ish, matching the recordings this repo was built around.
LAT = 40.05
LON = -75.95


def _km_span(lo: float, hi: float, latitude: float = LAT) -> float:
    """Longitude span in kilometres at a given latitude."""
    return (hi - lo) * METRES_PER_DEGREE * math.cos(math.radians(latitude)) / 1000.0


def _km_span_lat(lo: float, hi: float) -> float:
    return (hi - lo) * METRES_PER_DEGREE / 1000.0


def test_a_due_north_track_still_gets_longitude_margin():
    """
    The case `--buffer` alone cannot express.

    `inflate_bounds` scales each axis by *its own* range, so a track that is straight in one
    axis gets a pad of `0 * buffer` in it -- zero, for any buffer. A due-north drive therefore
    produces a map with no longitude width at all, and the first metre of eastward drift
    leaves it.
    """
    lon_min = lon_max = LON
    lat_min, lat_max = LAT, LAT + 0.15

    fractional = inflate_bounds(lon_min, lon_max, lat_min, lat_max, 0.1)
    assert _km_span(fractional[0], fractional[1]) == pytest.approx(0.0, abs=1e-9), (
        "precondition: the fractional pad gives a due-north track zero longitude width"
    )

    padded = pad_bounds(
        lon_min, lon_max, lat_min, lat_max, buffer=0.1, margin_km=DEFAULT_MAP_MARGIN_KM
    )
    assert _km_span(padded[0], padded[1]) == pytest.approx(2 * DEFAULT_MAP_MARGIN_KM, rel=0.02)


def test_a_due_east_track_still_gets_latitude_margin():
    """The same degeneracy in the other axis."""
    padded = pad_bounds(LON, LON + 0.15, LAT, LAT, buffer=0.1, margin_km=DEFAULT_MAP_MARGIN_KM)
    assert _km_span_lat(padded[2], padded[3]) == pytest.approx(2 * DEFAULT_MAP_MARGIN_KM, rel=0.02)


def test_a_short_track_gets_the_absolute_margin_not_a_fraction_of_nothing():
    """
    A 2 km track at a 10% buffer gets 200 m of margin, which 120 s of unaided drift clears
    easily. The absolute floor is what makes a short recording usable.
    """
    two_km_deg = 2.0 * 1000.0 / (METRES_PER_DEGREE * math.cos(math.radians(LAT)))
    padded = pad_bounds(
        LON, LON + two_km_deg, LAT, LAT + 0.02, buffer=0.1, margin_km=DEFAULT_MAP_MARGIN_KM
    )
    # 2 km of track plus 5 km on each side.
    assert _km_span(padded[0], padded[1]) == pytest.approx(
        2.0 + 2 * DEFAULT_MAP_MARGIN_KM, rel=0.02
    )


def test_a_long_track_keeps_the_larger_fractional_margin():
    """
    The absolute margin is a floor, not a replacement. Where the fraction is already more
    generous it wins, so this does not shrink anyone's existing maps.
    """
    padded = pad_bounds(-76.5, -75.0, 40.0, 41.0, buffer=0.5, margin_km=DEFAULT_MAP_MARGIN_KM)
    fractional = inflate_bounds(-76.5, -75.0, 40.0, 41.0, 0.5)
    assert padded == fractional


def test_zero_margin_is_exactly_the_old_behaviour():
    """`--margin-km 0` opts back out, byte for byte."""
    assert pad_bounds(LON, LON + 0.05, LAT, LAT + 0.05, buffer=0.1, margin_km=0.0) == (
        inflate_bounds(LON, LON + 0.05, LAT, LAT + 0.05, 0.1)
    )


@pytest.mark.parametrize("latitude", [0.0, 40.0, 70.0, 85.0])
def test_the_absolute_margin_is_the_same_distance_at_every_latitude(latitude: float):
    """
    A degree of longitude shortens towards the poles, so a fixed number of kilometres is more
    degrees the further north you are. Converting at the box's mean latitude is what keeps the
    margin a distance rather than an angle.
    """
    padded = pad_bounds(LON, LON, latitude, latitude, buffer=0.0, margin_km=5.0)
    assert _km_span(padded[0], padded[1], latitude) == pytest.approx(10.0, rel=0.02)


def test_the_conversion_does_not_diverge_at_the_pole():
    """`cos()` reaches zero at the pole; the clamp keeps the bounds finite."""
    padded = pad_bounds(LON, LON, 90.0, 90.0, buffer=0.0, margin_km=5.0)
    assert all(math.isfinite(v) for v in padded)
