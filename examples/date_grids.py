"""
Programmatic forecast-origin schedules for the NZ retrospective experiments.

These grids replace duplicated hardcoded timestamp lists. The rules are kept
explicit so the experiment design remains easy to review and adjust.
"""

from __future__ import annotations

from datetime import datetime, timedelta


def _generate_regular_grid(start: datetime, end: datetime,
                           step: timedelta) -> list[datetime]:
    """Generate an inclusive regular grid."""
    dates = []
    current = start
    while current <= end:
        dates.append(current)
        current += step
    return dates


def _generate_weekday_grid(start: datetime, end: datetime,
                           weekdays: set[int]) -> list[datetime]:
    """Generate an inclusive daily scan filtered to selected weekdays."""
    dates = []
    current = start
    while current <= end:
        if current.weekday() in weekdays:
            dates.append(current)
        current += timedelta(days=1)
    return dates


def _merge_unique(*segments: list[datetime]) -> list[datetime]:
    """Merge sorted date segments and drop accidental duplicates."""
    merged = sorted({date for segment in segments for date in segment})
    return merged


def generate_kaikoura_dates() -> list[datetime]:
    """
    Forecast-origin schedule for the Kaikoura sequence.

    Design:
    - Daily noon checkpoints immediately after the mainshock.
    - 12-hour updates during the most rapidly evolving early period.
    - Daily noon updates through the remainder of the first dense phase.
    - Wednesday/Sunday checkpoints through the transition period.
    - Weekly Sunday checkpoints for the long tail.
    """
    return _merge_unique(
        _generate_regular_grid(
            datetime(2016, 11, 13, 12, 0, 0),
            datetime(2016, 11, 15, 12, 0, 0),
            timedelta(days=1),
        ),
        _generate_regular_grid(
            datetime(2016, 11, 16, 0, 0, 0),
            datetime(2016, 11, 19, 12, 0, 0),
            timedelta(hours=12),
        ),
        _generate_regular_grid(
            datetime(2016, 11, 20, 12, 0, 0),
            datetime(2016, 12, 2, 12, 0, 0),
            timedelta(days=1),
        ),
        _generate_weekday_grid(
            datetime(2016, 12, 7, 12, 0, 0),
            datetime(2017, 1, 1, 12, 0, 0),
            weekdays={2, 6},  # Wednesday, Sunday
        ),
        _generate_regular_grid(
            datetime(2017, 1, 8, 12, 0, 0),
            datetime(2017, 4, 2, 12, 0, 0),
            timedelta(days=7),
        ),
    )


def generate_canterbury_dates() -> list[datetime]:
    """
    Forecast-origin schedule for the Canterbury sequence.

    Design:
    - Daily 17:00 checkpoints across the first two weeks after the mainshock.
    """
    return _generate_regular_grid(
        datetime(2010, 9, 3, 17, 0, 0),
        datetime(2010, 9, 16, 17, 0, 0),
        timedelta(days=1),
    )


KAIKOURA_DATES = generate_kaikoura_dates()
CANTERBURY_DATES = generate_canterbury_dates()
SEQUENCE_DATE_GRIDS = {
    "Kaikoura": KAIKOURA_DATES,
    "Canterbury": CANTERBURY_DATES,
}

SEQUENCE_DATE_GRID_SUMMARIES = {
    "Kaikoura": [
        "Daily noon checkpoints for November 13-15, 2016",
        "12-hour checkpoints for November 16-19, 2016",
        "Daily noon checkpoints for November 20 to December 2, 2016",
        "Wednesday/Sunday noon checkpoints for December 7, 2016 to January 1, 2017",
        "Weekly Sunday noon checkpoints for January 8 to April 2, 2017",
    ],
    "Canterbury": [
        "Daily 17:00 checkpoints for September 3-16, 2010",
    ],
}

SEQUENCE_DATE_GRID_METADATA = {
    sequence: {
        "n_forecast_origins": len(SEQUENCE_DATE_GRIDS[sequence]),
        "first_forecast_origin": SEQUENCE_DATE_GRIDS[sequence][0].strftime("%Y-%m-%d %H:%M:%S"),
        "last_forecast_origin": SEQUENCE_DATE_GRIDS[sequence][-1].strftime("%Y-%m-%d %H:%M:%S"),
        "design": SEQUENCE_DATE_GRID_SUMMARIES[sequence],
    }
    for sequence in SEQUENCE_DATE_GRIDS
}
