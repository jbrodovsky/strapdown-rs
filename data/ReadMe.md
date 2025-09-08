# Data

This directory contains the data files used in the project. The files are organized by their respective sources, processing type, and frequency.

## Structure

- `raw/`: Contains raw data files as recorded from the source. These files are not tracked in version control, due to their size but the folder structure is maintained.
- `input/`: Contains the pre-processed input files that are ready for processing as well as a map that highlights the route.
- `truth/`: Contains the ground truth data files of processed trajectories that are used for validation and testing. These files should be used to compare against alternative processing methods and navigation algorithms.

The following folders contain processed trajectory data files that simulate various GPS conditions that are less than ideal:

- `degraded/`: Contains trajectory data files that simulate degraded GPS conditions (i.e. less accurate GPS data).
- `spoofed/`: Contains trajectory data files that simulate GPS spoofing conditions (i.e. GPS data that has been altered to mislead the navigation system and contains a fixed offset).
- `intermittent/`: Contains trajectory data files that simulate intermittent GPS conditions (i.e. GPS data that is not available for certain periods of time).
- `combo/`: Contains trajectory data files that simulate a combination of degraded, spoofed, and intermittent GPS conditions.

Below each root data folder is a set of directories indicating the source frequency of the data recorded. The original frequency recorded was at 10 Hz. These have been down-sampled respectively to 5 Hz, 2 Hz, and 1 Hz. Any additional processing (calculation of truth, degradation, etc.) should be applied within these frequency-specific folders. In other words, an intermittent GPS-denied scenario could have a GPS measurement applied every 60 seconds. Thus this folder structure would look like

```plaintext
intermittent
-- 1Hz
    -- 60s
-- 2Hz
    -- 60s
-- 5Hz
    -- 60s
-- 10Hz
    -- 60s
```
