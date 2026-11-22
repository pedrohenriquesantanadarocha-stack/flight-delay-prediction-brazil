"""Pipelines: orchestrate domain logic and adapters, layer by layer.

    etl       ANAC VRA -> 01_raw -> 02_intermediate -> 03_primary (flights)
              NOAA GSOD -> 03_primary (weather)
    features  03_primary -> 04_feature (every flight) -> 05_model_input (sample)
    modeling  05_model_input -> train, evaluate, diagnose -> 07_model_output, 08_reporting

Every step skips outputs that already exist, so an interrupted run resumes where
it stopped. Pass force=True to recompute.
"""
