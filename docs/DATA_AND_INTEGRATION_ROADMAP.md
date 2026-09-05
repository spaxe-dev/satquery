# Data & Integration Roadmap

- MUST (SIH): BigEarthNet train manifests real (fix EuroSAT copy-paste), reBEN materialize or drop claim, VRSBench/RSVQA/CDVQA eval harness (scaffold `download_datasets.py --list` only now).
- SHOULD: STAC catalog read (Sentinel-1/2), COG windowed-read, OSM overlay (context only), reports DB.
- FUTURE: weather/rainfall/disaster feeds, elevation, land-cover, temporal catalog.
- DISTRACTION: live satellite tasking, multi-user auth/SSO, billing, mobile app.

Each integration needs WHY/user-problem/contract/cost/privacy/SIH-value/now? Only STAC + eval harness are SIH-critical now.
