
## Getting Started
This repository contains scripts for Video QoE Analysis.
The `trace_generation` folder will provide scripts to create packet traces for  different video streaming services based on varying network conditions.
The `spark_analysis` folder provides pyspark queries that can be run over these packet traces.

## Installation
`trace_generation` requirements:
- installation of [Mininet](http://mininet.org/download/) to run.

`spark_analysis` requirements:
- installation of [tshark](https://linoxide.com/how-tos/howto-install-wireshark-on-linux/)
- installation of [pyspark](https://www.roseindia.net/bigdata/pyspark/install-pyspark-on-ubuntu.shtml)