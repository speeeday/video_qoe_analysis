## Usage

Convert the PCAP into a table to run Spark queries over

`python convert_pcap.py <input_pcap> <output_file>`

NOTE: If there are any fields that you want to extract and run queries over, they need to be included in the list in `fields.py`

To run spark queries, simply run a python file and provide the data file to execute the query over

For example, using the same `<output_file>` that we converted we can run

`python identify_video_spark.py <output_file>`

### IP Protocol Numbers

The IP Protocol shows up only as a number unfortunately and not a string. Using the `proto` script, you can provide a number or protocol to get the its respective mapping

Example: `./proto 16`
Example: `./proto ipv4`


