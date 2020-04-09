CHROME_BINARY_LOCATION = "/usr/bin/google-chrome"
CHROME_ADBLOCK_LOCATION= "/home/ec2-user/Downloads/extension_3_7_0_0.crx" # note - couldn't get this to work
FIREFOX_BINARY_LOCATION = "/usr/bin/firefox"
FIREFOX_PROFILE_LOCATION = "./profiles"
ERROR_REPORT_DIR = "./error_reports"
METADATA_DIR = "./metadata"
KNOWN_IP_LIST_FN = "known_ips.pkl"
TRACES_DIR = "./throughput_traces/traces"
PCAP_DIR = "./pcaps"
LOG_DIR = "./logs"

# IPs internal to the observed network
INTERNAL_IPS = ["172.31.30.176"]
ASN_LIST = ["GOOGLE", "NETFLIX-ASN", "AMAZON-AES", "AKAMAI-AS", "FACEBOOK", "JUSTINTV", "OTHER"]
ASN_LIST = {k:i for i, k in enumerate(ASN_LIST)}
MAX_TIME = 120
NETFLIX_PROFILE_INDEX = 3 # number in the list of netflix profiles on the accouont you should use (zero indexed)
T_INTERVAL = 1
