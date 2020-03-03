CHROME_BINARY_LOCATION = "/usr/bin/google-chrome"
CHROME_ADBLOCK_LOCATION= "/home/ec2-user/Downloads/extension_3_7_0_0.crx" # note - couldn't get this to work
ERROR_REPORT_DIR = "./error_reports"

# IPs internal to the observed network
INTERNAL_IPS = ["172.31.38.155"]
ASN_LIST = ["GOOGLE", "NETFLIX-ASN", "AMAZON-AES", "AKAMAI-AS", "FACEBOOK", "JUSTINTV", "OTHER"]
ASN_LIST = {k:i for i, k in enumerate(ASN_LIST)}
MAX_TIME = 120
NETFLIX_PROFILE_INDEX = 3 # number in the list of netflix profiles on the accouont you should use (zero indexed)
