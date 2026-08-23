import os, json, requests
from requests.auth import HTTPBasicAuth

ENDPOINT = "https://fems-qa.fs2c.usda.gov/api/ext-climatology/graphql"
USERNAME = os.environ["FEMS_USERNAME"]
API_KEY  = os.environ["FEMS_API_KEY"]

auth    = HTTPBasicAuth(USERNAME, API_KEY)
headers = {"Content-Type":"application/json","Accept":"application/json"}

q = """query StationMetaData { stationMetaData(returnAll: true, hasHistoricData: ALL) { _metadata { total_count } } }"""

r = requests.post(ENDPOINT, headers=headers, auth=auth, json={"query": q, "variables": None}, timeout=90)
print("HTTP", r.status_code, ENDPOINT)
print("Content-Type:", r.headers.get("content-type"))
print("Body:", r.text[:400])

# Fail loudly with server's exact message so you can see it in the logs
if r.headers.get("content-type","").startswith("application/json"):
    j = r.json()
    if "errors" in j:
        raise RuntimeError(j["errors"])
    print("OK:", j["data"]["stationMetaData"]["_metadata"])
else:
    r.raise_for_status()
