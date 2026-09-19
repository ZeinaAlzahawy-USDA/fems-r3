import os

from arcgis.gis import GIS


PORTAL_URL = "https://www.arcgis.com"


def main():
    client_id = os.environ["AGOL_CLIENT_ID"]
    client_secret = os.environ["AGOL_CLIENT_SECRET"]

    print("Connecting to ArcGIS Online with OAuth app authentication...")
    gis = GIS(PORTAL_URL, client_id=client_id, client_secret=client_secret)

    print("Connection successful.")
    print(f"Organization: {gis.properties.name}")
    print("No ArcGIS content was created or changed.")


if __name__ == "__main__":
    main()
