import os

from arcgis.gis import GIS


PORTAL_URL = "https://www.arcgis.com"


def main():
    username = os.environ["AGOL_USERNAME"]
    password = os.environ["AGOL_PASSWORD"]

    print("Connecting to ArcGIS Online...")
    gis = GIS(PORTAL_URL, username, password)

    print("Connection successful.")
    print(f"Signed in as: {gis.users.me.username}")
    print(f"Organization: {gis.properties.name}")
    print("No ArcGIS content was created or changed.")


if __name__ == "__main__":
    main()
