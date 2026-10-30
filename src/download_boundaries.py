
import requests
import geopandas as gpd
from pathlib import Path

url = "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/PAK/ADM2/geoBoundaries-PAK-ADM2.geojson"

output = Path("data/raw/pakistan_districts.geojson")

print("Downloading district boundaries...")

response = requests.get(url, timeout=120)
response.raise_for_status()

output.write_bytes(response.content)

print("Download complete!")

districts = gpd.read_file(output)

print("\nDataset shape:", districts.shape)
print("\nAvailable columns:")
print(districts.columns.tolist())

print("\nSearching for Sahiwal...")

for column in districts.select_dtypes(include="object").columns:
    matches = districts[
        districts[column].astype(str).str.contains(
            "Sahiwal", case=False, na=False
        )
    ]

    if not matches.empty:
        print("\nMatched column:", column)
        print(matches.drop(columns="geometry").to_string(index=False))
