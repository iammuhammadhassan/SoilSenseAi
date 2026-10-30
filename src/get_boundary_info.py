
import requests

url = "https://www.geoboundaries.org/api/current/gbOpen/PAK/ADM2/"

response = requests.get(url, timeout=60)
response.raise_for_status()

boundary = response.json()

print("Boundary dataset:", boundary["boundaryName"])
print("Administrative level:", boundary["boundaryType"])
print("Year:", boundary["boundaryYearRepresented"])
print("Number of districts:", boundary["admUnitCount"])
print("Source:", boundary["boundarySource"])
print("\nGeoJSON download URL:")
print(boundary["gjDownloadURL"])
