import geopandas as gpd
from pathlib import Path

input_file = Path("data/raw/pakistan_districts.geojson")
output_file = Path("data/processed/sahiwal_boundary.geojson")

print("Loading Pakistan district boundaries...")

districts = gpd.read_file(input_file)

sahiwal = districts[
    districts["shapeName"].str.strip().str.lower() == "sahiwal"
].copy()

if len(sahiwal) != 1:
    raise ValueError(f"Expected 1 Sahiwal boundary, found {len(sahiwal)}")

print("Sahiwal boundary found!")

print("Coordinate Reference System:", sahiwal.crs)
print("Geometry valid:", sahiwal.geometry.is_valid.all())
print("Area (square degrees):", sahiwal.geometry.area.iloc[0])
print("Geographic bounds:", sahiwal.total_bounds)

sahiwal.to_file(output_file, driver="GeoJSON")

print("\nSahiwal boundary saved to:", output_file)