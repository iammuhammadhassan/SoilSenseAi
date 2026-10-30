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

sahiwal_metric = sahiwal.to_crs("EPSG:32643")

area_km2 = sahiwal_metric.geometry.area.iloc[0] / 1_000_000

print("Area (square kilometres):", round(area_km2, 2))

sahiwal.to_file(output_file, driver="GeoJSON")

print("\nSahiwal boundary saved to:", output_file)