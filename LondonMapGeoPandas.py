from IPython.display import display, Markdown
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import plotly.express as px
import geopandas as gpd
import seaborn as sns
import warnings
import sys
import os
import io

sys.path.append('/kaggle/input/modules/pyfiles/')
from datacleaning import DataCleaning

data = pd.read_csv("/kaggle/input/london-house-price-prediction-advanced-techniques/train.csv")

# Use the class
cleaner = DataCleaning(data)
cleaner.show_info()
cleaner.drop_duplicates()
cleaner.fill_missing(strategy='median', columns=['bathrooms', 'bedrooms', 'livingRooms', 'floorAreaSqM'])
cleaner.fill_missing(strategy='mode', columns=['tenure', 'propertyType', 'currentEnergyRating'])
cleaner.remove_outliers(columns=['price', 'floorAreaSqM', 'bathrooms', 'bedrooms', 'livingRooms', 'floorAreaSqM'])
cleaner.drop_duplicates()
cleaner.show_info()

# Get the cleaned data
df = cleaner.get_clean_data()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.describe()

########################################################################################
## Template 1 plotly 
fig = px.scatter_mapbox(
    df,
    lat='latitude',
    lon='longitude',
    color='price',
    zoom=10,
    mapbox_style="carto-positron"
)
fig.show()

#########################################################################################
## Template 2 
# Your house price data (WGS84 coordinates)
london_boroughs_data = gpd.read_file("/kaggle/input/londongeospatialdata/statistical-gis-boundaries-london/ESRI/London_Borough_Excluding_MHW.shp")
geometry = [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
houses = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")  # WGS84 (lat/lon)
houses['price'] = houses['price']  * (10**-6)

# Transform to match boroughs (EPSG:27700)
houses = houses.to_crs(london_boroughs_data.crs)  # Now in meters

ax = london_boroughs_data.plot(color='grey',
                               #facecolor='none',
                               #cmap='tab20',
                               edgecolor='black',
                               column='NAME',
                               figsize=(12, 10),
                               legend=False,
                               alpha=0.3,
                               legend_kwds={'loc': 'upper right', 'bbox_to_anchor': (1.3, 1)})
houses.plot(
    ax=ax,
    markersize=5,
    column='price',
    cmap='Oranges',
    alpha=0.03,
    figsize=(12, 10),
    legend=True,
    legend_kwds={
        'label': "House Prices",
        'orientation': "vertical",  # Changed to vertical for better readability
        'shrink': 0.5,                # Controls the size of the colorbar
        'aspect': 25,                 # Controls the aspect ratio (thickness) of the colorbar
        'pad': 0.01,                  # Padding from the axes
    }
)

# Add borough names at centroids
for idx, row in london_boroughs_data.iterrows():
    ax.text(
        x=row.geometry.centroid.x,
        y=row.geometry.centroid.y,
        s=row['NAME'],
        fontsize=6,
        ha='center',
        weight='bold',
        bbox=dict(facecolor='white', alpha=0.3, edgecolor='none', pad=0.3)
    )

plt.title("London House Prices WRT Boroughs", pad=20)
plt.axis('off')  # Remove axis
plt.tight_layout()
plt.show()

#########################################################################################
## Template 3
# 2. Make sure both GeoDataFrames have the same CRS (Coordinate Reference System)
houses = houses.to_crs(epsg=4326)
london_boroughs_data = london_boroughs_data.to_crs(epsg=4326)

# 3. Spatial Join: Assign each house to a borough
houses_with_borough = gpd.sjoin(houses, london_boroughs_data, how="inner", predicate="within")

# 4. Group by Borough and calculate Average Price
borough_price = houses_with_borough.groupby('NAME').agg(avg_price=('price', 'mean')).reset_index()

# 5. Merge Average Price back with Borough polygons
london_boroughs_price = london_boroughs_data.merge(borough_price, on='NAME', how='left')

# Plot Static Choropleth
fig, ax = plt.subplots(1, 1, figsize=(14, 12))

# Choropleth
london_boroughs_price.plot(
    column='avg_price',
    cmap='viridis',
    linewidth=0.8,
    ax=ax,
    edgecolor='0.8',
    legend=False,
    legend_kwds={'label': "Average House Price (£)", 'orientation': "vertical"}
)

# Add Borough Labels (NAME at centroid)
for idx, row in london_boroughs_price.iterrows():
    plt.annotate(
        row['NAME'],   # <-- Corrected here (remove s=)
        xy=(row['geometry'].centroid.x, row['geometry'].centroid.y),
        horizontalalignment='center',
        fontsize=7,
        color='black'
    )

# Add the colorbar above the map
cbar = fig.colorbar(
    sm,
    ax=ax,
    orientation='vertical',
    fraction=0.04,   # Adjusts the size of the colorbar
    pad=0.04,        # Adjusts the distance between the colorbar and the plot
    aspect=30        # Adjusts the aspect ratio of the colorbar
)

ax.set_title("London Boroughs - Average House Prices", fontdict={'fontsize': 18})
ax.axis('off')
plt.show()

#########################################################################################
#Template 4
#1
london_boroughs_data = gpd.read_file("/kaggle/input/londongeospatialdata/statistical-gis-boundaries-london/ESRI/London_Borough_Excluding_MHW.shp")
geometry = [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
houses = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")  # WGS84 (lat/lon)
houses['price'] = houses['price']  * (10**-6)

# 2. Make sure both GeoDataFrames have the same CRS (Coordinate Reference System)
houses = houses.to_crs(epsg=4326)
london_boroughs_data = london_boroughs_data.to_crs(epsg=4326)

# 3. Spatial Join: Assign each house to a borough
houses_with_borough = gpd.sjoin(houses, london_boroughs_data, how="inner", predicate="within")

# 4. Group by Borough and calculate Average Price
borough_price = houses_with_borough.groupby('NAME').agg(avg_price=('price', 'mean')).reset_index()

# 5. Merge Average Price back with Borough polygons
london_boroughs_price = london_boroughs_data.merge(borough_price, on='NAME', how='left')

# Plot Static Choropleth
fig, ax = plt.subplots(1, 1, figsize=(9.5, 9))

# Choropleth
london_boroughs_price.plot(
    column='avg_price',
    cmap='viridis',
    linewidth=0.8,
    ax=ax,
    edgecolor='0.8',
    legend=False,
    legend_kwds={'label': "Average House Price (£)", 'orientation': "vertical"}
)

# Add Borough Labels (NAME at centroid)
for idx, row in london_boroughs_price.iterrows():
    plt.annotate(
        row['NAME'],   # <-- Corrected here (remove s=)
        xy=(row['geometry'].centroid.x, row['geometry'].centroid.y),
        horizontalalignment='center',
        fontsize=7,
        color='black'
    )

# Create a ScalarMappable for the colorbar
norm = mpl.colors.Normalize(
    vmin=london_boroughs_price['avg_price'].min(),
    vmax=london_boroughs_price['avg_price'].max()
)
sm = mpl.cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])  # Only needed for older versions of Matplotlib


# Add the colorbar above the map
cax = inset_axes(ax, width="10%", height="5%", loc='upper left',
                 bbox_to_anchor=(0.8, 0.46, 3, 0.5), bbox_transform=ax.transAxes, borderpad=0)

# Add the colorbar to the new axis
cbar = fig.colorbar(sm, cax=cax, orientation='horizontal', 
                    fraction=0.04,   # Adjusts the size of the colorbar
                    pad=0.04,        # Adjusts the distance between the colorbar and the plot
                    aspect=50        # Adjusts the aspect ratio of the color
                   )

cbar.set_label('Average House Price in Millions (£)', fontsize=8)

# --- KEY CHANGE: Move everything right by adjusting axis position ---
# Original position: [left, bottom, width, height] in figure coordinates (0-1)
# Shift right by reducing width and increasing left offset
ax.set_position([0.98, 0.5, 0.7, 0.8])  # Experiment with these values

ax.set_title("London Boroughs - Average House Prices")
ax.axis('off')
plt.show()


#########################################################################################
# Template 5
# Prepare for Plotly
london_boroughs_price['centroid_lon'] = london_boroughs_price.geometry.centroid.x
london_boroughs_price['centroid_lat'] = london_boroughs_price.geometry.centroid.y

# Interactive Map
fig = px.choropleth_mapbox(
    london_boroughs_price,
    geojson=london_boroughs_price.geometry,
    locations=london_boroughs_price.index,
    color="avg_price",
    hover_name="NAME",
    hover_data={"avg_price": ":,.0f"},
    mapbox_style="carto-positron",
    center={"lat": 51.5072, "lon": -0.1276},
    zoom=9,
    opacity=0.7,
    color_continuous_scale="Viridis",
)

fig.update_layout(
    title_text="London Boroughs - Average House Prices (Interactive)",
    margin={"r":0,"t":30,"l":0,"b":0},
)
fig.show()
