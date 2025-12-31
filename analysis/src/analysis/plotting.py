from cartopy import crs as ccrs
from cartopy.io import img_tiles as cimgt
from matplotlib import pyplot as plt
from matplotlib.figure import Figure


def inflate_bounds(
    x_min: int | float,
    x_max: int | float,
    y_min: int | float,
    y_max: int | float,
    buffer: float = 0.1,
) -> tuple[int | float, int | float, int | float, int | float]:
    """Inflate bounding box by a buffer amount. Intended to be used to provide a margin around the area of interest by
    increasing the bounds by a percentage

    :param x_min: minimum value along the x-axis
    :type x_min: int | float
    :param x_max: maximum value along the x-axis
    :type x_max: int | float
    :param y_min: minimum value along the y-axis
    :type y_min: int | float
    :param y_max: maximum value along the y-axis
    :type y_max: int | float
    :param buffer: Buffer amount. Defaults to 0.1.
    :type buffer: float

    Returns:
        tuple[int | float, int | float, int | float, int | float]: Inflated bounding box.
    """
    x_range = x_max - x_min
    y_range = y_max - y_min
    return (
        x_min - x_range * buffer,
        x_max + x_range * buffer,
        y_min - y_range * buffer,
        y_max + y_range * buffer,
    )


def plot_street_map(
    latitude: list[float], longitude: list[float], margin=0.01, title: str | None = None
) -> Figure:
    """
    Plots a street map using OpenStreetMap tiles.

    :param latitude: list of latitudes
    :type latitude: list[float]
    :param longitude: list of longitudes
    :type longitude: list[float]
    :param title: optional title for the plot
    :type title: str | None
    """
    # Define the map extent
    lat_min, lat_max = min(latitude), max(latitude)
    lon_min, lon_max = min(longitude), max(longitude)

    # Create a Stamen Terrain instance
    osm_tiles = cimgt.OSM()

    # Create a figure
    # Create a map using cartopy with OpenStreetMap background
    fig, ax = plt.subplots(
        figsize=(12, 10), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    ax.set_extent(  # ty:ignore[unresolved-attribute]
        [lon_min - margin, lon_max + margin, lat_min - margin, lat_max + margin],
        crs=ccrs.PlateCarree(),
    )  # type: ignore

    # Add the OSM tiles to the map
    ax.add_image(osm_tiles, 12)  # type: ignore

    # Plot the trajectory points
    ax.plot(longitude, latitude, "r.", transform=ccrs.PlateCarree())

    # Add gridlines with labels
    gl = ax.gridlines(draw_labels=True, alpha=0.2)  # type: ignore
    gl.top_labels = False
    gl.right_labels = False

    if title is None:
        ax.set_title("Street Map with Trajectory Points", fontsize=16)
    else:
        ax.set_title(title, fontsize=16)

    return fig
