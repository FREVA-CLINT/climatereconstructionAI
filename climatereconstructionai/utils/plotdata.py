def get_longname(var):
    if var in ("tas", "temperature_anomaly", "tas_mean", "temperature_anomaly_mean"):
        lname = "Near surface temperature anomaly (°C)"
    else:
        lname = ""

    return lname


def plot_data(coords, data, titles, output_name, data_type, time_label, vlim, cmap):

    import numpy as np
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy

    dims = {}
    for dim in coords.dims:
        for key in ("time", "lon", "lat"):
            if key in dim:
                dims[key] = coords[dim].values

    ndata = len(data)

    if cmap is None:
        cmap = "RdBu_r"

    if vlim is None:
        vmin = min([np.nanmin(data[i]) for i in range(ndata)])
        vmax = max([np.nanmax(data[i]) for i in range(ndata)])
    else:
        vmin, vmax = vlim

    fig = plt.figure(figsize=(9 * ndata, 6))
    axes = []
    for i in range(ndata):
        axes.append(fig.add_subplot(1, ndata, i + 1, projection=ccrs.Robinson()))
        # axes[i].axis('off')
        gl = axes[i].gridlines(crs=ccrs.Robinson(), draw_labels=False, linewidth=0.1)
        gl.top_labels = False
        gl.right_labels = False
        axes[i].add_feature(cartopy.feature.COASTLINE, edgecolor="black")
        axes[i].add_feature(cartopy.feature.BORDERS, edgecolor="black", linestyle="--")

        image = axes[i].pcolormesh(dims["lon"], dims["lat"], data[i], vmin=vmin, vmax=vmax,
                                   cmap=cmap, transform=ccrs.PlateCarree(), shading='auto')
        axes[i].set_facecolor('grey')
        axes[i].yaxis.set_ticks_position("left")
        axes[i].set_title(titles[i], size=18)

    cb = plt.colorbar(image, location="bottom", ax=axes, fraction=0.1, pad=0.1)
    cb.set_label(get_longname(data_type), size=14)

    fig.suptitle(time_label, size=20)

    if ndata == 2:
        bbox_props = dict(boxstyle="rarrow,pad=0.3", fc="black", lw=2)
        plt.text(0.51, 0.53, "CRAI", ha="center", va="center", color="white", size=18, bbox=bbox_props,
                 transform=plt.gcf().transFigure)

    plt.savefig(output_name, dpi=150, bbox_inches='tight')
