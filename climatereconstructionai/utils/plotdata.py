
def get_longname(var):

    if var in ("tas", "temperature_anomaly", "tas_mean", "temperature_anomaly_mean"):
        lname = "Near surface temperature anomaly (°C)"
    else:
        lname = ""

    return lname

def plot_data(lon,lat,data,output_names,data_type,time_indices,scale_type,cmap):

    if not time_indices is None:

        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib import colors
        import cartopy.crs as ccrs
        import cartopy
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        ndata = len(data)

        if cmap is None:
            cmap = "RdBu_r"

        vmin = min([np.nanmin(data[i][time_indices]) for i in range(ndata)])
        vmax = max([np.nanmax(data[i][time_indices]) for i in range(ndata)])
        if scale_type == "symmetric":
            vlim = max([abs(vmin),abs(vmax)])
            vmin = -vlim
            vmax = vlim

        for i in range(ndata):
            for j in time_indices:

                fig, ax = plt.subplots()
                ax.axis('off')

                ax = plt.axes(projection=ccrs.Robinson(central_longitude=180))
                gl = ax.gridlines(crs=ccrs.Robinson(), draw_labels=False, linewidth=0.1)
                gl.top_labels = False
                gl.right_labels = False
                ax.add_feature(cartopy.feature.COASTLINE, edgecolor="black")
                ax.add_feature(cartopy.feature.BORDERS, edgecolor="black", linestyle="--")

                image = ax.pcolormesh(lon,lat,data[i][j].squeeze(),vmin=-vlim,vmax=vlim,cmap=cmap,transform=ccrs.PlateCarree(),shading='auto')
                ax.set_facecolor('grey')
                ax.yaxis.set_ticks_position("left")
                cb = plt.colorbar(image,location="bottom")
                cb.set_label(get_longname(data_type),size=14)
                plt.savefig(output_names[i]+"_"+str(j)+".png", dpi=150, bbox_inches='tight')
