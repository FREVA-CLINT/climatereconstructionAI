
def get_longname(var):

    if var in ("tas", "temperature_anomaly", "tas_mean", "temperature_anomaly_mean"):
        lname = "Near surface temperature anomaly (°C)"
    else:
        lname = ""

    return lname

def plot_data(lon,lat,data,data_type,time_indices,output_name,cmap):

    if not time_indices is None:

        import matplotlib.pyplot as plt
        from matplotlib import colors
        import cartopy.crs as ccrs
        import cartopy
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        if cmap is None:
            cmap = "RdBu_r"

        for i in time_indices:

            fig, ax = plt.subplots()
            ax.axis('off')

            ax = plt.axes(projection=ccrs.Robinson())
            gl = ax.gridlines(crs=ccrs.Robinson(), draw_labels=False, linewidth=0.1)
            gl.top_labels = False
            gl.right_labels = False
            ax.add_feature(cartopy.feature.COASTLINE, edgecolor="black")
            ax.add_feature(cartopy.feature.BORDERS, edgecolor="black", linestyle="--")

            image = ax.pcolormesh(lon,lat,data[i].squeeze(),cmap=cmap,transform=ccrs.PlateCarree(),shading='auto')
            ax.set_facecolor('grey')
            ax.yaxis.set_ticks_position("left")
            cb = plt.colorbar(image,location="bottom")
            cb.set_label(get_longname(data_type),size=14)
            plt.savefig(output_name+"_"+str(i)+".png", dpi=150, bbox_inches='tight')
