############################################################
# Juilia Miniweather plotting script
############################################################

### SECTION : USER DEFINITION ##############

file_name = "./output.nc"   # Miniweather output file name
fig_dir_name = "./figures"  # Figure directory
make_ani = true             # Make animation

############################################

using NCDatasets
using PyPlot
using Printf

XLEN = Float64(2.E4) # Miniweather X-dir length
ZLEN = Float64(1.E4) # Miniweather Z-dir height

extent=[0,XLEN/1000.0,0,ZLEN/1000.0]

# Open NetCDF output file with a create mode
ds = Dataset(file_name,"r")

ds_t = ds["t"]
ds_dens = ds["dens"]
ds_uwnd = ds["uwnd"]
ds_wwnd = ds["wwnd"]
ds_theta = ds["theta"]

nt = ds.dim["t"]
nx = ds.dim["x"]
nz = ds.dim["z"]

# Make a figure directory (Remove & Make if exist)
if isdir(fig_dir_name)
   run(Cmd(`rm -rf $fig_dir_name`))
   run(Cmd(`mkdir -p $fig_dir_name`))
else
   run(Cmd(`mkdir -p $fig_dir_name`))
end

# Time loop
for n in 1:nt
    println("Processing time: ",n," / ",nt)
 
    time = ds_t[n]
    dens = ds_dens[:,:,n]
    uwnd = ds_uwnd[:,:,n]
    wwnd = ds_wwnd[:,:,n]
    rhot = ds_theta[:,:,n]

    it = @sprintf("%.4i",n)
    ctime = @sprintf("%6.1f",time)

    # Display
    fig = figure("out",figsize=(8,5.5))
    subplots_adjust(hspace=0.1)
    suptitle(string("Julia Miniweather
            \nnx=",nx,"; nz=",nz,"; Time=",ctime," s"))
    subplot(221)
       xlabel("X (km)")
       ylabel("Z (km)")
       title("Density")
       imshow(transpose(reverse(dens,dims=2)),extent=extent)
    subplot(222)
       xlabel("X (km)")
       ylabel("Z (km)")
       title("Potential temperature")
       imshow(transpose(reverse(rhot,dims=2)),extent=extent)
    subplot(223)
       xlabel("X (km)")
       ylabel("Z (km)")
       title("U-wind component")
       imshow(transpose(reverse(uwnd,dims=2)),extent=extent,cmap="bwr")
    subplot(224)
       xlabel("X (km)")
       ylabel("Z (km)")
       title("W-wind component")
       imshow(transpose(reverse(wwnd,dims=2)),extent=extent,cmap="bwr")

    fig_name = string(fig_dir_name,"/fig_",it,".png")
    savefig(fig_name,bbox_inches="tight",dpi=120)
    close(fig)
end

# Make figures as animation
if make_ani
   run(Cmd(`convert -delay 0 -loop 0 $fig_dir_name/\*.png anim.gif`))
end
