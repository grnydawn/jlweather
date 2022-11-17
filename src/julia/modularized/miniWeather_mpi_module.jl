import TimerOutputs.@timeit,
       TimerOutputs.show

import OffsetArrays.OffsetArray,
       OffsetArrays.OffsetVector

import NCDatasets.Dataset,
       NCDatasets.defDim,
       NCDatasets.defVar

import MPI.Init,
       MPI.Allreduce!,
       MPI.Barrier,
       MPI.Recv!,
       MPI.Send

import Printf.@printf

# MPI INIT
Init()

include("constants.jl")

using .Constants: COMM, NRANKS, MYRANK, SIM_TIME, NX_GLOB, NZ_GLOB, OUT_FREQ, LOG_FREQ, DATA_SPEC
using .Constants: OUTFILE, WORKDIR, DEBUGDIR, NPER, I_BEG, I_END, NX, NZ, LEFT_RANK, RIGHT_RANK
using .Constants: K_BEG, MASTERRANK, MASTERPROC, HS, STEN_SIZE, NUM_VARS, XLEN, ZLEN, HV_BETA
using .Constants: CFL, MAX_SPEED, DX, DZ, DT, NQPOINTS, PI, GRAV, CP, CV, RD, P0, C0, GAMMA
using .Constants: ID_DENS, ID_UMOM, ID_WMOM, ID_RHOT, DIR_X, DIR_Z, DATA_SPEC_COLLISION
using .Constants: DATA_SPEC_THERMAL, DATA_SPEC_GRAVITY_WAVES, DATA_SPEC_DENSITY_CURRENT
using .Constants: FLOAT, INTEGER, DATA_SPEC_INJECTION, qpoints, qweights

include("variables.jl")

using .Variables: state, statetmp, flux, tend, hy_dens_cell, hy_dens_theta_cell
using .Variables: hy_dens_int, hy_dens_theta_int, hy_pressure_int
using .Variables: sendbuf_l, sendbuf_r, recvbuf_l, recvbuf_r, etime

include("timestep.jl")
using .Timestep: perform_timestep!


##############
# functions
##############

"""
    main()

simulate weather-like flows.

# Examples
```julia-repl
julia> main()
```
"""
function main()

    local stime = FLOAT(0.0)
    local output_counter = FLOAT(0.0)
    local log_counter = FLOAT(0.0)
    local dt = DT
    local nt = INTEGER(1)
    
    if MASTERPROC
        println("nx_glob, nz_glob: $NX_GLOB $NZ_GLOB")
        println("dx, dz: $DX $DZ")
        println("dt: $DT")
    end
        
    #println("nx, nz at $MYRANK: $NX($I_BEG:$I_END) $NZ($K_BEG:$NZ)")
    
    Barrier(COMM)

    #Initial reductions for mass, kinetic energy, and total energy
    local mass0, te0 = reductions(state, hy_dens_cell, hy_dens_theta_cell)

    #Output the initial state
    output(state,stime,nt,hy_dens_cell,hy_dens_theta_cell)
    
    # main loop
    cputime = @elapsed while stime < SIM_TIME

        #If the time step leads to exceeding the simulation time, shorten it for the last step
        if stime + dt > SIM_TIME
            dt = SIM_TIME - stime
        end

        #Perform a single time step
        @timeit etime "timestep" perform_timestep!(state, statetmp, flux, tend, dt, recvbuf_l, recvbuf_r,
                  sendbuf_l, sendbuf_r, hy_dens_cell, hy_dens_theta_cell,
                  hy_dens_int, hy_dens_theta_int, hy_pressure_int)

        #Update the elapsed time and output counter
        stime = stime + dt
        output_counter = output_counter + dt
        log_counter = log_counter + dt

        #If it's time for output, reset the counter, and do output
        if (output_counter >= OUT_FREQ)
          #Increment the number of outputs
          nt = nt + 1
          output(state,stime,nt,hy_dens_cell,hy_dens_theta_cell)
          output_counter = output_counter - OUT_FREQ
        end

        if MASTERPROC && (log_counter >= LOG_FREQ)
          @printf("[%3.1f%% of %2.1f]\n", stime/SIM_TIME*100, SIM_TIME)
          log_counter = log_counter - LOG_FREQ
        end

    end
 
    local mass, te = reductions(state, hy_dens_cell, hy_dens_theta_cell)

    if MASTERPROC
        println( "CPU Time: $cputime")
        @printf("d_mass: %.15e\n", (mass - mass0)/mass0)
        @printf("d_te  : %.15e\n", (te - te0)/te0)
        show(etime); println("")
    end
    finalize!(state)

end

function reductions(state::OffsetArray{FLOAT, 3, Array{FLOAT, 3}},
                    hy_dens_cell::OffsetVector{FLOAT, Vector{FLOAT}},
                    hy_dens_theta_cell::OffsetVector{FLOAT, Vector{FLOAT}})
    
    local mass, te, r, u, w, th, p, t, ke, le = [zero(FLOAT) for _ in 1:10] 
    glob = Array{FLOAT}(undef, 2)
    
    for k in 1:NZ
        for i in 1:NX
            r  =   state[i,k,ID_DENS] + hy_dens_cell[k]             # Density
            u  =   state[i,k,ID_UMOM] / r                           # U-wind
            w  =   state[i,k,ID_WMOM] / r                           # W-wind
            th = ( state[i,k,ID_RHOT] + hy_dens_theta_cell[k] ) / r # Potential Temperature (theta)
            p  = C0*(r*th)^GAMMA      # Pressure
            t  = th / (P0/p)^(RD/CP)  # Temperature
            ke = r*(u*u+w*w)          # Kinetic Energy
            ie = r*CV*t               # Internal Energy
            mass = mass + r            *DX*DZ # Accumulate domain mass
            te   = te   + (ke + r*CV*t)*DX*DZ # Accumulate domain total energy
        end
    end
    
    Allreduce!(Array{FLOAT}([mass,te]), glob, +, COMM)
    
    return glob
end

#Output the fluid state (state) to a NetCDF file at a given elapsed model time (etime)
function output(state::OffsetArray{FLOAT, 3, Array{FLOAT, 3}},
                stime::FLOAT,
                nt::INTEGER,
                hy_dens_cell::OffsetVector{FLOAT, Vector{FLOAT}},
                hy_dens_theta_cell::OffsetVector{FLOAT, Vector{FLOAT}})

    var_local  = zeros(FLOAT, NX, NZ, NUM_VARS)

    if MASTERPROC
       var_global  = zeros(FLOAT, NX_GLOB, NZ_GLOB, NUM_VARS)
    end

    #Store perturbed values in the temp arrays for output
    for k in 1:NZ
        for i in 1:NX
            var_local[i,k,ID_DENS]  = state[i,k,ID_DENS]
            var_local[i,k,ID_UMOM]  = state[i,k,ID_UMOM] / ( hy_dens_cell[k] + state[i,k,ID_DENS] )
            var_local[i,k,ID_WMOM]  = state[i,k,ID_WMOM] / ( hy_dens_cell[k] + state[i,k,ID_DENS] )
            var_local[i,k,ID_RHOT] = ( state[i,k,ID_RHOT] + hy_dens_theta_cell[k] ) / ( hy_dens_cell[k] + state[i,k,ID_DENS] ) - hy_dens_theta_cell[k] / hy_dens_cell[k]
        end
    end

    #Gather data from multiple CPUs
    #  - Implemented in an inefficient way for the purpose of tests
    #  - Will be improved in next version.
    if MASTERPROC
       ibeg_chunk = zeros(INTEGER,NRANKS)
       iend_chunk = zeros(INTEGER,NRANKS)
       nchunk     = zeros(INTEGER,NRANKS)
       for n in 1:NRANKS
          ibeg_chunk[n] = trunc(INTEGER, round(NPER* (n-1))+1)
          iend_chunk[n] = trunc(INTEGER, round(NPER*((n-1)+1)))
          nchunk[n]     = iend_chunk[n] - ibeg_chunk[n] + 1
       end
    end

    if MASTERPROC
       var_global[I_BEG:I_END,:,:] = var_local[:,:,:]
       if NRANKS > 1
          for i in 2:NRANKS
              var_local = Array{FLOAT}(undef, nchunk[i],NZ,NUM_VARS)
              status = Recv!(var_local,i-1,0,COMM)
              var_global[ibeg_chunk[i]:iend_chunk[i],:,:] = var_local[:,:,:]
          end
       end
    else
       Send(var_local,MASTERRANK,0,COMM)
    end

    # Write output only in MASTER
    if MASTERPROC

       #If the elapsed time is zero, create the file. Otherwise, open the file
       if ( stime == 0.0 )

          # Open NetCDF output file with a create mode
          ds = Dataset(OUTFILE,"c")

          defDim(ds,"t",Inf)
          defDim(ds,"x",NX_GLOB)
          defDim(ds,"z",NZ_GLOB)

          nc_var = defVar(ds,"t",FLOAT,("t",))
          nc_var[nt] = stime
          nc_var = defVar(ds,"dens",FLOAT,("x","z","t"))
          nc_var[:,:,nt] = var_global[:,:,ID_DENS]
          nc_var = defVar(ds,"uwnd",FLOAT,("x","z","t"))
          nc_var[:,:,nt] = var_global[:,:,ID_UMOM]
          nc_var = defVar(ds,"wwnd",FLOAT,("x","z","t"))
          nc_var[:,:,nt] = var_global[:,:,ID_WMOM]
          nc_var = defVar(ds,"theta",FLOAT,("x","z","t"))
          nc_var[:,:,nt] = var_global[:,:,ID_RHOT]

          # Close NetCDF file
          close(ds)

       else

          # Open NetCDF output file with an append mode
          ds = Dataset(OUTFILE,"a")

          nc_var = ds["t"]
          nc_var[nt] = stime
          nc_var = ds["dens"]
          nc_var[:,:,nt] = var_global[:,:,ID_DENS]
          nc_var = ds["uwnd"]
          nc_var[:,:,nt] = var_global[:,:,ID_UMOM]
          nc_var = ds["wwnd"]
          nc_var[:,:,nt] = var_global[:,:,ID_WMOM]
          nc_var = ds["theta"]
          nc_var[:,:,nt] = var_global[:,:,ID_RHOT]

          # Close NetCDF file
          close(ds)

       end # etime
    end # MASTER
end

function finalize!(state::OffsetArray{FLOAT, 3, Array{FLOAT, 3}})

    #println(axes(state))
    
end

# invoke main function
main()
