using AccelInterfaces

import Profile

import OffsetArrays.OffsetArray,
       OffsetArrays.OffsetVector

import ArgParse.ArgParseSettings,
       ArgParse.parse_args,
       ArgParse.@add_arg_table!

import NCDatasets.Dataset,
       NCDatasets.defDim,
       NCDatasets.defVar


import Match.@match

import MPI.Init,
       MPI.COMM_WORLD,
       MPI.Comm_rank,
       MPI.Comm_size,
       MPI.Allreduce!,
       MPI.Barrier,
       MPI.Waitall!,
       MPI.Request,
       MPI.Irecv!,
       MPI.Isend

import Debugger

import Printf.@printf

import Libdl

##############
# Accelerators
##############

const COMPILE_FOPENACC_CRAY = "ftn -shared -fPIC -h acc,noomp"
#const COMPILE_FORTRAN = "ftn -fPIC -shared -h noacc,noomp"
const COMPILE = COMPILE_FOPENACC_CRAY
#const COMPILE = COMPILE_FORTRAN

const PATH_REDUCTION_KERNEL = joinpath(@__DIR__, "reduction.knl") 
const PATH_APPLY_SAME_KERNEL = joinpath(@__DIR__, "apply_same.knl") 
const PATH_APPLY_DIFF_KERNEL = joinpath(@__DIR__, "apply_diff.knl") 
const PATH_TEND_X_KERNEL = joinpath(@__DIR__, "tend_x.knl") 
const PATH_TEND_Z_KERNEL = joinpath(@__DIR__, "tend_z.knl") 

##############
# constants
##############
    
# julia command to link MPI.jl to system MPI installation
# julia --project=. -e 'ENV["JULIA_MPI_BINARY"]="system"; 
# ENV["JULIA_MPI_PATH"]="/opt/cray/pe/mpich/8.1.16/ofi/crayclang/10.0";
# using Pkg; Pkg.build("MPI"; verbose=true)'
# MPI.install_mpiexecjl()

Init()
const COMM   = COMM_WORLD
const NRANKS = Comm_size(COMM)
const MYRANK = Comm_rank(COMM)

s = ArgParseSettings()
@add_arg_table! s begin
    "--simtime", "-s"
        help = "simulation time"
        arg_type = Float64
        default = 400.0
    "--nx", "-x"
        help = "x-dimension"
        arg_type = Int64
        default = 100
    "--nz", "-z"
        help = "z-dimension"
        arg_type = Int64
        default = 50
    "--outfreq", "-f"
        help = "output frequency in time"
        arg_type = Float64
        default = 400.0
    "--dataspec", "-d"
        help = "data spec"
        default = 2
    "--outfile", "-o"
        help = "output file path"
        default = "output.nc"
end

parsed_args = parse_args(ARGS, s)

const SIM_TIME    = parsed_args["simtime"]
const NX_GLOB     = parsed_args["nx"]
const NZ_GLOB     = parsed_args["nz"]
const OUT_FREQ    = parsed_args["outfreq"]
const DATA_SPEC   = parsed_args["dataspec"]
const OUTFILE     = parsed_args["outfile"]

const NPER  = Float64(NX_GLOB)/NRANKS
const I_BEG = trunc(Int, round(NPER* MYRANK)+1)
const I_END = trunc(Int, round(NPER*(MYRANK+1)))
const NX    = I_END - I_BEG + 1
const NZ    = NZ_GLOB

const LEFT_RANK = MYRANK-1 == -1 ? NRANKS - 1 : MYRANK - 1 
const RIGHT_RANK = MYRANK+1 == NRANKS ? 0 : MYRANK + 1 

#Vertical direction isn't MPI-ized, so the rank's local values = the global values
const K_BEG       = 1
const MASTERPROC  = (MYRANK == 0)

const HS          = 2
const STEN_SIZE   = 4 #Size of the stencil used for interpolation
const NUM_VARS    = 4
const XLEN        = Float64(2.E4) # Length of the domain in the x-direction (meters)
const ZLEN        = Float64(1.E4) # Length of the domain in the z-direction (meters)
const HV_BETA     = Float64(0.05) # How strong to diffuse the solution: hv_beta \in [0:1]
const CFL         = Float64(1.5)  # "Courant, Friedrichs, Lewy" number (for numerical stability)
const MAX_SPEED   = Float64(450.0)# Assumed maximum wave speed during the simulation (speed of sound + speed of wind) (meter / sec)
const DX          = XLEN / NX_GLOB
const DZ          = ZLEN / NZ_GLOB
const DT          = min(DX,DZ) / MAX_SPEED * CFL
const NQPOINTS    = 3
const GRAV        = Float64(9.8)
const CP          = Float64(1004.0) # Specific heat of dry air at constant pressure
const CV          = Float64(717.0)  # Specific heat of dry air at constant volume
const RD          = Float64(287.0)  # Dry air constant for equation of state (P=rho*rd*T)
const P0          = Float64(1.0E5)  # Standard pressure at the surface in Pascals
const C0          = Float64(27.5629410929725921310572974482)
const GAMMA       = Float64(1.40027894002789400278940027894)

const ID_DENS     = 1
const ID_UMOM     = 2
const ID_WMOM     = 3
const ID_RHOT     = 4
                    
const DIR_X       = 1 #Integer constant to express that this operation is in the x-direction
const DIR_Z       = 2 #Integer constant to express that this operation is in the z-direction

const DATA_SPEC_COLLISION       = 1
const DATA_SPEC_THERMAL         = 2
const DATA_SPEC_MOUNTAIN        = 3
const DATA_SPEC_TURBULENCE      = 4
const DATA_SPEC_DENSITY_CURRENT = 5
const DATA_SPEC_INJECTION       = 6

const qpoints     = Array{Float64}([0.112701665379258311482073460022E0 , 0.500000000000000000000000000000E0 , 0.887298334620741688517926539980E0])
const qweights    = Array{Float64}([0.277777777777777777777777777779E0 , 0.444444444444444444444444444444E0 , 0.277777777777777777777777777779E0])

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
function main(args::Vector{String})

    ######################
    # top-level variables
    ######################

    local etime = Float64(0.0)
    local output_counter = Float64(0.0)
    local dt = DT
    local nt = Int(1)


    @jaccel myaccel framework(fortran_openacc) constant(NX, NZ, DX, DZ, HS, NUM_VARS, C0, GAMMA, P0, HV_BETA, GRAV,
                RD, CP, CV, ID_DENS, ID_UMOM, ID_WMOM, ID_RHOT, STEN_SIZE) compile(COMPILE) set(master=MASTERPROC,
                debugdir=".jaitmp")

    @jkernel reduce_kernel myaccel PATH_REDUCTION_KERNEL
    @jkernel tend_x_kernel myaccel PATH_TEND_X_KERNEL
    @jkernel tend_z_kernel myaccel PATH_TEND_Z_KERNEL

    #Initialize the grid and the data  
    (state, statetmp, flux, tend, hy_dens_cell, hy_dens_theta_cell,
            hy_dens_int, hy_dens_theta_int, hy_pressure_int, sendbuf_l,
            sendbuf_r, recvbuf_l, recvbuf_r) = init!()

	@jenterdata myaccel allocate(state, statetmp, flux, tend, hy_dens_cell, hy_dens_theta_cell,
			hy_dens_int, hy_dens_theta_int, hy_pressure_int)

	# NOTE: add filename and line # to generate hash of jai functions

	@jenterdata myaccel update(hy_dens_cell, hy_dens_theta_cell,
            hy_dens_int, hy_dens_theta_int, hy_pressure_int)


    #Initial reductions for mass, kinetic energy, and total energy

    local mass0, te0 = reductions_accel(state, hy_dens_cell, hy_dens_theta_cell)

    #Output the initial state
    output(state,etime,nt,hy_dens_cell,hy_dens_theta_cell)
    
    # main loop
    elapsedtime = @elapsed while etime < SIM_TIME

        #If the time step leads to exceeding the simulation time, shorten it for the last step
        if etime + dt > SIM_TIME
            dt = SIM_TIME - etime
        end

        #Perform a single time step
        if MASTERPROC
            #Profile.@profile timestep!(state, statetmp, flux, tend, dt, recvbuf_l, recvbuf_r,
            timestep!(state, statetmp, flux, tend, dt, recvbuf_l, recvbuf_r,
                      sendbuf_l, sendbuf_r, hy_dens_cell, hy_dens_theta_cell,
                      hy_dens_int, hy_dens_theta_int, hy_pressure_int)
        else
            timestep!(state, statetmp, flux, tend, dt, recvbuf_l, recvbuf_r,
                      sendbuf_l, sendbuf_r, hy_dens_cell, hy_dens_theta_cell,
                      hy_dens_int, hy_dens_theta_int, hy_pressure_int)
        end

        #Update the elapsed time and output counter
        etime = etime + dt
        output_counter = output_counter + dt

        #If it's time for output, reset the counter, and do output
        if (output_counter >= OUT_FREQ)
          #Increment the number of outputs
          nt = nt + 1
          output(state,etime,nt,hy_dens_cell,hy_dens_theta_cell)
          output_counter = output_counter - OUT_FREQ
        end

    end

    if MASTERPROC
	    #Profile.print()
    end

    local mass, te = reductions_accel(state, hy_dens_cell, hy_dens_theta_cell)
 
 	@jexitdata myaccel deallocate(state, statetmp, flux, tend, hy_dens_cell, hy_dens_theta_cell,
			hy_dens_int, hy_dens_theta_int, hy_pressure_int)
  
    if MASTERPROC
        println( "CPU Time: $elapsedtime")
        @printf("d_mass: %f\n", (mass - mass0)/mass0)
        @printf("d_te:   %f\n", (te - te0)/te0)
    end

    @jdecel myaccel

    finalize!(state)

end

function init!()
    
    if MASTERPROC
        println("nx_glob, nz_glob: $NX_GLOB $NZ_GLOB")
        println("dx, dz: $DX $DZ")
        println("dt: $DT")
    end
        
    #println("nx, nz at $MYRANK: $NX($I_BEG:$I_END) $NZ($K_BEG:$NZ)")
    
    Barrier(COMM)
    
    _state      = zeros(Float64, NX+2*HS, NZ+2*HS, NUM_VARS) 
    state       = OffsetArray(_state, 1-HS:NX+HS, 1-HS:NZ+HS, 1:NUM_VARS)
    _statetmp   = Array{Float64}(undef, NX+2*HS, NZ+2*HS, NUM_VARS) 
    statetmp    = OffsetArray(_statetmp, 1-HS:NX+HS, 1-HS:NZ+HS, 1:NUM_VARS)
    
    flux        = zeros(Float64, NX+1, NZ+1, NUM_VARS) 
    tend        = zeros(Float64, NX, NZ, NUM_VARS) 
 
    _hy_dens_cell       = zeros(Float64, NZ+2*HS) 
    hy_dens_cell        = OffsetArray(_hy_dens_cell, 1-HS:NZ+HS)
    _hy_dens_theta_cell = zeros(Float64, NZ+2*HS) 
    hy_dens_theta_cell  = OffsetArray(_hy_dens_theta_cell, 1-HS:NZ+HS)   
    
    hy_dens_int         = Array{Float64}(undef, NZ+1)
    hy_dens_theta_int   = Array{Float64}(undef, NZ+1)
    hy_pressure_int     = Array{Float64}(undef, NZ+1)   
    
    sendbuf_l = Array{Float64}(undef, HS, NZ, NUM_VARS)
    sendbuf_r = Array{Float64}(undef, HS, NZ, NUM_VARS)
    recvbuf_l = Array{Float64}(undef, HS, NZ, NUM_VARS)
    recvbuf_r = Array{Float64}(undef, HS, NZ, NUM_VARS)   
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #! Initialize the cell-averaged fluid state via Gauss-Legendre quadrature
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    for k in 1-HS:NZ+HS
      for i in 1-HS:NX+HS
        #Initialize the state to zero
        #Use Gauss-Legendre quadrature to initialize a hydrostatic balance + temperature perturbation
        for kk in 1:NQPOINTS
          for ii in 1:NQPOINTS
            #Compute the x,z location within the global domain based on cell and quadrature index
            x = (I_BEG-1 + i-0.5) * DX + (qpoints[ii]-0.5)*DX
            z = (K_BEG-1 + k-0.5) * DZ + (qpoints[kk]-0.5)*DZ

            #Set the fluid state based on the user's specification
            r, u, w, t, hr, ht = @match DATA_SPEC begin
                DATA_SPEC_COLLISION       => collision!(x,z)
                DATA_SPEC_THERMAL         => thermal!(x,z)
                DATA_SPEC_MOUNTAIN        => mountain_waves!(x,z)
                DATA_SPEC_TURBULENCE      => turbulence!(x,z)
                DATA_SPEC_DENSITY_CURRENT => density_current!(x,z)
                DATA_SPEC_INJECTION       => injection!(x,z)
            end

            #Store into the fluid state array
            state[i,k,ID_DENS] = state[i,k,ID_DENS] + r                         * qweights[ii]*qweights[kk]
            state[i,k,ID_UMOM] = state[i,k,ID_UMOM] + (r+hr)*u                  * qweights[ii]*qweights[kk]
            state[i,k,ID_WMOM] = state[i,k,ID_WMOM] + (r+hr)*w                  * qweights[ii]*qweights[kk]
            state[i,k,ID_RHOT] = state[i,k,ID_RHOT] + ( (r+hr)*(t+ht) - hr*ht ) * qweights[ii]*qweights[kk]
          end
        end
        for ll in 1:NUM_VARS
          statetmp[i,k,ll] = state[i,k,ll]
        end
      end
    end

    for k in 1-HS:NZ+HS
        for kk in 1:NQPOINTS
            z = (K_BEG-1 + k-0.5) * DZ + (qpoints[kk]-0.5)*DZ
            
            #Set the fluid state based on the user's specification
            r, u, w, t, hr, ht = @match DATA_SPEC begin
                DATA_SPEC_COLLISION       => collision!(0.0,z)
                DATA_SPEC_THERMAL         => thermal!(0.0,z)
                DATA_SPEC_MOUNTAIN        => mountain_waves!(0.0,z)
                DATA_SPEC_TURBULENCE      => turbulence!(0.0,z)
                DATA_SPEC_DENSITY_CURRENT => density_current!(0.0,z)
                DATA_SPEC_INJECTION       => injection!(0.0,z)
            end           

            hy_dens_cell[k]       = hy_dens_cell[k]       + hr    * qweights[kk]
            hy_dens_theta_cell[k] = hy_dens_theta_cell[k] + hr*ht * qweights[kk]
        end
    end
    
    #Compute the hydrostatic background state at vertical cell interfaces
    for k in 1:NZ+1
        z = (K_BEG-1 + k-1) * DZ
        #Set the fluid state based on the user's specification
        r, u, w, t, hr, ht = @match DATA_SPEC begin
            DATA_SPEC_COLLISION       => collision!(0.0,z)
            DATA_SPEC_THERMAL         => thermal!(0.0,z)
            DATA_SPEC_MOUNTAIN        => mountain_waves!(0.0,z)
            DATA_SPEC_TURBULENCE      => turbulence!(0.0,z)
            DATA_SPEC_DENSITY_CURRENT => density_current!(0.0,z)
            DATA_SPEC_INJECTION       => injection!(0.0,z)
        end                  

      hy_dens_int[k] = hr
      hy_dens_theta_int[k] = hr*ht
      hy_pressure_int[k] = C0*(hr*ht)^GAMMA
    end
    
    return (state, statetmp, flux, tend, hy_dens_cell, hy_dens_theta_cell,
            hy_dens_int, hy_dens_theta_int, hy_pressure_int, sendbuf_l,
            sendbuf_r, recvbuf_l, recvbuf_r)
end

function injection!(x::Float64, z::Float64)

    #Hydrostatic density and potential temperature
    hr,ht = hydro_const_theta!(z)

    r  = Float64(0.0) # Density
    t  = Float64(0.0) # Potential temperature
    u  = Float64(0.0) # Uwind
    w  = Float64(0.0) # Wwind
    
    return r, u, w, t, hr, ht
end

function density_current!(x::Float64, z::Float64)

    #Hydrostatic density and potential temperature
    hr,ht = hydro_const_theta!(z)

    r  = Float64(0.0) # Density
    t  = Float64(0.0) # Potential temperature
    u  = Float64(0.0) # Uwind
    w  = Float64(0.0) # Wwind
    
    t = t + sample_ellipse_cosine!(x,z,-20.0,XLEN/2,5000.0,4000.0,2000.0)

    return r, u, w, t, hr, ht
end

function turbulence!(x::Float64, z::Float64)

    #Hydrostatic density and potential temperature
    hr,ht = hydro_const_theta!(z)

    r  = Float64(0.0) # Density
    t  = Float64(0.0) # Potential temperature
    u  = Float64(0.0) # Uwind
    w  = Float64(0.0) # Wwind
    
    return r, u, w, t, hr, ht
end

function mountain_waves!(x::Float64, z::Float64)

    #Hydrostatic density and potential temperature
    hr,ht = hydro_const_theta!(z)

    r  = Float64(0.0) # Density
    t  = Float64(0.0) # Potential temperature
    u  = Float64(0.0) # Uwind
    w  = Float64(0.0) # Wwind
    
    return r, u, w, t, hr, ht
end

#Rising thermal
function thermal!(x::Float64, z::Float64)

    #Hydrostatic density and potential temperature
    hr,ht = hydro_const_theta!(z)

    r  = Float64(0.0) # Density
    t  = Float64(0.0) # Potential temperature
    u  = Float64(0.0) # Uwind
    w  = Float64(0.0) # Wwind
    
    t = t + sample_ellipse_cosine!(x,z,3.0,XLEN/2,2000.0,2000.0,2000.0) 

    return r, u, w, t, hr, ht
end

#Colliding thermals
function collision!(x::Float64, z::Float64)
    
    #Hydrostatic density and potential temperature
    hr,ht = hydro_const_theta!(z)

    r  = Float64(0.0) # Density
    t  = Float64(0.0) # Potential temperature
    u  = Float64(0.0) # Uwind
    w  = Float64(0.0) # Wwind

    t = t + sample_ellipse_cosine!(x,z, 20.0,XLEN/2,2000.0,2000.0,2000.0)
    t = t + sample_ellipse_cosine!(x,z,-20.0,XLEN/2,8000.0,2000.0,2000.0)

    return r, u, w, t, hr, ht
end

#Establish hydrstatic balance using constant potential temperature (thermally neutral atmosphere)
function hydro_const_theta!(z::Float64)

    r      = Float64(0.0) # Density
    t      = Float64(0.0) # Potential temperature

    theta0 = Float64(300.0) # Background potential temperature
    exner0 = Float64(1.0)   # Surface-level Exner pressure

    t      = theta0                            # Potential temperature at z
    exner  = exner0 - GRAV * z / (CP * theta0) # Exner pressure at z
    p      = P0 * exner^(CP/RD)                # Pressure at z
    rt     = (p / C0)^(Float64(1.0)/GAMMA)     # rho*theta at z
    r      = rt / t                            # Density at z

    return r, t
end

#Sample from an ellipse of a specified center, radius, and amplitude at a specified location
function sample_ellipse_cosine!(   x::Float64,    z::Float64, amp::Float64, 
                                  x0::Float64,   z0::Float64, 
                                xrad::Float64, zrad::Float64 )

    #Compute distance from bubble center
    local dist = sqrt( ((x-x0)/xrad)^2 + ((z-z0)/zrad)^2 ) * pi / Float64(2.0)
 
    #If the distance from bubble center is less than the radius, create a cos**2 profile
    if (dist <= pi / Float64(2.0) ) 
      val = amp * cos(dist)^2
    else
      val = Float64(0.0)
    end
    
    return val
end

#Performs a single dimensionally split time step using a simple low-storate three-stage Runge-Kutta time integrator
#The dimensional splitting is a second-order-accurate alternating Strang splitting in which the
#order of directions is alternated each time step.
#The Runge-Kutta method used here is defined as follows:
# q*     = q[n] + dt/3 * rhs(q[n])
# q**    = q[n] + dt/2 * rhs(q*  )
# q[n+1] = q[n] + dt/1 * rhs(q** )
function timestep!(state::OffsetArray{Float64, 3, Array{Float64, 3}},
                   statetmp::OffsetArray{Float64, 3, Array{Float64, 3}},
                   flux::Array{Float64, 3},
                   tend::Array{Float64, 3},
                   dt::Float64,
                   recvbuf_l::Array{Float64, 3},
                   recvbuf_r::Array{Float64, 3},
                   sendbuf_l::Array{Float64, 3},
                   sendbuf_r::Array{Float64, 3},
                   hy_dens_cell::OffsetVector{Float64, Vector{Float64}},
                   hy_dens_theta_cell::OffsetVector{Float64, Vector{Float64}},
                   hy_dens_int::Vector{Float64},
                   hy_dens_theta_int::Vector{Float64},
                   hy_pressure_int::Vector{Float64})
    
    local direction_switch = true
    
    if direction_switch
        
        #x-direction first
        semi_discrete_step!(state , state    , statetmp , dt / 3 , DIR_X , flux , tend,
            recvbuf_l, recvbuf_r, sendbuf_l, sendbuf_r, hy_dens_cell, hy_dens_theta_cell,
            hy_dens_int, hy_dens_theta_int, hy_pressure_int)
        
        semi_discrete_step!(state , statetmp , statetmp , dt / 2 , DIR_X , flux , tend,
            recvbuf_l, recvbuf_r, sendbuf_l, sendbuf_r, hy_dens_cell, hy_dens_theta_cell,
            hy_dens_int, hy_dens_theta_int, hy_pressure_int)
        
        semi_discrete_step!(state , statetmp , state    , dt / 1 , DIR_X , flux , tend,
            recvbuf_l, recvbuf_r, sendbuf_l, sendbuf_r, hy_dens_cell, hy_dens_theta_cell,
            hy_dens_int, hy_dens_theta_int, hy_pressure_int)
        
        #z-direction second
        semi_discrete_step!(state , state    , statetmp , dt / 3 , DIR_Z , flux , tend,
            recvbuf_l, recvbuf_r, sendbuf_l, sendbuf_r, hy_dens_cell, hy_dens_theta_cell,
            hy_dens_int, hy_dens_theta_int, hy_pressure_int)
        
        semi_discrete_step!(state , statetmp , statetmp , dt / 2 , DIR_Z , flux , tend,
            recvbuf_l, recvbuf_r, sendbuf_l, sendbuf_r, hy_dens_cell, hy_dens_theta_cell,
            hy_dens_int, hy_dens_theta_int, hy_pressure_int)
        
        semi_discrete_step!(state , statetmp , state    , dt / 1 , DIR_Z , flux , tend,
            recvbuf_l, recvbuf_r, sendbuf_l, sendbuf_r, hy_dens_cell, hy_dens_theta_cell,
            hy_dens_int, hy_dens_theta_int, hy_pressure_int)
        
    else
        
        #z-direction second
        semi_discrete_step!(state , state    , statetmp , dt / 3 , DIR_Z , flux , tend,
            recvbuf_l, recvbuf_r, sendbuf_l, sendbuf_r, hy_dens_cell, hy_dens_theta_cell,
            hy_dens_int, hy_dens_theta_int, hy_pressure_int)
        
        semi_discrete_step!(state , statetmp , statetmp , dt / 2 , DIR_Z , flux , tend,
            recvbuf_l, recvbuf_r, sendbuf_l, sendbuf_r, hy_dens_cell, hy_dens_theta_cell,
            hy_dens_int, hy_dens_theta_int, hy_pressure_int)
        
        semi_discrete_step!(state , statetmp , state    , dt / 1 , DIR_Z , flux , tend,
            recvbuf_l, recvbuf_r, sendbuf_l, sendbuf_r, hy_dens_cell, hy_dens_theta_cell,
            hy_dens_int, hy_dens_theta_int, hy_pressure_int)
        
        #x-direction first
        semi_discrete_step!(state , state    , statetmp , dt / 3 , DIR_X , flux , tend,
            recvbuf_l, recvbuf_r, sendbuf_l, sendbuf_r, hy_dens_cell, hy_dens_theta_cell,
            hy_dens_int, hy_dens_theta_int, hy_pressure_int)
        
        semi_discrete_step!(state , statetmp , statetmp , dt / 2 , DIR_X , flux , tend,
            recvbuf_l, recvbuf_r, sendbuf_l, sendbuf_r, hy_dens_cell, hy_dens_theta_cell,
            hy_dens_int, hy_dens_theta_int, hy_pressure_int)
        
        semi_discrete_step!(state , statetmp , state    , dt / 1 , DIR_X , flux , tend,
            recvbuf_l, recvbuf_r, sendbuf_l, sendbuf_r, hy_dens_cell, hy_dens_theta_cell,
            hy_dens_int, hy_dens_theta_int, hy_pressure_int)
    end

end

        
#Perform a single semi-discretized step in time with the form:
#state_out = state_init + dt * rhs(state_forcing)
#Meaning the step starts from state_init, computes the rhs using state_forcing, and stores the result in state_out
function semi_discrete_step!(stateinit::OffsetArray{Float64, 3, Array{Float64, 3}},
                    stateforcing::OffsetArray{Float64, 3, Array{Float64, 3}},
                    stateout::OffsetArray{Float64, 3, Array{Float64, 3}},
                    dt::Float64,
                    dir::Int,
                    flux::Array{Float64, 3},
                    tend::Array{Float64, 3},
                    recvbuf_l::Array{Float64, 3},
                    recvbuf_r::Array{Float64, 3},
                    sendbuf_l::Array{Float64, 3},
                    sendbuf_r::Array{Float64, 3},
                    hy_dens_cell::OffsetVector{Float64, Vector{Float64}},
                    hy_dens_theta_cell::OffsetVector{Float64, Vector{Float64}},
                    hy_dens_int::Vector{Float64},
                    hy_dens_theta_int::Vector{Float64},
                    hy_pressure_int::Vector{Float64})

    if dir == DIR_X
        #Set the halo values for this MPI task's fluid state in the x-direction
        set_halo_values_x!(stateforcing, recvbuf_l, recvbuf_r, sendbuf_l,
                           sendbuf_r, hy_dens_cell, hy_dens_theta_cell)

        #Compute the time tendencies for the fluid state in the x-direction
        compute_tendencies_x!(stateforcing,flux,tend,dt, hy_dens_cell, hy_dens_theta_cell)

        
    elseif dir == DIR_Z
        #Set the halo values for this MPI task's fluid state in the z-direction
        set_halo_values_z!(stateforcing, hy_dens_cell, hy_dens_theta_cell)
        
        #Compute the time tendencies for the fluid state in the z-direction
        compute_tendencies_z_accel!(stateforcing,flux,tend,dt,
                    hy_dens_int, hy_dens_theta_int, hy_pressure_int)
        
    end
  
    #Apply the tendencies to the fluid state
    for ll in 1:NUM_VARS
        for k in 1:NZ
            for i in 1:NX
                stateout[i,k,ll] = stateinit[i,k,ll] + dt * tend[i,k,ll]
            end
        end
    end
end

#Set this MPI task's halo values in the x-direction. This routine will require MPI
function set_halo_values_x!(state::OffsetArray{Float64, 3, Array{Float64, 3}},
                    recvbuf_l::Array{Float64, 3},
                    recvbuf_r::Array{Float64, 3},
                    sendbuf_l::Array{Float64, 3},
                    sendbuf_r::Array{Float64, 3},
                    hy_dens_cell::OffsetVector{Float64, Vector{Float64}},
                    hy_dens_theta_cell::OffsetVector{Float64, Vector{Float64}})


    local req_r = Vector{Request}(undef, 2)
    local req_s = Vector{Request}(undef, 2)

    
    #Prepost receives
    req_r[1] = Irecv!(recvbuf_l, LEFT_RANK,0,COMM)
    req_r[2] = Irecv!(recvbuf_r,RIGHT_RANK,1,COMM)

    #Pack the send buffers
    for ll in 1:NUM_VARS
        for k in 1:NZ
            for s in 1:HS
                sendbuf_l[s,k,ll] = state[s      ,k,ll]
                sendbuf_r[s,k,ll] = state[NX-HS+s,k,ll]
            end
        end
    end

    #Fire off the sends
    req_s[1] = Isend(sendbuf_l, LEFT_RANK,1,COMM)
    req_s[2] = Isend(sendbuf_r,RIGHT_RANK,0,COMM)

    #Wait for receives to finish
    local statuses = Waitall!(req_r)

    #Unpack the receive buffers
    for ll in 1:NUM_VARS
        for k in 1:NZ
            for s in 1:HS
                state[-HS+s,k,ll] = recvbuf_l[s,k,ll]
                state[ NX+s,k,ll] = recvbuf_r[s,k,ll]
            end
        end
    end

    #Wait for sends to finish
    local statuses = Waitall!(req_s)
    
    if (DATA_SPEC == DATA_SPEC_INJECTION)
       if (MYRANK == 0)
          for k in 1:NZ
              z = (K_BEG-1 + k-0.5)*DZ
              if (abs(z-3*ZLEN/4) <= ZLEN/16) 
                 state[-1:0,k,ID_UMOM] = (state[-1:0,k,ID_DENS]+hy_dens_cell[k]) * 50.0
                 state[-1:0,k,ID_RHOT] = (state[-1:0,k,ID_DENS]+hy_dens_cell[k]) * 298.0 - hy_dens_theta_cell[k]
              end
          end
       end
    end
 
end

function compute_tendencies_x!(state::OffsetArray{Float64, 3, Array{Float64, 3}},
                    flux::Array{Float64, 3},
                    tend::Array{Float64, 3},
                    dt::Float64,
                    hy_dens_cell::OffsetVector{Float64, Vector{Float64}},
                    hy_dens_theta_cell::OffsetVector{Float64, Vector{Float64}})


	@jenterdata myaccel update(state)

    @jlaunch(tend_x_kernel, state, dt,hy_dens_cell, hy_dens_theta_cell ; output=(flux, tend,))

	@jexitdata myaccel update(tend)

end

#Set this MPI task's halo values in the z-direction. This does not require MPI because there is no MPI
#decomposition in the vertical direction
function set_halo_values_z!(state::OffsetArray{Float64, 3, Array{Float64, 3}},
                    hy_dens_cell::OffsetVector{Float64, Vector{Float64}},
                    hy_dens_theta_cell::OffsetVector{Float64, Vector{Float64}})
    
    for ll in 1:NUM_VARS
        for i in 1-HS:NX+HS
            if (ll == ID_WMOM)
               state[i,-1  ,ll] = 0
               state[i,0   ,ll] = 0
               state[i,NZ+1,ll] = 0
               state[i,NZ+2,ll] = 0
            elseif (ll == ID_UMOM)
               state[i,-1  ,ll] = state[i,1 ,ll] / hy_dens_cell[ 1] * hy_dens_cell[-1  ]
               state[i,0   ,ll] = state[i,1 ,ll] / hy_dens_cell[ 1] * hy_dens_cell[ 0  ]
               state[i,NZ+1,ll] = state[i,NZ,ll] / hy_dens_cell[NZ] * hy_dens_cell[NZ+1]
               state[i,NZ+2,ll] = state[i,NZ,ll] / hy_dens_cell[NZ] * hy_dens_cell[NZ+2]
            else
               state[i,-1  ,ll] = state[i,1 ,ll]
               state[i,0   ,ll] = state[i,1 ,ll]
               state[i,NZ+1,ll] = state[i,NZ,ll]
               state[i,NZ+2,ll] = state[i,NZ,ll]
            end
        end
    end

end

function compute_tendencies_z_accel!(state::OffsetArray{Float64, 3, Array{Float64, 3}},
                    flux::Array{Float64, 3},
                    tend::Array{Float64, 3},
                    dt::Float64,
                    hy_dens_int::Vector{Float64},
                    hy_dens_theta_int::Vector{Float64},
                    hy_pressure_int::Vector{Float64})

	@jenterdata myaccel update(state)

    @jlaunch(tend_z_kernel, state, dt, hy_dens_int, hy_dens_theta_int, hy_pressure_int; output=(flux, tend,))

	@jexitdata myaccel update(tend)

end

function reductions_accel(state::OffsetArray{Float64, 3, Array{Float64, 3}},
                    hy_dens_cell::OffsetVector{Float64, Vector{Float64}},
                    hy_dens_theta_cell::OffsetVector{Float64, Vector{Float64}})
    
    local mass = zero(Float64)
    local te = zero(Float64)
    glob = Array{Float64}(undef, 2)

	@jenterdata myaccel update(state)

    @jlaunch(reduce_kernel, state, hy_dens_cell, hy_dens_theta_cell; output=(glob,))

    mass = glob[1]
    te = glob[2]

    Allreduce!(Array{Float64}([mass,te]), glob, +, COMM)
    
    return glob
end

#Output the fluid state (state) to a NetCDF file at a given elapsed model time (etime)
function output(state::OffsetArray{Float64, 3, Array{Float64, 3}},
                etime::Float64,
                nt::Int,
                hy_dens_cell::OffsetVector{Float64, Vector{Float64}},
                hy_dens_theta_cell::OffsetVector{Float64, Vector{Float64}})

    var_local  = zeros(Float64, NX, NZ, NUM_VARS)

    if MASTERPROC
       var_global  = zeros(Float64, NX_GLOB, NZ_GLOB, NUM_VARS)
    end

    #Store perturbed values in the temp arrays for output
    for k in 1:NZ
        for i in 1:NX
            var_local[i,k,ID_DENS]  = state[i,k,ID_DENS]
            var_local[i,k,ID_UMOM]  = state[i,k,ID_UMOM] / ( hy_dens_cell[k] + state[i,k,ID_DENS] )
            var_local[i,k,ID_WMOM]  = state[i,k,ID_WMOM] / ( hy_dens_cell[k] + state[i,k,ID_DENS] )
            var_local[i,k,ID_RHOT] = ( state[i,k,ID_RHOT] + hy_dens_theta_cell[k] ) / ( hy_dens_cell[k] + state[i,k,ID_DENS] )
                         - hy_dens_theta_cell[k] / hy_dens_cell[k]
        end
    end

    #Gather data from multiple CPUs
    #  - Implemented in an inefficient way for the purpose of tests
    #  - Will be improved in next version.
    if MASTERPROC
       ibeg_chunk = zeros(Int,NRANKS)
       iend_chunk = zeros(Int,NRANKS)
       nchunk     = zeros(Int,NRANKS)
       for n in 1:NRANKS
          ibeg_chunk[n] = trunc(Int, round(NPER* (n-1))+1)
          iend_chunk[n] = trunc(Int, round(NPER*((n-1)+1)))
          nchunk[n]     = iend_chunk[n] - ibeg_chunk[n] + 1
       end
    end

    if MASTERPROC
       var_global[I_BEG:I_END,:,:] = var_local[:,:,:]
       if NRANKS > 1
          for i in 2:NRANKS
              var_local = Array{Float64}(undef, nchunk[i],NZ,NUM_VARS)
              status = MPI.Recv!(var_local,i-1,0,COMM)
              var_global[ibeg_chunk[i]:iend_chunk[i],:,:] = var_local[:,:,:]
          end
       end
    else
       MPI.Send(var_local,MASTERRANK,0,COMM)
    end

    # Write output only in MASTER
    if MASTERPROC

       #If the elapsed time is zero, create the file. Otherwise, open the file
       if ( etime == 0.0 )

          # Open NetCDF output file with a create mode
          ds = Dataset(OUTFILE,"c")

          defDim(ds,"t",Inf)
          defDim(ds,"x",NX_GLOB)
          defDim(ds,"z",NZ_GLOB)

          nc_var = defVar(ds,"t",Float64,("t",))
          nc_var[nt] = etime
          nc_var = defVar(ds,"dens",Float64,("x","z","t"))
          nc_var[:,:,nt] = var_global[:,:,ID_DENS]
          nc_var = defVar(ds,"uwnd",Float64,("x","z","t"))
          nc_var[:,:,nt] = var_global[:,:,ID_UMOM]
          nc_var = defVar(ds,"wwnd",Float64,("x","z","t"))
          nc_var[:,:,nt] = var_global[:,:,ID_WMOM]
          nc_var = defVar(ds,"theta",Float64,("x","z","t"))
          nc_var[:,:,nt] = var_global[:,:,ID_RHOT]

          # Close NetCDF file
          close(ds)

       else

          # Open NetCDF output file with an append mode
          ds = Dataset(OUTFILE,"a")

          nc_var = ds["t"]
          nc_var[nt] = etime
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

function finalize!(state::OffsetArray{Float64, 3, Array{Float64, 3}})

    #println(axes(state))
    
end

# invoke main function
main(ARGS)
