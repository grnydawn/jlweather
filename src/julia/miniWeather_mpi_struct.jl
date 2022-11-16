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

import MPI.Init,
       MPI.COMM_WORLD,
       MPI.Comm_rank,
       MPI.Comm_size,
       MPI.Allreduce!,
       MPI.Barrier,
       MPI.Waitall!,
       MPI.Request,
       MPI.Irecv!,
       MPI.Isend,
       MPI.Recv!,
       MPI.Send

import Debugger

import Printf.@printf

import Libdl

##############
# constants
##############
   
const FLOAT = Float64
   
# julia command to link MPI.jl to system MPI installation
# julia -e 'ENV["JULIA_MPI_BINARY"]="system"; ENV["JULIA_MPI_PATH"]="/Users/8yk/opt/usr/local"; using Pkg; Pkg.build("MPI"; verbose=true)'

Init()

const COMM   = COMM_WORLD
const NRANKS = Comm_size(COMM)
const MYRANK = Comm_rank(COMM)

s = ArgParseSettings()
@add_arg_table! s begin
    "--simtime", "-s"
        help = "simulation time"
        arg_type = FLOAT
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
        arg_type = FLOAT
        default = 400.0
    "--logfreq", "-l"
        help = "logging frequency in time"
        arg_type = FLOAT
        default = 0.0
    "--dataspec", "-d"
        help = "data spec"
        arg_type = Int64
        default = 2
    "--outfile", "-o"
        help = "output file path"
        default = "output.nc"
    "--workdir", "-w"
        help = "working directory path"
        default = ".jaitmp"
    "--debugdir", "-b"
        help = "debugging output directory path"
        default = ".jaitmp"

end

parsed_args = parse_args(ARGS, s)

const SIM_TIME    = parsed_args["simtime"]
const NX_GLOB     = parsed_args["nx"]
const NZ_GLOB     = parsed_args["nz"]
const OUT_FREQ    = parsed_args["outfreq"]
const _logfreq    = parsed_args["logfreq"]
const LOG_FREQ    = (_logfreq == 0.0) ? SIM_TIME / 10 : _logfreq
const DATA_SPEC   = parsed_args["dataspec"]
const OUTFILE     = parsed_args["outfile"]
const WORKDIR     = parsed_args["workdir"]
const DEBUGDIR    = parsed_args["debugdir"]

const NPER  = FLOAT(NX_GLOB)/NRANKS
const I_BEG = trunc(Int, round(NPER* MYRANK)+1)
const I_END = trunc(Int, round(NPER*(MYRANK+1)))
const NX    = I_END - I_BEG + 1
const NZ    = NZ_GLOB

const LEFT_RANK = MYRANK-1 == -1 ? NRANKS - 1 : MYRANK - 1 
const RIGHT_RANK = MYRANK+1 == NRANKS ? 0 : MYRANK + 1 

#Vertical direction isn't MPI-ized, so the rank's local values = the global values
const K_BEG       = 1
const MASTERRANK  = 0
const MASTERPROC  = (MYRANK == MASTERRANK)

const HS          = 2
const STEN_SIZE   = 4 #Size of the stencil used for interpolation
const NUM_VARS    = 4
const XLEN        = FLOAT(2.E4) # Length of the domain in the x-direction (meters)
const ZLEN        = FLOAT(1.E4) # Length of the domain in the z-direction (meters)
const HV_BETA     = FLOAT(0.05) # How strong to diffuse the solution: hv_beta \in [0:1]
const CFL         = FLOAT(1.5)  # "Courant, Friedrichs, Lewy" number (for numerical stability)
const MAX_SPEED   = FLOAT(450.0)# Assumed maximum wave speed during the simulation (speed of sound + speed of wind) (meter / sec)
const DX          = XLEN / NX_GLOB
const DZ          = ZLEN / NZ_GLOB
const DT          = min(DX,DZ) / MAX_SPEED * CFL
const NQPOINTS    = 3
const PI          = FLOAT(3.14159265358979323846264338327)
const GRAV        = FLOAT(9.8)
const CP          = FLOAT(1004.0) # Specific heat of dry air at constant pressure
const CV          = FLOAT(717.0)  # Specific heat of dry air at constant volume
const RD          = FLOAT(287.0)  # Dry air constant for equation of state (P=rho*rd*T)
const P0          = FLOAT(1.0E5)  # Standard pressure at the surface in Pascals
const C0          = FLOAT(27.5629410929725921310572974482)
const GAMMA       = FLOAT(1.40027894002789400278940027894)

const ID_DENS     = 1
const ID_UMOM     = 2
const ID_WMOM     = 3
const ID_RHOT     = 4
                    
const DIR_X       = 1 #Integer constant to express that this operation is in the x-direction
const DIR_Z       = 2 #Integer constant to express that this operation is in the z-direction

const DATA_SPEC_COLLISION       = 1
const DATA_SPEC_THERMAL         = 2
const DATA_SPEC_GRAVITY_WAVES   = 3
const DATA_SPEC_DENSITY_CURRENT = 5
const DATA_SPEC_INJECTION       = 6

const qpoints     = Array{FLOAT}([0.112701665379258311482073460022E0 , 0.500000000000000000000000000000E0 , 0.887298334620741688517926539980E0])
const qweights    = Array{FLOAT}([0.277777777777777777777777777779E0 , 0.444444444444444444444444444444E0 , 0.277777777777777777777777777779E0])

 
##############
# types
##############

struct VarsType{T <: FLOAT}
    state :: OffsetArray{T, 3, Array{T, 3}}
    statetmp :: OffsetArray{T, 3, Array{T, 3}}
    flux :: Array{T, 3}
    tend :: Array{T, 3}
    hy_dens_cell :: OffsetVector{T, Vector{T}}
    hy_dens_theta_cell :: OffsetVector{T, Vector{T}}
    hy_dens_int::Vector{T}
    hy_dens_theta_int::Vector{T}
    hy_pressure_int::Vector{T}
    recvbuf_l :: Array{T, 3}
    recvbuf_r :: Array{T, 3}
    sendbuf_l :: Array{T, 3}
    sendbuf_r :: Array{T, 3}

    function VarsType{T}(
            _state :: OffsetArray{T, 3, Array{T, 3}},
            _statetmp :: OffsetArray{T, 3, Array{T, 3}},
            _flux :: Array{T, 3},
            _tend :: Array{T, 3},
            _hy_dens_cell :: OffsetVector{T, Vector{T}},
            _hy_dens_theta_cell :: OffsetVector{T, Vector{T}},
            _hy_dens_int :: Vector{T},
            _hy_dens_theta_int :: Vector{T},
            _hy_pressure_int :: Vector{T},
            _recvbuf_l :: Array{T, 3},
            _recvbuf_r :: Array{T, 3},
            _sendbuf_l :: Array{T, 3},
            _sendbuf_r :: Array{T, 3}
        ) where T <: FLOAT

        new(_state, _statetmp, _flux, _tend, _hy_dens_cell, _hy_dens_theta_cell,
            _hy_dens_int, _hy_dens_theta_int, _hy_pressure_int,_recvbuf_l,
            _recvbuf_r, _sendbuf_l, _sendbuf_r)
    end    
end
 
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

    local etime = FLOAT(0.0)
    local output_counter = FLOAT(0.0)
    local log_counter = FLOAT(0.0)
    local dt = DT
    local nt = Int(1)

    #Initialize the grid and the data  
    vars = init!()

    #Initial reductions for mass, kinetic energy, and total energy
    local mass0, te0 = reductions(vars)

    #Output the initial state
    output(vars,etime,nt)

    
    # main loop
    elapsedtime = @elapsed while etime < SIM_TIME

        #If the time step leads to exceeding the simulation time, shorten it for the last step
        if etime + dt > SIM_TIME
            dt = SIM_TIME - etime
        end

        #Perform a single time step
        perform_timestep!(vars, dt)

        #Update the elapsed time and output counter
        etime = etime + dt
        output_counter = output_counter + dt
        log_counter = log_counter + dt

        #If it's time for output, reset the counter, and do output
        if (output_counter >= OUT_FREQ)
          #Increment the number of outputs
          nt = nt + 1
          output(vars, etime,nt)
          output_counter = output_counter - OUT_FREQ
        end

        if MASTERPROC && (log_counter >= LOG_FREQ)
          @printf("[%3.1f%% of %2.1f]\n", etime/SIM_TIME*100, SIM_TIME)
          log_counter = log_counter - LOG_FREQ
        end

    end
 
    local mass, te = reductions(vars)

    if MASTERPROC
        println( "CPU Time: $elapsedtime")
        @printf("d_mass: %.15e\n", (mass - mass0)/mass0)
        @printf("d_te  : %.15e\n", (te - te0)/te0)
    end
    finalize!(vars)

end

function init!()
    
    if MASTERPROC
        println("nx_glob, nz_glob: $NX_GLOB $NZ_GLOB")
        println("dx, dz: $DX $DZ")
        println("dt: $DT")
    end
        
    #println("nx, nz at $MYRANK: $NX($I_BEG:$I_END) $NZ($K_BEG:$NZ)")
    
    Barrier(COMM)
    
    _state      = zeros(FLOAT, NX+2*HS, NZ+2*HS, NUM_VARS) 
    state       = OffsetArray(_state, 1-HS:NX+HS, 1-HS:NZ+HS, 1:NUM_VARS)
    _statetmp   = Array{FLOAT}(undef, NX+2*HS, NZ+2*HS, NUM_VARS) 
    statetmp    = OffsetArray(_statetmp, 1-HS:NX+HS, 1-HS:NZ+HS, 1:NUM_VARS)
    
    flux        = zeros(FLOAT, NX+1, NZ+1, NUM_VARS) 
    tend        = zeros(FLOAT, NX, NZ, NUM_VARS) 
 
    _hy_dens_cell       = zeros(FLOAT, NZ+2*HS) 
    hy_dens_cell        = OffsetArray(_hy_dens_cell, 1-HS:NZ+HS)
    _hy_dens_theta_cell = zeros(FLOAT, NZ+2*HS) 
    hy_dens_theta_cell  = OffsetArray(_hy_dens_theta_cell, 1-HS:NZ+HS)   
    
    hy_dens_int         = Array{FLOAT}(undef, NZ+1)
    hy_dens_theta_int   = Array{FLOAT}(undef, NZ+1)
    hy_pressure_int     = Array{FLOAT}(undef, NZ+1)   
    
    sendbuf_l = Array{FLOAT}(undef, HS, NZ, NUM_VARS)
    sendbuf_r = Array{FLOAT}(undef, HS, NZ, NUM_VARS)
    recvbuf_l = Array{FLOAT}(undef, HS, NZ, NUM_VARS)
    recvbuf_r = Array{FLOAT}(undef, HS, NZ, NUM_VARS)   
    
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
            if(DATA_SPEC==DATA_SPEC_COLLISION)      ; r,u,w,t,hr,ht = collision!(x,z)      ; end
            if(DATA_SPEC==DATA_SPEC_THERMAL)        ; r,u,w,t,hr,ht = thermal!(x,z)        ; end
            if(DATA_SPEC==DATA_SPEC_GRAVITY_WAVES)  ; r,u,w,t,hr,ht = gravity_waves!(x,z)  ; end
            if(DATA_SPEC==DATA_SPEC_DENSITY_CURRENT); r,u,w,t,hr,ht = density_current!(x,z); end
            if(DATA_SPEC==DATA_SPEC_INJECTION)      ; r,u,w,t,hr,ht = injection!(x,z)      ; end

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
            if(DATA_SPEC==DATA_SPEC_COLLISION)      ; r,u,w,t,hr,ht = collision!(0.0,z)      ; end
            if(DATA_SPEC==DATA_SPEC_THERMAL)        ; r,u,w,t,hr,ht = thermal!(0.0,z)        ; end
            if(DATA_SPEC==DATA_SPEC_GRAVITY_WAVES)  ; r,u,w,t,hr,ht = gravity_waves!(0.0,z)  ; end
            if(DATA_SPEC==DATA_SPEC_DENSITY_CURRENT); r,u,w,t,hr,ht = density_current!(0.0,z); end
            if(DATA_SPEC==DATA_SPEC_INJECTION)      ; r,u,w,t,hr,ht = injection!(0.0,z)      ; end

            hy_dens_cell[k]       = hy_dens_cell[k]       + hr    * qweights[kk]
            hy_dens_theta_cell[k] = hy_dens_theta_cell[k] + hr*ht * qweights[kk]
        end
    end
    
    #Compute the hydrostatic background state at vertical cell interfaces
    for k in 1:NZ+1
        z = (K_BEG-1 + k-1) * DZ
        #Set the fluid state based on the user's specification
        if(DATA_SPEC==DATA_SPEC_COLLISION)      ; r,u,w,t,hr,ht = collision!(0.0,z)      ; end
        if(DATA_SPEC==DATA_SPEC_THERMAL)        ; r,u,w,t,hr,ht = thermal!(0.0,z)        ; end
        if(DATA_SPEC==DATA_SPEC_GRAVITY_WAVES)  ; r,u,w,t,hr,ht = gravity_waves!(0.0,z)  ; end
        if(DATA_SPEC==DATA_SPEC_DENSITY_CURRENT); r,u,w,t,hr,ht = density_current!(0.0,z); end
        if(DATA_SPEC==DATA_SPEC_INJECTION)      ; r,u,w,t,hr,ht = injection!(0.0,z)      ; end

        hy_dens_int[k] = hr
        hy_dens_theta_int[k] = hr*ht
        hy_pressure_int[k] = C0*(hr*ht)^GAMMA
    end

    vars = VarsType{FLOAT}(state, statetmp, flux, tend, hy_dens_cell, hy_dens_theta_cell,
            hy_dens_int, hy_dens_theta_int, hy_pressure_int, sendbuf_l,
            sendbuf_r, recvbuf_l, recvbuf_r)
    #println("EEEEE", methods(VarsType{FLOAT}))

    return vars
end

function injection!(x::FLOAT, z::FLOAT)

    #Hydrostatic density and potential temperature
    hr,ht = hydro_const_theta!(z)

    r  = FLOAT(0.0) # Density
    t  = FLOAT(0.0) # Potential temperature
    u  = FLOAT(0.0) # Uwind
    w  = FLOAT(0.0) # Wwind
    
    return r, u, w, t, hr, ht
end

function density_current!(x::FLOAT, z::FLOAT)

    #Hydrostatic density and potential temperature
    hr,ht = hydro_const_theta!(z)

    r  = FLOAT(0.0) # Density
    t  = FLOAT(0.0) # Potential temperature
    u  = FLOAT(0.0) # Uwind
    w  = FLOAT(0.0) # Wwind
    
    t = t + sample_ellipse_cosine!(x,z,-20.0,XLEN/2,5000.0,4000.0,2000.0)

    return r, u, w, t, hr, ht
end

function gravity_waves!(x::FLOAT, z::FLOAT)

    #Hydrostatic density and potential temperature
    hr,ht = hydro_const_bvfreq!(z, FLOAT(0.02))

    r  = FLOAT(0.0) # Density
    t  = FLOAT(0.0) # Potential temperature
    u  = FLOAT(15.0) # Uwind
    w  = FLOAT(0.0) # Wwind
    
    return r, u, w, t, hr, ht
end


#Rising thermal
function thermal!(x::FLOAT, z::FLOAT)

    #Hydrostatic density and potential temperature
    hr,ht = hydro_const_theta!(z)

    r  = FLOAT(0.0) # Density
    t  = FLOAT(0.0) # Potential temperature
    u  = FLOAT(0.0) # Uwind
    w  = FLOAT(0.0) # Wwind
    
    t = t + sample_ellipse_cosine!(x,z,3.0,XLEN/2,2000.0,2000.0,2000.0) 

    return r, u, w, t, hr, ht
end

#Colliding thermals
function collision!(x::FLOAT, z::FLOAT)
    
    #Hydrostatic density and potential temperature
    hr,ht = hydro_const_theta!(z)

    r  = FLOAT(0.0) # Density
    t  = FLOAT(0.0) # Potential temperature
    u  = FLOAT(0.0) # Uwind
    w  = FLOAT(0.0) # Wwind

    t = t + sample_ellipse_cosine!(x,z, 20.0,XLEN/2,2000.0,2000.0,2000.0)
    t = t + sample_ellipse_cosine!(x,z,-20.0,XLEN/2,8000.0,2000.0,2000.0)

    return r, u, w, t, hr, ht
end

#Establish hydrstatic balance using constant potential temperature (thermally neutral atmosphere)
function hydro_const_theta!(z::FLOAT)

    r      = FLOAT(0.0) # Density
    t      = FLOAT(0.0) # Potential temperature

    theta0 = FLOAT(300.0) # Background potential temperature
    exner0 = FLOAT(1.0)   # Surface-level Exner pressure

    t      = theta0                            # Potential temperature at z
    exner  = exner0 - GRAV * z / (CP * theta0) # Exner pressure at z
    p      = P0 * exner^(CP/RD)                # Pressure at z
    rt     = (p / C0)^(FLOAT(1.0)/GAMMA)     # rho*theta at z
    r      = rt / t                            # Density at z

    return r, t
end

function hydro_const_bvfreq!(z::FLOAT, bv_freq0::FLOAT)

    r      = FLOAT(0.0) # Density
    t      = FLOAT(0.0) # Potential temperature

    theta0 = FLOAT(300.0) # Background potential temperature
    exner0 = FLOAT(1.0)   # Surface-level Exner pressure

    t      = theta0 * exp(bv_freq0^FLOAT(2.0) / GRAV * z) # Potential temperature at z
    exner  = exner0 - GRAV^FLOAT(2.0) / (CP * bv_freq0^FLOAT(2.0)) * (t - theta0) / (t * theta0) # Exner pressure at z
    p      = P0 * exner^(CP/RD)                # Pressure at z
    rt     = (p / C0)^(FLOAT(1.0)/GAMMA)     # rho*theta at z
    r      = rt / t                            # Density at z

    return r, t
end


#Sample from an ellipse of a specified center, radius, and amplitude at a specified location
function sample_ellipse_cosine!(   x::FLOAT,    z::FLOAT, amp::FLOAT, 
                                  x0::FLOAT,   z0::FLOAT, 
                                xrad::FLOAT, zrad::FLOAT )

    #Compute distance from bubble center
    local dist = sqrt( ((x-x0)/xrad)^2 + ((z-z0)/zrad)^2 ) * PI / FLOAT(2.0)
 
    #If the distance from bubble center is less than the radius, create a cos**2 profile
    if (dist <= PI / FLOAT(2.0) ) 
      val = amp * cos(dist)^2
    else
      val = FLOAT(0.0)
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
function perform_timestep!(vars::VarsType{FLOAT}, dt::FLOAT)
    
    local direction_switch = true
    
    if direction_switch
        
        #x-direction first
        semi_discrete_step!(vars.state , vars.state    , vars.statetmp , dt / 3 , DIR_X ,
            vars.flux , vars.tend, vars.recvbuf_l, vars.recvbuf_r, vars.sendbuf_l,
            vars.sendbuf_r, vars.hy_dens_cell, vars.hy_dens_theta_cell,
            vars.hy_dens_int, vars.hy_dens_theta_int, vars.hy_pressure_int)
        
        semi_discrete_step!(vars.state , vars.statetmp , vars.statetmp , dt / 2 , DIR_X ,
            vars.flux , vars.tend, vars.recvbuf_l, vars.recvbuf_r, vars.sendbuf_l,
            vars.sendbuf_r, vars.hy_dens_cell, vars.hy_dens_theta_cell,
            vars.hy_dens_int, vars.hy_dens_theta_int, vars.hy_pressure_int)
        
        semi_discrete_step!(vars.state , vars.statetmp , vars.state    , dt / 1 , DIR_X ,
            vars.flux , vars.tend, vars.recvbuf_l, vars.recvbuf_r, vars.sendbuf_l,
            vars.sendbuf_r, vars.hy_dens_cell, vars.hy_dens_theta_cell,
            vars.hy_dens_int, vars.hy_dens_theta_int, vars.hy_pressure_int)
        
        #z-direction second
        semi_discrete_step!(vars.state , vars.state    , vars.statetmp , dt / 3 , DIR_Z ,
            vars.flux , vars.tend, vars.recvbuf_l, vars.recvbuf_r, vars.sendbuf_l,
            vars.sendbuf_r, vars.hy_dens_cell, vars.hy_dens_theta_cell,
            vars.hy_dens_int, vars.hy_dens_theta_int, vars.hy_pressure_int)
        
        semi_discrete_step!(vars.state , vars.statetmp , vars.statetmp , dt / 2 , DIR_Z ,
            vars.flux , vars.tend, vars.recvbuf_l, vars.recvbuf_r, vars.sendbuf_l,
            vars.sendbuf_r, vars.hy_dens_cell, vars.hy_dens_theta_cell,
            vars.hy_dens_int, vars.hy_dens_theta_int, vars.hy_pressure_int)
        
        semi_discrete_step!(vars.state , vars.statetmp , vars.state    , dt / 1 , DIR_Z ,
            vars.flux , vars.tend, vars.recvbuf_l, vars.recvbuf_r, vars.sendbuf_l,
            vars.sendbuf_r, vars.hy_dens_cell, vars.hy_dens_theta_cell,
            vars.hy_dens_int, vars.hy_dens_theta_int, vars.hy_pressure_int)
        
    else
        
        #z-direction second
        semi_discrete_step!(vars.state , vars.state    , vars.statetmp , dt / 3 , DIR_Z ,
            vars.flux , vars.tend, vars.recvbuf_l, vars.recvbuf_r, vars.sendbuf_l,
            vars.sendbuf_r, vars.hy_dens_cell, vars.hy_dens_theta_cell,
            vars.hy_dens_int, vars.hy_dens_theta_int, vars.hy_pressure_int)
        
        semi_discrete_step!(vars.state , vars.statetmp , vars.statetmp , dt / 2 , DIR_Z ,
            vars.flux , vars.tend, vars.recvbuf_l, vars.recvbuf_r, vars.sendbuf_l,
            vars.sendbuf_r, vars.hy_dens_cell, vars.hy_dens_theta_cell,
            vars.hy_dens_int, vars.hy_dens_theta_int, vars.hy_pressure_int)
        
        semi_discrete_step!(vars.state , vars.statetmp , vars.state    , dt / 1 , DIR_Z ,
            vars.flux , vars.tend, vars.recvbuf_l, vars.recvbuf_r, vars.sendbuf_l,
            vars.sendbuf_r, vars.hy_dens_cell, vars.hy_dens_theta_cell,
            vars.hy_dens_int, vars.hy_dens_theta_int, vars.hy_pressure_int)
        
        #x-direction first
        semi_discrete_step!(vars.state , vars.state    , vars.statetmp , dt / 3 , DIR_X ,
            vars.flux , vars.tend, vars.recvbuf_l, vars.recvbuf_r, vars.sendbuf_l,
            vars.sendbuf_r, vars.hy_dens_cell, vars.hy_dens_theta_cell,
            vars.hy_dens_int, vars.hy_dens_theta_int, vars.hy_pressure_int)
        
        semi_discrete_step!(vars.state , vars.statetmp , vars.statetmp , dt / 2 , DIR_X ,
            vars.flux , vars.tend, vars.recvbuf_l, vars.recvbuf_r, vars.sendbuf_l,
            vars.sendbuf_r, vars.hy_dens_cell, vars.hy_dens_theta_cell,
            vars.hy_dens_int, vars.hy_dens_theta_int, vars.hy_pressure_int)
        
        semi_discrete_step!(vars.state , vars.statetmp , vars.state    , dt / 1 , DIR_X ,
            vars.flux , vars.tend, vars.recvbuf_l, vars.recvbuf_r, vars.sendbuf_l,
            vars.sendbuf_r, vars.hy_dens_cell, vars.hy_dens_theta_cell,
            vars.hy_dens_int, vars.hy_dens_theta_int, vars.hy_pressure_int)
    end

end

        
#Perform a single semi-discretized step in time with the form:
#state_out = state_init + dt * rhs(state_forcing)
#Meaning the step starts from state_init, computes the rhs using state_forcing, and stores the result in state_out
function semi_discrete_step!(state_init::OffsetArray{FLOAT, 3, Array{FLOAT, 3}},
                    state_forcing::OffsetArray{FLOAT, 3, Array{FLOAT, 3}},
                    state_out::OffsetArray{FLOAT, 3, Array{FLOAT, 3}},
                    dt::FLOAT,
                    dir::Int,
                    flux::Array{FLOAT, 3},
                    tend::Array{FLOAT, 3},
                    recvbuf_l::Array{FLOAT, 3},
                    recvbuf_r::Array{FLOAT, 3},
                    sendbuf_l::Array{FLOAT, 3},
                    sendbuf_r::Array{FLOAT, 3},
                    hy_dens_cell::OffsetVector{FLOAT, Vector{FLOAT}},
                    hy_dens_theta_cell::OffsetVector{FLOAT, Vector{FLOAT}},
                    hy_dens_int::Vector{FLOAT},
                    hy_dens_theta_int::Vector{FLOAT},
                    hy_pressure_int::Vector{FLOAT})

    if dir == DIR_X
        #Set the halo values for this MPI task's fluid state in the x-direction
        set_halo_values_x!(state_forcing, recvbuf_l, recvbuf_r, sendbuf_l,
                           sendbuf_r, hy_dens_cell, hy_dens_theta_cell)

        #Compute the time tendencies for the fluid state in the x-direction
        compute_tendencies_x!(state_forcing,flux,tend,dt, hy_dens_cell, hy_dens_theta_cell)

        
    elseif dir == DIR_Z
        #Set the halo values for this MPI task's fluid state in the z-direction
        set_halo_values_z!(state_forcing, hy_dens_cell, hy_dens_theta_cell)
        
        #Compute the time tendencies for the fluid state in the z-direction
        compute_tendencies_z!(state_forcing,flux,tend,dt,
                    hy_dens_int, hy_dens_theta_int, hy_pressure_int)
        
    end
  
    #Apply the tendencies to the fluid state
    for ll in 1:NUM_VARS
        for k in 1:NZ
            for i in 1:NX
                if DATA_SPEC == DATA_SPEC_GRAVITY_WAVES
                    x = (I_BEG-1 + i-FLOAT(0.5)) * DX
                    z = (K_BEG-1 + k-FLOAT(0.5)) * DZ
                    # The following requires "acc routine" in OpenACC and "declare target" in OpenMP offload
                    # Neither of these are particularly well supported by compilers, so I'm manually inlining
                    # wpert = sample_ellipse_cosine( x,z , 0.01_rp , xlen/8,1000._rp, 500._rp,500._rp )
                    x0 = XLEN/FLOAT(8.)
                    z0 = FLOAT(1000.0)
                    xrad = FLOAT(500.)
                    zrad = FLOAT(500.)
                    amp = FLOAT(0.01)
                    #Compute distance from bubble center
                    dist = sqrt( ((x-x0)/xrad)^FLOAT(2.0) + ((z-z0)/zrad)^FLOAT(2.0) ) * PI / FLOAT(2.0)
                    #If the distance from bubble center is less than the radius, create a cos**2 profile
                    if dist <= PI / FLOAT(2.0)
                        wpert = amp * cos(dist)^FLOAT(2.0)
                    else
                        wpert = FLOAT(0.0)
                    end
                    tend[i,k,ID_WMOM] = tend[i,k,ID_WMOM] + wpert*hy_dens_cell[k]
                end

                state_out[i,k,ll] = state_init[i,k,ll] + dt * tend[i,k,ll]
            end
        end
    end

end

#Set this MPI task's halo values in the x-direction. This routine will require MPI
function set_halo_values_x!(state::OffsetArray{FLOAT, 3, Array{FLOAT, 3}},
                    recvbuf_l::Array{FLOAT, 3},
                    recvbuf_r::Array{FLOAT, 3},
                    sendbuf_l::Array{FLOAT, 3},
                    sendbuf_r::Array{FLOAT, 3},
                    hy_dens_cell::OffsetVector{FLOAT, Vector{FLOAT}},
                    hy_dens_theta_cell::OffsetVector{FLOAT, Vector{FLOAT}})

    if NRANKS == 1
        for ll in 1:NUM_VARS
            for k in 1:NZ
                state[-1  ,k,ll] = state[NX-1,k,ll]
                state[0   ,k,ll] = state[NX  ,k,ll]
                state[NX+1,k,ll] = state[1   ,k,ll]
                state[NX+2,k,ll] = state[2   ,k,ll]
            end
        end
        return
    end


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

function compute_tendencies_x!(state::OffsetArray{FLOAT, 3, Array{FLOAT, 3}},
                    flux::Array{FLOAT, 3},
                    tend::Array{FLOAT, 3},
                    dt::FLOAT,
                    hy_dens_cell::OffsetVector{FLOAT, Vector{FLOAT}},
                    hy_dens_theta_cell::OffsetVector{FLOAT, Vector{FLOAT}})

    local stencil = Array{FLOAT}(undef, STEN_SIZE)
    local d3_vals = Array{FLOAT}(undef, NUM_VARS)
    local vals    = Array{FLOAT}(undef, NUM_VARS)
    local (r, u, w, t, p) = [zero(FLOAT) for _ in 1:5]
    
    #Compute the hyperviscosity coeficient
    local hv_coef = -HV_BETA * DX / (16*dt)
    
    for k in 1:NZ
        for i in 1:(NX+1)
            #Use fourth-order interpolation from four cell averages to compute the value at the interface in question
            for ll in 1:NUM_VARS
                for s in 1:STEN_SIZE
                    stencil[s] = state[i-HS-1+s,k,ll]
                end # s
                #Fourth-order-accurate interpolation of the state
                vals[ll] = -stencil[1]/12 + 7*stencil[2]/12 + 7*stencil[3]/12 - stencil[4]/12
                #First-order-accurate interpolation of the third spatial derivative of the state (for artificial viscosity)
                d3_vals[ll] = -stencil[1] + 3*stencil[2] - 3*stencil[3] + stencil[4]
            end # ll
 

            #Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
            r = vals[ID_DENS] + hy_dens_cell[k]
            u = vals[ID_UMOM] / r
            w = vals[ID_WMOM] / r
            t = ( vals[ID_RHOT] + hy_dens_theta_cell[k] ) / r
            p = C0*(r*t)^GAMMA

            #Compute the flux vector
            flux[i,k,ID_DENS] = r*u     - hv_coef*d3_vals[ID_DENS]
            flux[i,k,ID_UMOM] = r*u*u+p - hv_coef*d3_vals[ID_UMOM]
            flux[i,k,ID_WMOM] = r*u*w   - hv_coef*d3_vals[ID_WMOM]
            flux[i,k,ID_RHOT] = r*u*t   - hv_coef*d3_vals[ID_RHOT]
        end # i
    end # k
    
    for ll in 1:NUM_VARS
        for k in 1:NZ
            for i in 1:NX
                tend[i,k,ll] = -( flux[i+1,k,ll] - flux[i,k,ll] ) / DX
            end
        end
    end

end

#Set this MPI task's halo values in the z-direction. This does not require MPI because there is no MPI
#decomposition in the vertical direction
function set_halo_values_z!(state::OffsetArray{FLOAT, 3, Array{FLOAT, 3}},
                    hy_dens_cell::OffsetVector{FLOAT, Vector{FLOAT}},
                    hy_dens_theta_cell::OffsetVector{FLOAT, Vector{FLOAT}})
    
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
        
function compute_tendencies_z!(state::OffsetArray{FLOAT, 3, Array{FLOAT, 3}},
                    flux::Array{FLOAT, 3},
                    tend::Array{FLOAT, 3},
                    dt::FLOAT,
                    hy_dens_int::Vector{FLOAT},
                    hy_dens_theta_int::Vector{FLOAT},
                    hy_pressure_int::Vector{FLOAT})
    
    local stencil = Array{FLOAT}(undef, STEN_SIZE)
    local d3_vals = Array{FLOAT}(undef, NUM_VARS)
    local vals    = Array{FLOAT}(undef, NUM_VARS)
    local (r, u, w, t, p) = [zero(FLOAT) for _ in 1:5]
 
    #Compute the hyperviscosity coeficient
    local hv_coef = -HV_BETA * DZ / (16*dt)

    #Compute fluxes in the x-direction for each cell
    for k in 1:NZ+1
      for i in 1:NX
        #Use fourth-order interpolation from four cell averages to compute the value at the interface in question
        for ll in 1:NUM_VARS
          for s in 1:STEN_SIZE
            stencil[s] = state[i,k-HS-1+s,ll]
          end # s
          #Fourth-order-accurate interpolation of the state
          vals[ll] = -stencil[1]/12 + 7*stencil[2]/12 + 7*stencil[3]/12 - stencil[4]/12
          #First-order-accurate interpolation of the third spatial derivative of the state
          d3_vals[ll] = -stencil[1] + 3*stencil[2] - 3*stencil[3] + stencil[4]
        end # ll

        #Compute density, u-wind, w-wind, potential temperature, and pressure (r,u,w,t,p respectively)
        r = vals[ID_DENS] + hy_dens_int[k]
        u = vals[ID_UMOM] / r
        w = vals[ID_WMOM] / r
        t = ( vals[ID_RHOT] + hy_dens_theta_int[k] ) / r
        p = C0*(r*t)^GAMMA - hy_pressure_int[k]
        #Enforce vertical boundary condition and exact mass conservation
        if (k == 1 || k == NZ+1) 
          w                = 0
          d3_vals[ID_DENS] = 0
        end

        #Compute the flux vector with hyperviscosity
        flux[i,k,ID_DENS] = r*w     - hv_coef*d3_vals[ID_DENS]
        flux[i,k,ID_UMOM] = r*w*u   - hv_coef*d3_vals[ID_UMOM]
        flux[i,k,ID_WMOM] = r*w*w+p - hv_coef*d3_vals[ID_WMOM]
        flux[i,k,ID_RHOT] = r*w*t   - hv_coef*d3_vals[ID_RHOT]
      end
    end

    #Use the fluxes to compute tendencies for each cell
    for ll in 1:NUM_VARS
        for k in 1:NZ
            for i in 1:NX
                tend[i,k,ll] = -( flux[i,k+1,ll] - flux[i,k,ll] ) / DZ
                if (ll == ID_WMOM)
                   tend[i,k,ID_WMOM] = tend[i,k,ID_WMOM] - state[i,k,ID_DENS]*GRAV
                end
            end
        end
    end


end

function reductions(vars::VarsType{FLOAT})
    
    local mass, te, r, u, w, th, p, t, ke, le = [zero(FLOAT) for _ in 1:10] 
    glob = Array{FLOAT}(undef, 2)
    
    for k in 1:NZ
        for i in 1:NX
            r  =   vars.state[i,k,ID_DENS] + vars.hy_dens_cell[k]             # Density
            u  =   vars.state[i,k,ID_UMOM] / r                           # U-wind
            w  =   vars.state[i,k,ID_WMOM] / r                           # W-wind
            th = ( vars.state[i,k,ID_RHOT] + vars.hy_dens_theta_cell[k] ) / r # Potential Temperature (theta)
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
function output(vars::VarsType{FLOAT},
                etime::FLOAT,
                nt::Int)

    var_local  = zeros(FLOAT, NX, NZ, NUM_VARS)

    if MASTERPROC
       var_global  = zeros(FLOAT, NX_GLOB, NZ_GLOB, NUM_VARS)
    end

    #Store perturbed values in the temp arrays for output
    for k in 1:NZ
        for i in 1:NX
            var_local[i,k,ID_DENS]  = vars.state[i,k,ID_DENS]
            var_local[i,k,ID_UMOM]  = vars.state[i,k,ID_UMOM] / ( vars.hy_dens_cell[k] + vars.state[i,k,ID_DENS] )
            var_local[i,k,ID_WMOM]  = vars.state[i,k,ID_WMOM] / ( vars.hy_dens_cell[k] + vars.state[i,k,ID_DENS] )
            var_local[i,k,ID_RHOT] = (( vars.state[i,k,ID_RHOT] + vars.hy_dens_theta_cell[k] ) /
                                      ( vars.hy_dens_cell[k] + vars.state[i,k,ID_DENS] ) -
                                      vars.hy_dens_theta_cell[k] / vars.hy_dens_cell[k])
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
       if ( etime == 0.0 )

          # Open NetCDF output file with a create mode
          ds = Dataset(OUTFILE,"c")

          defDim(ds,"t",Inf)
          defDim(ds,"x",NX_GLOB)
          defDim(ds,"z",NZ_GLOB)

          nc_var = defVar(ds,"t",FLOAT,("t",))
          nc_var[nt] = etime
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

function finalize!(vars::VarsType{FLOAT})

    #println(axes(state))
    
end

# invoke main function
main(ARGS)
