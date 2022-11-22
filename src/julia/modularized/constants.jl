
#### importing modules #####

import OffsetArrays.OffsetArray,
       OffsetArrays.OffsetVector

import ArgParse.ArgParseSettings,
       ArgParse.parse_args,
       ArgParse.@add_arg_table!

import MPI.COMM_WORLD,
       MPI.Comm_rank,
       MPI.Comm_size
    
##### basic constant definitions #####

const COMM   = COMM_WORLD
const NRANKS = Comm_size(COMM)
const MYRANK = Comm_rank(COMM)

const FLOAT32 = Float32
const FLOAT64 = Float64
const FLOAT   = FLOAT64

const INT32   = Int32
const INT64   = Int64
const INTEGER = INT64

##### runtime inputs #####

s = ArgParseSettings()
@add_arg_table! s begin
    "--simtime", "-s"
        help = "simulation time"
        arg_type = FLOAT
        default = 400.0
    "--nx", "-x"
        help = "x-dimension"
        arg_type = INTEGER
        default = 100
    "--nz", "-z"
        help = "z-dimension"
        arg_type = INTEGER
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
        arg_type = INTEGER
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
