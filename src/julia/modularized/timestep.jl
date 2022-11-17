module Timestep

import TimerOutputs.@timeit

import MPI.Waitall!,
       MPI.Request,
       MPI.Irecv!,
       MPI.Isend

import OffsetArrays.OffsetArray,
       OffsetArrays.OffsetVector

using ..Constants: COMM, NRANKS, MYRANK, DATA_SPEC
using ..Constants: I_BEG, NX, NZ, LEFT_RANK, RIGHT_RANK
using ..Constants: K_BEG, HS, STEN_SIZE, NUM_VARS, XLEN, ZLEN, HV_BETA
using ..Constants: CFL, MAX_SPEED, DX, DZ, DT, NQPOINTS, PI, GRAV, CP, CV, RD, P0, C0, GAMMA
using ..Constants: ID_DENS, ID_UMOM, ID_WMOM, ID_RHOT, DIR_X, DIR_Z
using ..Constants: DATA_SPEC_GRAVITY_WAVES
using ..Constants: FLOAT, INTEGER, DATA_SPEC_INJECTION

using ..Variables: state, statetmp, flux, tend, hy_dens_cell, hy_dens_theta_cell
using ..Variables: hy_dens_int, hy_dens_theta_int, hy_pressure_int
using ..Variables: sendbuf_l, sendbuf_r, recvbuf_l, recvbuf_r, etime

#Performs a single dimensionally split time step using a simple low-storate three-stage Runge-Kutta time integrator
#The dimensional splitting is a second-order-accurate alternating Strang splitting in which the
#order of directions is alternated each time step.
#The Runge-Kutta method used here is defined as follows:
# q*     = q[n] + dt/3 * rhs(q[n])
# q**    = q[n] + dt/2 * rhs(q*  )
# q[n+1] = q[n] + dt/1 * rhs(q** )
function perform_timestep!(state::OffsetArray{FLOAT, 3, Array{FLOAT, 3}},
                   statetmp::OffsetArray{FLOAT, 3, Array{FLOAT, 3}},
                   flux::Array{FLOAT, 3},
                   tend::Array{FLOAT, 3},
                   dt::FLOAT,
                   recvbuf_l::Array{FLOAT, 3},
                   recvbuf_r::Array{FLOAT, 3},
                   sendbuf_l::Array{FLOAT, 3},
                   sendbuf_r::Array{FLOAT, 3},
                   hy_dens_cell::OffsetVector{FLOAT, Vector{FLOAT}},
                   hy_dens_theta_cell::OffsetVector{FLOAT, Vector{FLOAT}},
                   hy_dens_int::Vector{FLOAT},
                   hy_dens_theta_int::Vector{FLOAT},
                   hy_pressure_int::Vector{FLOAT})
    
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
function semi_discrete_step!(state_init::OffsetArray{FLOAT, 3, Array{FLOAT, 3}},
                    state_forcing::OffsetArray{FLOAT, 3, Array{FLOAT, 3}},
                    state_out::OffsetArray{FLOAT, 3, Array{FLOAT, 3}},
                    dt::FLOAT,
                    dir::INTEGER,
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
        @timeit etime "halo_x" set_halo_values_x!(state_forcing, recvbuf_l, recvbuf_r, sendbuf_l,
                           sendbuf_r, hy_dens_cell, hy_dens_theta_cell)

        #Compute the time tendencies for the fluid state in the x-direction
        @timeit etime "tend_x" compute_tendencies_x!(state_forcing,flux,tend,dt, hy_dens_cell, hy_dens_theta_cell)

        
    elseif dir == DIR_Z
        #Set the halo values for this MPI task's fluid state in the z-direction
        @timeit etime "halo_z" set_halo_values_z!(state_forcing, hy_dens_cell, hy_dens_theta_cell)
        
        #Compute the time tendencies for the fluid state in the z-direction
        @timeit etime "tend_z" compute_tendencies_z!(state_forcing,flux,tend,dt,
                    hy_dens_int, hy_dens_theta_int, hy_pressure_int)
        
    end
  
    #Apply the tendencies to the fluid state
    @timeit etime "update" for ll in 1:NUM_VARS
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

export perform_timestep!

end
