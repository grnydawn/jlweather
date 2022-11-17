module Variables

import TimerOutputs.TimerOutput

import OffsetArrays.OffsetArray,
       OffsetArrays.OffsetVector

using ..Constants: FLOAT, NX, NZ, HS, NUM_VARS, NQPOINTS, I_BEG, K_BEG
using ..Constants: DX, DZ, qpoints, qweights, DATA_SPEC, DATA_SPEC_COLLISION
using ..Constants: DATA_SPEC_THERMAL, DATA_SPEC_GRAVITY_WAVES
using ..Constants: DATA_SPEC_DENSITY_CURRENT, DATA_SPEC_INJECTION
using ..Constants: GRAV, CP, P0, RD, C0, GAMMA
using ..Constants: ID_DENS, ID_UMOM, ID_WMOM, ID_RHOT

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
    
const etime = TimerOutput()

export state, statetmp, flux, tend, hy_dens_cell, hy_dens_theta_cell
export hy_dens_int, hy_dens_theta_int, hy_pressure_int
export sendbuf_l, sendbuf_r, recvbuf_l, recvbuf_r, etime

end
