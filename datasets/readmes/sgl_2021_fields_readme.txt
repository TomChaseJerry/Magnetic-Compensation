HDF5 data fields for 2021 SGL flight data. Data sampled at 10Hz. Includes 
magnetic sensors, inertial navigation system, avionics, and electrical 
currents and voltages. (WGS-84) is altitude above WGS-84 ellipsoid.

================================================================================
       Field    Units    Description
--------------------------------------------------------------------------------
        line        -    line number
      flight        -    flight number
        year        -    year
         doy        -    day of year
          tt        s    fiducial seconds past midnight UTC
       utm_x        m    x-coordinate, WGS-84 UTM zone 18N
       utm_y        m    y-coordinate, WGS-84 UTM zone 18N
       utm_z        m    z-coordinate, GPS altitude (WGS-84)
         msl        m    z-coordinate, GPS altitude above EGM2008 Geoid
         lat      deg    latitude, WGS-84
         lon      deg    longitude, WGS-84
        baro        m    barometric altimeter
   ins_pitch      deg    INS-computed aircraft pitch
    ins_roll      deg    INS-computed aircraft roll
     ins_yaw      deg    INS-computed aircraft yaw
     diurnal       nT    measured diurnal
     mag_1_c       nT    Mag 1: compensated magnetic field
   mag_1_lag       nT    Mag 1: lag-corrected magnetic field
    mag_1_dc       nT    Mag 1: diurnal-corrected magnetic field
  mag_1_igrf       nT    Mag 1: IGRF & diurnal-corrected magnetic field
    mag_1_uc       nT    Mag 1: uncompensated magnetic field
    mag_2_uc       nT    Mag 2: uncompensated magnetic field
    mag_3_uc       nT    Mag 3: uncompensated magnetic field
    mag_4_uc       nT    Mag 4: uncompensated magnetic field
    mag_5_uc       nT    Mag 5: uncompensated magnetic field
    flux_a_x       nT    Flux A: fluxgate x-axis
    flux_a_y       nT    Flux A: fluxgate y-axis
    flux_a_z       nT    Flux A: fluxgate z-axis
    flux_a_t       nT    Flux A: fluxgate total
    flux_b_x       nT    Flux B: fluxgate x-axis
    flux_b_y       nT    Flux B: fluxgate y-axis
    flux_b_z       nT    Flux B: fluxgate z-axis
    flux_b_t       nT    Flux B: fluxgate total
    flux_c_x       nT    Flux C: fluxgate x-axis
    flux_c_y       nT    Flux C: fluxgate y-axis
    flux_c_z       nT    Flux C: fluxgate z-axis
    flux_c_t       nT    Flux C: fluxgate total
    flux_d_x       nT    Flux D: fluxgate x-axis
    flux_d_y       nT    Flux D: fluxgate y-axis
    flux_d_z       nT    Flux D: fluxgate z-axis
    flux_d_t       nT    Flux D: fluxgate total
   ins_acc_x    m/s^2    INS x-acceleration
   ins_acc_y    m/s^2    INS y-acceleration
   ins_acc_z    m/s^2    INS z-acceleration
  ins_wander      rad    INS-computed wander angle (ccw from north)
     ins_lat      rad    INS-computed latitude
     ins_lon      rad    INS-computed longitude
     ins_alt        m    INS-computed altitude (WGS-84)
      ins_vn      m/s    INS-computed north velocity
      ins_vw      m/s    INS-computed west velocity
      ins_vu      m/s    INS-computed vertical (up) velocity
   cur_com_1        A    current sensor: aircraft radio 1
   cur_ac_hi        A    current sensor: air conditioner fan high
   cur_ac_lo        A    current sensor: air conditioner fan low
    cur_tank        A    current sensor: cabin fuel pump
    cur_flap        A    current sensor: flap motor
    cur_strb        A    current sensor: strobe lights
   vol_block        V    voltage sensor: block
    vol_back        V    voltage sensor: backplane
    vol_cabt        V    voltage sensor: cabinet
     vol_fan        V    voltage sensor: cooling fan
================================================================================

Notes on specific flight data fields:

ins_pitch, ins_roll, ins_yaw
- convention: yaw clockwise from north, then pitch up, then roll to starboard

ins_acc_x, ins_acc_y, ins_acc_z
- AIRGrav (airborne gravimeter) system (http://www.sgl.com/Gravity.html)
- 3-axis gimballed INS, run in local-level, wander-angle mechanization
- corrected for temperature, misalignment, and scale factor
- digitizers used for accelerometers are sigma-delta type and require low-pass 
  filtering to increase resolution
- ins_acc_x: nominally horizontal, ccw from north by ins_wander [rad]
- ins_acc_y: nominally horizontal, 90 [deg] ccw from ins_acc_x
- ins_acc_z: nominally vertical (up), includes gravity term (~navigation frame)

ins_alt
- combined GPS/inertial altitude computed in real time, contains lag error

================================================================================

Magnetometer/Fluxgate positions in reference to front seat rail [m]
- positive directions: forward (X), port (Y), up (Z)

  Sensor   Description                      X       Y       Z
-------------------------------------------------------------------
   Mag 1   Tail stinger                  -12.01    0       1.37
   Mag 2   Front cabin, aft of cockpit    -0.60   -0.36    0
   Mag 3   Mid cabin, near INS            -1.28   -0.36    0
   Mag 4   Rear cabin, floor              -3.53    0       0
   Mag 5   Rear cabin, ceiling            -3.79    0       1.20
  Flux A   Mid cabin, near fuel tank      -3.27   -0.60    0
  Flux B   Tail, base of stinger          -8.92    0       0.96
  Flux C   Rear cabin, port               -4.06    0.42    0
  Flux D   Rear cabin, starboard          -4.06   -0.42    0
